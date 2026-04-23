#!/usr/bin/env python
"""Pauli-expectation AC + low-k marginal estimator CLI.

Subcommands:
  estimate  -- run estimator on a single dataset's checkpoint
  validate  -- compare estimator to exact ground truth (Phase 2 harness)
  all       -- sequentially dispatch estimator across the 5 big-n datasets

Example:
    python scripts/pauli_estimator_investigation.py estimate --dataset 8_blobs \\
        --ckpt results/iqp_mmd/8_blobs/checkpoint.npz \\
        --num-pauli-samples 5000 --n-expval-samples 2000 --seed 666
"""
from __future__ import annotations

import os

# JAX env defaults -- set BEFORE any jax import. Users may override via env.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from iqpopt import IqpSimulator
from iqpopt.utils import local_gates

from iqp_bp.estimators import (
    ACResult,
    MarginalResult,
    pauli_ac_estimator,
    pauli_marginal_mismatch,
)
from iqp_bp.experiments import check_anti_concentration
from iqp_bp.experiments.run_validation import load_iqp_checkpoint
from iqp_bp.iqp.model import IQPModel
from iqp_mmd.config import DATASET_NAMES, DatasetPaths
from iqp_mmd.datasets.loaders import load_csv_dataset  # pd.read_csv wrapper

# --- Direct-fd logger (Windows-safe); mirrors investigate_iqp_mmd_ac.py:25-36.
_LOG_FD: int | None = None
_orig_print = print


def _log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _orig_print(msg, flush=True)
    if _LOG_FD is not None:
        os.write(_LOG_FD, (msg + "\n").encode("utf-8"))


# Replace module-level print so existing print(...) calls route through the fd.
print = _log  # noqa: A001


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "pauli_estimator_datasets.yaml"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "pauli_scale"
DEFAULT_DATA_ROOT = REPO_ROOT  # DatasetPaths(base_dir=REPO_ROOT).datasets_dir -> REPO_ROOT/datasets


def _load_dataset_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _open_progress_log(out_dir: Path) -> None:
    """Set the module-level _LOG_FD to a direct os.open handle on <out_dir>/progress.log."""
    global _LOG_FD
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FD = os.open(
        str(out_dir / "progress.log"),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    )


def _rel_to_repo(path: Path) -> str:
    """Return path as a string relative to REPO_ROOT when possible, else absolute str.

    Users may pass --out as a relative path; resolve to absolute before attempting
    relative_to(REPO_ROOT) so the JSON schema field is always well-formed.
    """
    abs_path = Path(path).resolve()
    try:
        return str(abs_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(abs_path)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_simulator_from_checkpoint(
    ckpt_path: Path,
    max_weight: int,
    spin_sym: bool,
    n_qubits: int | None = None,
) -> tuple[IqpSimulator, np.ndarray, IQPModel, dict[str, Any]]:
    """Load .npz checkpoint and construct an iqpopt.IqpSimulator.

    Resolution order for (max_weight, spin_sym):
      1. CLI flags (caller-provided).
      2. YAML config defaults (caller resolves against dataset name before calling this).
      3. Never from the .npz -- research confirms checkpoints omit spin_sym; they
         do carry n_qubits/max_weight as npz scalars but we treat the YAML/CLI
         values as authoritative at estimate time.

    Asserts len(theta) == len(local_gates(n_qubits, max_weight)); raises with both
    lengths in the message if mismatched (research Gotcha: "gates length != theta").

    Returns: (simulator, theta_np, model_bp, meta). `meta` is the dict emitted by
    load_iqp_checkpoint -- contains every non-G/theta npz key (e.g. n_qubits,
    max_weight, seed, source, optional provenance_json).
    """
    model_bp, meta = load_iqp_checkpoint(str(ckpt_path))
    theta = np.asarray(model_bp.theta, dtype=np.float64)
    # IQPModel infers n_qubits from G.shape[1] (see src/iqp_bp/iqp/model.py:43).
    n = int(n_qubits) if n_qubits is not None else int(model_bp.n)
    gates = local_gates(n_qubits=n, max_weight=max_weight)
    assert len(theta) == len(gates), (
        f"theta has {len(theta)} entries but local_gates(n={n}, "
        f"max_weight={max_weight}) produced {len(gates)} gates"
    )
    sim = IqpSimulator(n_qubits=n, gates=gates, sparse=False, spin_sym=spin_sym)
    return sim, theta, model_bp, meta


def _build_estimator_json(
    tag: str,
    config: dict[str, Any],
    ac: ACResult,
    marginals: MarginalResult,
    ckpt_path: Path,
    raw_Y_path: Path,
    wall_time_sec: float,
) -> dict[str, Any]:
    """Assemble the per-dataset estimator.json payload.

    Schema (LOCKED -- consumed by Phase 2 validation harness and Phase 4 plotting):

    {
      "tag": str,
      "config": {...all CLI flags and derived values...},
      "ac": {
        "scaled_ss_hat": float,
        "sigma": float,
        "by_weight": {str(k): {"contribution_to_total": float,
                               "contribution_sigma": float,
                               "count": int}, ...},
        "effective_m": int,
        "max_squared_expval": float,
        "max_support": [int, int, ...],   # length n_qubits
        "raw_Y_samples_path": str          # relative path to .npy; not inlined
      },
      "marginals": {
        str(k): {"mean_tv": float, "max_tv": float, "sigma": float,
                 "num_subsets": int, "per_subset_tvs": [float, ...]},
        ...
      },
      "provenance": {
        "wall_time_sec": float,
        "seed": int,
        "num_pauli_samples": int,
        "n_expval_samples": int,
        "iqpopt_version": str,
        "checkpoint_sha256": str,
        "checkpoint_path": str,
        "python_version": str,
        "platform": str
      }
    }
    """
    import iqpopt
    import platform as _platform
    return {
        "tag": tag,
        "config": config,
        "ac": {
            "scaled_ss_hat": float(ac.scaled_ss_hat),
            "sigma": float(ac.sigma),
            "by_weight": {str(k): v for k, v in ac.by_weight.items()},
            "effective_m": int(ac.effective_m),
            "max_squared_expval": float(ac.max_squared_expval),
            "max_support": list(map(int, ac.max_support)),
            "raw_Y_samples_path": _rel_to_repo(raw_Y_path),
        },
        "marginals": {
            str(k): {
                "mean_tv": float(v["mean_tv"]),
                "max_tv": float(v["max_tv"]),
                "sigma": float(v["sigma"]),
                "num_subsets": int(v["num_subsets"]),
                "per_subset_tvs": [float(x) for x in v["per_subset_tvs"]],
            }
            for k, v in marginals.per_k.items()
        },
        "provenance": {
            "wall_time_sec": float(wall_time_sec),
            "seed": int(config["seed"]),
            "num_pauli_samples": int(config["num_pauli_samples"]),
            "n_expval_samples": int(config["n_expval_samples"]),
            "iqpopt_version": getattr(iqpopt, "__version__", "unknown"),
            "checkpoint_sha256": _sha256_of_file(ckpt_path),
            "checkpoint_path": str(ckpt_path),
            "python_version": _platform.python_version(),
            "platform": _platform.platform(),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pauli_estimator_investigation",
        description="Parseval-MC AC estimator + low-k marginal mismatch CLI.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- estimate ---
    pe = sub.add_parser("estimate", help="Run estimator on one dataset's checkpoint.")
    pe.add_argument("--dataset", required=True, choices=DATASET_NAMES,
                    help="Dataset name; used to look up max_weight/spin_sym from config YAML.")
    pe.add_argument("--ckpt", type=Path, default=None,
                    help="Path to .npz checkpoint. Default: results/pauli_scale/<dataset>/checkpoint.npz")
    pe.add_argument("--out", type=Path, default=None,
                    help="Output dir. Default: results/pauli_scale/<dataset>/")
    pe.add_argument("--max-weight", type=int, default=None,
                    help="Override YAML default for local_gates max_weight.")
    spin = pe.add_mutually_exclusive_group()
    spin.add_argument("--spin-sym", dest="spin_sym", action="store_true", default=None)
    spin.add_argument("--no-spin-sym", dest="spin_sym", action="store_false")
    pe.add_argument("--num-pauli-samples", type=int, default=2000)
    pe.add_argument("--n-expval-samples", type=int, default=2000)
    pe.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    pe.add_argument("--subsets-per-order", type=int, default=128)
    pe.add_argument("--seed", type=int, default=666)
    pe.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pe.set_defaults(func=_cmd_estimate)

    # --- validate ---
    pv = sub.add_parser(
        "validate",
        help=(
            "Compare estimator to exact ground truth (Phase 2 harness). "
            "NOTE: the --no-spin-sym path uses IQPModel.probability_vector_exact, "
            "which is O(2**n) memory and REQUIRES n<=20. For n>20, use --spin-sym "
            "so ground truth routes through iqpopt.probs(theta) instead."
        ),
    )
    pv.add_argument("--ckpt", type=Path, required=True)
    pv.add_argument("--dataset", choices=DATASET_NAMES, default=None,
                    help="Optional; used for default out dir and config lookup.")
    pv.add_argument("--max-weight", type=int, default=None)
    sv = pv.add_mutually_exclusive_group()
    sv.add_argument("--spin-sym", dest="spin_sym", action="store_true", default=None)
    sv.add_argument("--no-spin-sym", dest="spin_sym", action="store_false")
    pv.add_argument("--out", type=Path, default=None)
    pv.add_argument("--num-pauli-samples", type=int, default=5000)
    pv.add_argument("--n-expval-samples", type=int, default=5000)
    pv.add_argument("--seed", type=int, default=666)
    pv.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pv.set_defaults(func=_cmd_validate)

    # --- all ---
    pa = sub.add_parser("all", help="Dispatch estimate over the 5 big-n datasets sequentially.")
    pa.add_argument("--num-pauli-samples", type=int, default=2000)
    pa.add_argument("--n-expval-samples", type=int, default=2000)
    pa.add_argument("--seed", type=int, default=666)
    pa.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pa.add_argument("--skip-missing-checkpoint", action="store_true", default=True,
                    help="If a dataset has no checkpoint.npz, log and continue (default).")
    pa.set_defaults(func=_cmd_all)

    return p


def _namespace_to_jsonable(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert argparse Namespace into a JSON-serializable dict (paths -> str)."""
    raw = vars(args).copy()
    raw.pop("func", None)
    for k, v in list(raw.items()):
        if isinstance(v, Path):
            raw[k] = str(v)
    if extra:
        raw.update(extra)
    return raw


def _cmd_estimate(args: argparse.Namespace) -> int:
    # 1. Load YAML config and resolve per-dataset defaults.
    cfg = _load_dataset_config(args.config)
    if args.dataset not in cfg["datasets"]:
        print(f"ERROR: dataset {args.dataset!r} missing from {args.config}")
        return 2
    dataset_cfg = cfg["datasets"][args.dataset]

    # 2. Resolve max_weight / spin_sym -- CLI overrides YAML.
    max_weight = args.max_weight if args.max_weight is not None else int(dataset_cfg["max_weight"])
    spin_sym = args.spin_sym if args.spin_sym is not None else bool(dataset_cfg["spin_sym"])

    # 3. Resolve paths.
    ckpt = args.ckpt if args.ckpt is not None else DEFAULT_OUT_ROOT / args.dataset / "checkpoint.npz"
    out_dir = args.out if args.out is not None else DEFAULT_OUT_ROOT / args.dataset
    out_dir = Path(out_dir)

    # 4. Open progress log + banner.
    _open_progress_log(out_dir)
    print(f"=== estimate dataset={args.dataset} ===")
    print(f"  ckpt={ckpt}")
    print(f"  out_dir={out_dir}")
    print(f"  resolved: max_weight={max_weight}, spin_sym={spin_sym}")
    print(f"  num_pauli_samples={args.num_pauli_samples}  n_expval_samples={args.n_expval_samples}")
    print(f"  k_values={args.k_values}  subsets_per_order={args.subsets_per_order}  seed={args.seed}")

    # 5. Build IqpSimulator from checkpoint.
    sim, theta, model_bp, meta = _build_simulator_from_checkpoint(
        ckpt, max_weight, spin_sym, dataset_cfg.get("n_qubits")
    )
    print(f"  sim.n_qubits={sim.n_qubits}  theta.shape={theta.shape}  checkpoint_meta={meta}")

    # 6. Load empirical target for marginals (best-effort).
    try:
        csv_path = DatasetPaths(base_dir=REPO_ROOT).train_path(args.dataset)
        target_empirical = load_csv_dataset(csv_path, delimiter=",", header=None).astype(np.uint8)
        # Defensive: some datasets may ship {-1, +1} encoded -- normalize to {0, 1}.
        if target_empirical.min() < 0:
            target_empirical = ((target_empirical + 1) // 2).astype(np.uint8)
        assert target_empirical.shape[1] == sim.n_qubits, (
            f"target_empirical has {target_empirical.shape[1]} columns but simulator has {sim.n_qubits} qubits"
        )
        print(f"  target_empirical loaded from {csv_path}  shape={target_empirical.shape}")
    except Exception as e:
        print(f"WARN: empirical target load failed ({type(e).__name__}: {e}); skipping marginals")
        target_empirical = None

    # 7. Run AC estimator.
    t0 = time.perf_counter()
    print("  running pauli_ac_estimator ...")
    ac = pauli_ac_estimator(
        sim, theta,
        num_pauli_samples=args.num_pauli_samples,
        n_expval_samples=args.n_expval_samples,
        seed=args.seed,
    )
    print(f"  AC done  scaled_ss_hat={ac.scaled_ss_hat:.6f}  sigma={ac.sigma:.6f}  "
          f"effective_m={ac.effective_m}  max_sq_expval={ac.max_squared_expval:.6e}")

    # 8. Run marginal mismatch if target available.
    if target_empirical is not None:
        print("  running pauli_marginal_mismatch ...")
        marginals = pauli_marginal_mismatch(
            sim, theta, target_empirical,
            k_values=tuple(args.k_values),
            num_subsets=args.subsets_per_order,
            n_expval_samples=args.n_expval_samples,
            seed=args.seed,
        )
        for k, stats in marginals.per_k.items():
            print(f"  marginals k={k}: mean_tv={stats['mean_tv']:.4f} +- "
                  f"{stats['sigma']:.4f}  max_tv={stats['max_tv']:.4f}  n_subsets={stats['num_subsets']}")
    else:
        marginals = MarginalResult(per_k={}, metadata={"error": "target_empirical unavailable"})

    wall = time.perf_counter() - t0

    # 9. Dump raw Y samples to sibling .npy.
    raw_Y_path = out_dir / "raw_Y.npy"
    np.save(raw_Y_path, ac.raw_Y_samples)
    print(f"  raw Y samples -> {raw_Y_path}")

    # 10. Assemble JSON payload.
    cfg_blob = _namespace_to_jsonable(
        args,
        extra={
            "resolved_max_weight": int(max_weight),
            "resolved_spin_sym": bool(spin_sym),
            "resolved_ckpt": str(ckpt),
            "resolved_out_dir": str(out_dir),
        },
    )
    tag = f"estimate/{args.dataset}"
    payload = _build_estimator_json(tag, cfg_blob, ac, marginals, ckpt, raw_Y_path, wall)

    out_json = out_dir / "estimator.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {out_json}")
    print(f"scaled_ss_hat = {ac.scaled_ss_hat:.4f} +- {ac.sigma:.4f}  (wall={wall:.1f}s)")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    # Resolve max_weight / spin_sym. If --dataset provided, use YAML defaults as fallback.
    dataset_cfg: dict[str, Any] | None = None
    if args.dataset is not None:
        cfg = _load_dataset_config(args.config)
        dataset_cfg = cfg["datasets"].get(args.dataset)

    if args.max_weight is not None:
        max_weight = args.max_weight
    elif dataset_cfg is not None:
        max_weight = int(dataset_cfg["max_weight"])
    else:
        print("ERROR: --max-weight required when --dataset is not given")
        return 2

    if args.spin_sym is not None:
        spin_sym = args.spin_sym
    elif dataset_cfg is not None:
        spin_sym = bool(dataset_cfg["spin_sym"])
    else:
        print("ERROR: --spin-sym / --no-spin-sym required when --dataset is not given")
        return 2

    # Resolve out dir.
    if args.out is not None:
        out_dir = Path(args.out)
    elif args.dataset is not None:
        out_dir = DEFAULT_OUT_ROOT / args.dataset / "validate"
    else:
        out_dir = DEFAULT_OUT_ROOT / args.ckpt.stem / "validate"
    _open_progress_log(out_dir)

    print(f"=== validate dataset={args.dataset} ckpt={args.ckpt} ===")
    print(f"  resolved: max_weight={max_weight}, spin_sym={spin_sym}")

    n_qubits_hint = dataset_cfg.get("n_qubits") if dataset_cfg is not None else None
    sim, theta, model_bp, meta = _build_simulator_from_checkpoint(
        args.ckpt, max_weight, spin_sym, n_qubits_hint
    )
    n = sim.n_qubits
    print(f"  n_qubits={n}  theta.shape={theta.shape}  meta={meta}")

    # Ground-truth routing with explicit n-range guard.
    t0 = time.perf_counter()
    if spin_sym:
        print("  ground truth via iqpopt.IqpSimulator.probs(theta)  [spin_sym=True]")
        q_exact = np.asarray(sim.probs(jnp.asarray(theta)))
    else:
        assert n <= 20, (
            f"validate --no-spin-sym: ground-truth path requires n<=20 "
            f"(IQPModel.probability_vector_exact is O(2**n)); got n={n}. "
            f"Use --spin-sym to route ground truth via iqpopt.probs(theta), "
            f"or reduce the checkpoint."
        )
        print("  ground truth via IQPModel.probability_vector_exact  [spin_sym=False, n<=20]")
        q_exact = model_bp.probability_vector_exact(max_qubits=20)
    t_truth = time.perf_counter() - t0
    truth_ac = check_anti_concentration(q_exact)
    truth_scaled_ss = float(truth_ac["scaled_second_moment"])
    print(f"  truth_scaled_ss={truth_scaled_ss:.6f}  (truth path {t_truth:.1f}s)")

    # Run AC estimator.
    t0 = time.perf_counter()
    ac = pauli_ac_estimator(
        sim, theta,
        num_pauli_samples=args.num_pauli_samples,
        n_expval_samples=args.n_expval_samples,
        seed=args.seed,
    )
    wall = time.perf_counter() - t0
    delta = ac.scaled_ss_hat - truth_scaled_ss
    within_2 = bool(abs(delta) <= 2.0 * ac.sigma)
    within_3 = bool(abs(delta) <= 3.0 * ac.sigma)
    print(f"  estimated={ac.scaled_ss_hat:.6f} +- {ac.sigma:.6f}  delta={delta:+.6f}  "
          f"within_2sigma={within_2}  within_3sigma={within_3}")

    # Dump raw Y samples (useful for validation post-mortem).
    raw_Y_path = out_dir / "raw_Y.npy"
    np.save(raw_Y_path, ac.raw_Y_samples)

    cfg_blob = _namespace_to_jsonable(
        args,
        extra={
            "resolved_max_weight": int(max_weight),
            "resolved_spin_sym": bool(spin_sym),
            "resolved_out_dir": str(out_dir),
        },
    )
    import iqpopt
    import platform as _platform
    payload = {
        "tag": f"validate/{args.dataset or Path(args.ckpt).stem}",
        "truth_scaled_ss": truth_scaled_ss,
        "estimated_scaled_ss_hat": float(ac.scaled_ss_hat),
        "estimated_sigma": float(ac.sigma),
        "delta": float(delta),
        "within_2sigma": within_2,
        "within_3sigma": within_3,
        "config": cfg_blob,
        "provenance": {
            "wall_time_sec": float(wall),
            "truth_wall_time_sec": float(t_truth),
            "seed": int(args.seed),
            "num_pauli_samples": int(args.num_pauli_samples),
            "n_expval_samples": int(args.n_expval_samples),
            "iqpopt_version": getattr(iqpopt, "__version__", "unknown"),
            "checkpoint_sha256": _sha256_of_file(Path(args.ckpt)),
            "checkpoint_path": str(args.ckpt),
            "raw_Y_samples_path": _rel_to_repo(raw_Y_path),
            "python_version": _platform.python_version(),
            "platform": _platform.platform(),
        },
    }

    out_json = out_dir / "validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {out_json}")
    return 0 if within_2 else 1


def _cmd_all(args: argparse.Namespace) -> int:
    cfg = _load_dataset_config(args.config)
    big_n = list(cfg["big_n"])
    assert set(big_n) <= set(DATASET_NAMES), (
        f"big_n datasets {set(big_n) - set(DATASET_NAMES)} not in DATASET_NAMES"
    )

    root_out = DEFAULT_OUT_ROOT
    root_out.mkdir(parents=True, exist_ok=True)
    # Use a top-level progress log for the --all run.
    _open_progress_log(root_out)
    print(f"=== all: dispatching over {big_n} ===")

    summary: dict[str, dict[str, Any]] = {}
    for name in big_n:
        ds_out = root_out / name
        default_ckpt = ds_out / "checkpoint.npz"
        if not default_ckpt.exists():
            msg = f"SKIP {name}: no checkpoint at {default_ckpt}"
            print(msg)
            if args.skip_missing_checkpoint:
                summary[name] = {"status": "skipped", "error": None, "json_path": None}
                continue
            summary[name] = {"status": "failed", "error": msg, "json_path": None}
            continue

        sub_args = argparse.Namespace(
            command="estimate",
            dataset=name,
            ckpt=None,
            out=None,
            max_weight=None,
            spin_sym=None,
            num_pauli_samples=args.num_pauli_samples,
            n_expval_samples=args.n_expval_samples,
            k_values=[1, 2, 3, 4, 6, 8],
            subsets_per_order=128,
            seed=args.seed,
            config=args.config,
            func=_cmd_estimate,
        )
        try:
            rc = _cmd_estimate(sub_args)
            if rc == 0:
                summary[name] = {
                    "status": "ok",
                    "error": None,
                    "json_path": _rel_to_repo(ds_out / "estimator.json"),
                }
            else:
                summary[name] = {
                    "status": "failed",
                    "error": f"estimate returned rc={rc}",
                    "json_path": None,
                }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"FAIL {name}: {type(e).__name__}: {e}\n{tb}")
            summary[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "json_path": None}

    out_json = root_out / "all_runs.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")

    # Return 0 unless every dataset failed.
    all_failed = all(v["status"] == "failed" for v in summary.values())
    return 1 if all_failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
