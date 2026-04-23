#!/usr/bin/env python
"""Phase 2 n=16 validation harness.

Gates Phase 3 (large-n runs) on estimator-vs-exact agreement at n=16.

VAL-01 (scaled second moment):
    Subprocess the existing
    ``scripts/pauli_estimator_investigation.py validate`` CLI for each
    (dataset, seed) pair; parse the emitted ``validation.json``.

VAL-02 (low-k marginal mismatch, k in {1, 2, 4}):
    Computed in-process. The ``validate`` subcommand does not emit marginal
    fields (see 02-RESEARCH Sec 1). The harness builds the same simulator the
    CLI does, calls ``pauli_marginal_mismatch`` for the estimator side, and
    imports ``per_order_marginal_mismatch`` from
    ``scripts/investigate_iqp_mmd_ac.py`` for the exact reference.

Both criteria are aggregated across 3 seeds (666, 667, 668) into a
combined JSON at ``results/pauli_estimator_validation_n16.json`` with
``{meta, criteria, overall_pass, overall_tier}`` shape.

Invocation:
    python scripts/validate_n16.py
    python scripts/validate_n16.py --config configs/validate_n16_checkpoints.yaml
    python scripts/validate_n16.py --skip-val02   # VAL-01 only (debug)
    python scripts/validate_n16.py --dry-run      # print planned argv, exit 0
"""
from __future__ import annotations

import os

# JAX env defaults mirror the CLI (set BEFORE any jax import).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# scripts/ on sys.path so ``from investigate_iqp_mmd_ac import ...`` works.
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Heavy imports (jax / iqpopt / iqp_bp) are deferred into run_val02 so that
# --dry-run and --skip-val02 paths remain fast and avoid JAX init.
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "validate_n16_checkpoints.yaml"

# Dataset-tag -> generator-tag mapping. The CLI / YAML use {2D_ising, 8_blobs};
# investigate_iqp_mmd_ac.py's sample generators use {ising, spin_blobs}. Keep
# the mapping explicit here so downstream reader can audit which generator was
# used to rebuild X_target (the training CSVs at DatasetPaths.train_path(...)
# are not populated locally -- STATE.md blocker).
GEN_TAG_FOR_DATASET = {
    "2D_ising": "ising",
    "8_blobs": "spin_blobs",
}
# Authoritative num_train used by investigate_iqp_mmd_ac.py --num-train default
# and matched to what the checkpoints in results/iqp_mmd_ac_investigation were
# trained on (checkpoint seed=666).
TARGET_NUM_TRAIN = 5000
TARGET_SEED = 666

log = logging.getLogger("validate_n16")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    """Streaming sha256, 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    """Best-effort git HEAD / branch / dirty detection. Never raises."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        head = "unknown"
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        branch = "unknown"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
        ).stdout
        dirty = bool(status.strip())
    except Exception:
        dirty = False
    return {"head": head, "branch": branch, "dirty": dirty}


def tier_for(z: float) -> str:
    """pass <= 2 sigma; warn 2--3 sigma; fail > 3 sigma."""
    az = abs(float(z))
    if az <= 2.0:
        return "pass"
    if az <= 3.0:
        return "warn"
    return "fail"


_TIER_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def worst_tier(tiers: Iterable[str]) -> str:
    """Max-by-severity over an iterable of tier strings."""
    tiers = list(tiers)
    if not tiers:
        return "pass"
    return max(tiers, key=lambda t: _TIER_ORDER.get(t, 0))


def _coerce_num_subsets(raw: dict) -> dict[int, int]:
    """Normalize YAML-parsed num_subsets keys (may be str or int) to int."""
    return {int(k): int(v) for k, v in raw.items()}


# --------------------------------------------------------------------------
# VAL-01: subprocess the existing validate CLI and parse its JSON output.
# --------------------------------------------------------------------------
def run_val01(
    dataset: str,
    ckpt: Path,
    seed: int,
    cfg: dict[str, Any],
    out_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Invoke `pauli_estimator_investigation.py validate` and parse the JSON.

    Returns a dict with the per-seed VAL-01 row. Does NOT aggregate across
    seeds; aggregation is done in Task 3.
    """
    seed_dir = out_root / dataset / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable,
        "scripts/pauli_estimator_investigation.py",
        "validate",
        "--dataset", dataset,
        "--ckpt", str(ckpt),
        "--num-pauli-samples", str(cfg["val01"]["num_pauli_samples"]),
        "--n-expval-samples", str(cfg["val01"]["n_expval_samples"]),
        "--seed", str(seed),
        "--out", str(seed_dir),
    ]

    if dry_run:
        log.info("[dry-run] VAL-01 argv: %s", " ".join(argv))
        return {
            "dataset": dataset,
            "criterion": "val01_scaled_ss",
            "seed": int(seed),
            "cli_argv": argv,
            "dry_run": True,
        }

    log.info("VAL-01 invoking: %s", " ".join(argv))
    t0 = time.perf_counter()
    completed = subprocess.run(argv, cwd=str(REPO_ROOT), check=False)
    wall = time.perf_counter() - t0
    rc = completed.returncode

    json_path = seed_dir / "validation.json"
    if not json_path.exists():
        raise RuntimeError(
            f"VAL-01 subprocess for dataset={dataset} seed={seed} did not "
            f"produce {json_path} (rc={rc}); cannot continue."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    row = {
        "dataset": dataset,
        "criterion": "val01_scaled_ss",
        "seed": int(seed),
        "exact": float(payload["truth_scaled_ss"]),
        "estimate": float(payload["estimated_scaled_ss_hat"]),
        "sigma": float(payload["estimated_sigma"]),
        "within_2sigma": bool(payload["within_2sigma"]),
        "within_3sigma": bool(payload["within_3sigma"]),
        "wall_time_s": float(payload["provenance"]["wall_time_sec"]),
        "harness_wall_time_s": float(wall),
        "cli_returncode": int(rc),
        "cli_argv": argv,
        "validation_json_path": str(json_path.relative_to(REPO_ROOT)),
    }

    # Sanity floor: if the CLI itself reports within_2sigma=False, surface it
    # immediately. Task 3 will compute the aggregate tier.
    if not row["within_2sigma"]:
        log.warning(
            "VAL-01 sanity-floor breach: dataset=%s seed=%d within_2sigma=False "
            "(estimate=%.6f exact=%.6f sigma=%.6f)",
            dataset, seed, row["estimate"], row["exact"], row["sigma"],
        )

    return row


# --------------------------------------------------------------------------
# VAL-02: in-process marginal-mismatch comparison.
# --------------------------------------------------------------------------
def _build_x_target(dataset: str, n: int) -> np.ndarray:
    """Rebuild the target samples the checkpoint was trained on.

    Deviation from plan text: the plan's VAL-02 path calls
    ``DatasetPaths(REPO_ROOT).train_path(dataset)`` to load a training CSV, but
    those CSVs are not populated locally (see STATE.md blockers and
    ``_cmd_estimate`` in pauli_estimator_investigation.py which falls through
    on FileNotFoundError). The checkpoints under
    ``results/iqp_mmd_ac_investigation/checkpoints/`` were produced by
    ``scripts/investigate_iqp_mmd_ac.py`` with ``--num-train 5000 --seed 666``,
    which generates X synthetically via gen_ising_gibbs / gen_spin_blobs. We
    regenerate from the same generator + seed so X_target matches the
    distribution the checkpoint was trained on.
    """
    from investigate_iqp_mmd_ac import gen_ising_gibbs, gen_spin_blobs

    gen_tag = GEN_TAG_FOR_DATASET[dataset]
    if gen_tag == "ising":
        width = int(round(np.sqrt(n)))
        assert width * width == n, f"ising expects square n, got n={n}"
        X = gen_ising_gibbs(
            width=width, temperature=2.5, num_samples=TARGET_NUM_TRAIN,
            seed=TARGET_SEED, burn_in=2000, thin=10,
        )
    elif gen_tag == "spin_blobs":
        peak_weight = max(1, n // 2)
        n_blobs = min(20, 2 ** (n - 2))
        X = gen_spin_blobs(
            n=n, n_blobs=n_blobs, peak_weight=peak_weight, noise_prob=0.05,
            num_train=TARGET_NUM_TRAIN, seed=TARGET_SEED,
        )
    else:
        raise ValueError(f"no generator mapping for dataset={dataset!r}")

    X = np.asarray(X, dtype=np.uint8)
    if X.min() < 0:
        # Defensive: mirrors the {-1,+1} -> {0,1} guard at
        # pauli_estimator_investigation.py:341-342.
        X = ((X + 1) // 2).astype(np.uint8)
    assert X.shape == (TARGET_NUM_TRAIN, n), (
        f"X_target shape {X.shape} != expected ({TARGET_NUM_TRAIN}, {n})"
    )
    return X


def run_val02(
    dataset: str,
    ckpt: Path,
    spin_sym: bool,
    seed: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute VAL-02 rows (one per k) for a single (dataset, seed) pair.

    Builds the IqpSimulator the same way the CLI does, calls
    ``pauli_marginal_mismatch`` for the estimator side and
    ``per_order_marginal_mismatch`` (imported from scripts/) for the exact
    reference. Returns a list of rows, one per k in ``cfg['val02']['k_values']``.

    Per pinned_decisions Sec 4: a separate call to ``pauli_marginal_mismatch``
    per k because the estimator takes a scalar ``num_subsets`` across all
    k_values but we want a different budget per k.
    """
    # Deferred heavy imports.
    from iqpopt import IqpSimulator
    from iqpopt.utils import local_gates
    from iqp_bp.estimators.pauli import pauli_marginal_mismatch
    from iqp_bp.experiments.run_validation import load_iqp_checkpoint

    from investigate_iqp_mmd_ac import per_order_marginal_mismatch

    # 1. Load checkpoint and reconstruct simulator.
    model_bp, meta = load_iqp_checkpoint(str(ckpt))
    theta = np.asarray(model_bp.theta, dtype=np.float64)
    n = int(meta.get("n_qubits", model_bp.n))
    max_weight = int(meta.get("max_weight", 4))
    assert n == 16, f"VAL-02 harness pins n=16; got n={n} from {ckpt}"

    gates = local_gates(n_qubits=n, max_weight=max_weight)
    assert len(theta) == len(gates), (
        f"theta has {len(theta)} entries but local_gates(n={n}, "
        f"max_weight={max_weight}) produced {len(gates)} gates"
    )
    sim = IqpSimulator(n_qubits=n, gates=gates, sparse=False, spin_sym=spin_sym)

    # 2. Build X_target (deviation: regenerate from generator, not CSV).
    X_target = _build_x_target(dataset, n)

    # 3. Exact full distribution q_theta via IQPModel.probability_vector_exact.
    # Research Sec 2: this routes through the n<=20 guard; fine at n=16.
    # Used for BOTH datasets regardless of spin_sym because
    # per_order_marginal_mismatch consumes a 2**n vector.
    q_theta = np.asarray(
        model_bp.probability_vector_exact(max_qubits=20), dtype=np.float64
    )

    rows: list[dict[str, Any]] = []

    for k in cfg["val02"]["k_values"]:
        k_int = int(k)
        K = int(cfg["val02"]["num_subsets"][k_int])
        n_expval = int(cfg["val02"]["n_expval_samples"])

        t0 = time.perf_counter()
        est = pauli_marginal_mismatch(
            sim, theta, X_target,
            k_values=(k_int,),
            num_subsets=K,
            n_expval_samples=n_expval,
            seed=int(seed),
        )
        est_wall = time.perf_counter() - t0

        per_k_est = est.per_k[k_int]
        mean_tv_est = float(per_k_est["mean_tv"])
        sigma_est = float(per_k_est["sigma"])
        est_num_subsets = int(per_k_est["num_subsets"])

        # per_order_marginal_mismatch returns {k: {"mean_tv", "max_tv", ...}}.
        # Flat keyed on k -- NOT {"mean_tv": {k: ...}} as the plan text
        # speculated. Verified at scripts/investigate_iqp_mmd_ac.py:150-175.
        t0 = time.perf_counter()
        mk_exact = per_order_marginal_mismatch(
            q_theta, X_target, n=n,
            max_subsets_per_order=K,
            seed=int(seed),
        )
        exact_wall = time.perf_counter() - t0

        mean_tv_exact = float(mk_exact[k_int]["mean_tv"])
        exact_num_subsets = int(mk_exact[k_int]["num_subsets"])

        row = {
            "dataset": dataset,
            "criterion": f"val02_mean_tv_k{k_int}",
            "k": k_int,
            "seed": int(seed),
            "exact": mean_tv_exact,
            "estimate": mean_tv_est,
            "sigma": sigma_est,
            "num_subsets": K,
            "estimator_num_subsets": est_num_subsets,
            "exact_num_subsets": exact_num_subsets,
            "n_expval_samples": n_expval,
            "wall_time_s": float(est_wall + exact_wall),
            "est_wall_time_s": float(est_wall),
            "exact_wall_time_s": float(exact_wall),
            "metadata_wall_time_sec": float(
                est.metadata.get("wall_time_sec", 0.0)
            ),
        }
        rows.append(row)

        log.info(
            "VAL-02 dataset=%s seed=%d k=%d | est=%.4f +- %.4f  exact=%.4f  "
            "(z=%.2f)  est_subsets=%d exact_subsets=%d",
            dataset, seed, k_int,
            mean_tv_est, sigma_est, mean_tv_exact,
            (mean_tv_est - mean_tv_exact) / sigma_est if sigma_est > 0 else float("nan"),
            est_num_subsets, exact_num_subsets,
        )

    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_n16",
        description="Phase 2 n=16 validation harness (VAL-01 + VAL-02).",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to harness YAML config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Override combined JSON output path (default: from config).",
    )
    parser.add_argument("--skip-val01", action="store_true", default=False)
    parser.add_argument("--skip-val02", action="store_true", default=False)
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print planned subprocess argv for VAL-01; exit 0 without running.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Coerce num_subsets keys to int (PyYAML may return str for quoted keys).
    cfg["val02"]["num_subsets"] = _coerce_num_subsets(cfg["val02"]["num_subsets"])
    cfg["val02"]["k_values"] = [int(k) for k in cfg["val02"]["k_values"]]
    cfg["seeds"] = [int(s) for s in cfg["seeds"]]

    per_seed_root = REPO_ROOT / cfg["output"]["per_seed_root"]
    combined_json = args.out if args.out is not None else REPO_ROOT / cfg["output"]["combined_json"]
    combined_json = Path(combined_json)
    combined_json.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "validate_n16 starting: %d checkpoints x %d seeds, "
        "val01_skip=%s val02_skip=%s dry_run=%s",
        len(cfg["checkpoints"]), len(cfg["seeds"]),
        args.skip_val01, args.skip_val02, args.dry_run,
    )

    # VAL-01 (subprocess loop).
    val01_rows: list[dict[str, Any]] = []
    if not args.skip_val01:
        for ck in cfg["checkpoints"]:
            ds = ck["dataset"]
            ckpt_path = REPO_ROOT / ck["ckpt"]
            if not ckpt_path.exists() and not args.dry_run:
                raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
            for seed in cfg["seeds"]:
                row = run_val01(ds, ckpt_path, seed, cfg, per_seed_root,
                                dry_run=args.dry_run)
                val01_rows.append(row)

    if args.dry_run:
        log.info("dry-run: %d VAL-01 argvs enumerated; exiting without running.", len(val01_rows))
        return 0

    # VAL-02 (in-process).
    val02_rows: list[dict[str, Any]] = []
    if not args.skip_val02:
        for ck in cfg["checkpoints"]:
            ds = ck["dataset"]
            ckpt_path = REPO_ROOT / ck["ckpt"]
            spin_sym = bool(ck["spin_sym"])
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
            for seed in cfg["seeds"]:
                rows = run_val02(ds, ckpt_path, spin_sym, seed, cfg)
                val02_rows.extend(rows)

    # --- Stub combined JSON (Task 3 will replace with aggregated shape). ---
    payload = {
        "meta": {
            "harness": "scripts/validate_n16.py",
            "config_path": str(args.config.relative_to(REPO_ROOT) if args.config.is_absolute() else args.config),
            "checkpoints": [
                {
                    "dataset": ck["dataset"],
                    "path": ck["ckpt"],
                    "sha256": sha256_of(REPO_ROOT / ck["ckpt"]),
                }
                for ck in cfg["checkpoints"]
            ],
            "budgets": {
                "val01": dict(cfg["val01"]),
                "val02": {
                    "k_values": list(cfg["val02"]["k_values"]),
                    "num_subsets": {str(k): int(v) for k, v in cfg["val02"]["num_subsets"].items()},
                    "n_expval_samples": int(cfg["val02"]["n_expval_samples"]),
                },
            },
            "seeds": list(cfg["seeds"]),
            "git": git_state(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "notes": {
                "k_subset_alignment": "unaligned_same_distribution",
                "exact_mk_routing": "IQPModel.probability_vector_exact for BOTH datasets at n=16 (n<=20 guard)",
                "val02_sigma_source": "estimator per-k plug-in SE (MarginalResult.per_k[k]['sigma'])",
            },
        },
        "raw_val01_rows": val01_rows,
        "raw_val02_rows": val02_rows,
        "overall_pass": None,
        "overall_tier": None,
        "note": "TODO: val02 loop + aggregation + overall tier (Tasks 2, 3).",
    }

    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("wrote combined JSON: %s", combined_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
