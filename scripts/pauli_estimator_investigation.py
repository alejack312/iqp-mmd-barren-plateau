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
            "raw_Y_samples_path": str(raw_Y_path.relative_to(REPO_ROOT)),
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


def _cmd_estimate(args: argparse.Namespace) -> int: raise NotImplementedError  # Task 3
def _cmd_validate(args: argparse.Namespace) -> int: raise NotImplementedError  # Task 3
def _cmd_all(args: argparse.Namespace) -> int: raise NotImplementedError       # Task 3


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
