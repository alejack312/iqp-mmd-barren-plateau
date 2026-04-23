#!/usr/bin/env python
"""Phase 3: Dataset acquisition + IqpSimulator training pipeline.

Usage:
    python scripts/pauli_scale_pipeline.py --dataset dwave
    python scripts/pauli_scale_pipeline.py --all
    python scripts/pauli_scale_pipeline.py --all --acquire-only   # preflight

Outputs per dataset (if acquisition + training succeed):
    results/pauli_scale/<dataset>/checkpoint.npz
    results/pauli_scale/<dataset>/train_progress.log

Summary (always written, updated after each dataset):
    results/pauli_scale/datasets_status.json

Execution order: smallest-n-first to maximize partial results under deadline.
spin_glass + scale_free are expected to SKIP due to qml_benchmarks / numpyro
being broken in this environment (see Phase 3 RESEARCH).
"""
from __future__ import annotations

import os

# JAX env defaults -- must be set BEFORE any jax import. Users may override via env.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "pauli_scale"
DATASETS_ROOT = REPO_ROOT / "datasets"
STATUS_JSON_PATH = RESULTS_ROOT / "datasets_status.json"


# --- Direct-fd logger (Windows-safe; mirrors pauli_estimator_investigation.py). ----
_LOG_FD: int | None = None
_orig_print = print


def _log(*args, **kwargs) -> None:
    msg = " ".join(str(a) for a in args)
    _orig_print(msg, flush=True)
    if _LOG_FD is not None:
        os.write(_LOG_FD, (msg + "\n").encode("utf-8"))


# Route module-level print through the direct-fd logger.
print = _log  # noqa: A001


def _open_progress_log(out_dir: Path) -> None:
    """Open a direct os.open handle at <out_dir>/train_progress.log."""
    global _LOG_FD
    out_dir.mkdir(parents=True, exist_ok=True)
    if _LOG_FD is not None:
        try:
            os.close(_LOG_FD)
        except OSError:
            pass
        _LOG_FD = None
    _LOG_FD = os.open(
        str(out_dir / "train_progress.log"),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    )


def _close_progress_log() -> None:
    global _LOG_FD
    if _LOG_FD is not None:
        try:
            os.close(_LOG_FD)
        except OSError:
            pass
        _LOG_FD = None


# --- Hyperparameter table --------------------------------------------------
# Embedded here (not read from YAML at training time) to avoid circular
# dependency with configs/pauli_estimator_datasets.yaml. Values are paper-spec
# (Recio-Armengol et al.; cross-checked against configs/hyperparameters.yaml).
DATASET_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "spin_glass": dict(
        n_qubits=256, max_weight=2, spin_sym=False,
        sigma=[31.75, 21.53, 11.31], init_scale=0.01, param_noise=0.0,
        n_ops=1000, n_samples=1000, stepsize=0.0001,
    ),
    "dwave": dict(
        n_qubits=484, max_weight=2, spin_sym=False,
        sigma=[7.762, 6.151, 3.928], init_scale=0.001, param_noise=0.0,
        n_ops=1000, n_samples=1000, stepsize=0.001,
    ),
    "MNIST": dict(
        n_qubits=784, max_weight=2, spin_sym=False,
        sigma=[9.887, 7.391, 3.390], init_scale=0.001, param_noise=0.0,
        n_ops=1000, n_samples=1000, stepsize=0.001,
    ),
    "genomic-805": dict(
        n_qubits=805, max_weight=2, spin_sym=False,
        sigma=[10.019, 7.701, 4.269], init_scale=0.001, param_noise=0.0,
        n_ops=1000, n_samples=1000, stepsize=0.001,
    ),
    "scale_free": dict(
        n_qubits=1000, max_weight=2, spin_sym=False,
        sigma=[11.169, 8.348, 3.825], init_scale=0.01, param_noise=0.0,
        n_ops=1000, n_samples=1000, stepsize=0.001,
    ),
}

# Smallest-n-first order, so partial results accumulate before deadline.
TRAIN_ORDER = ["spin_glass", "dwave", "MNIST", "genomic-805", "scale_free"]


# --- Exception types -------------------------------------------------------
class SkipDataset(Exception):
    """Raised when a dataset cannot be acquired for environmental reasons
    (e.g. qml_benchmarks / numpyro broken). Caller should log + continue."""


class AcquisitionError(Exception):
    """Raised when an acquisition attempt fails for a transient reason
    (network error, format mismatch). Caller should log + continue."""


# --- Acquisition functions -------------------------------------------------
def _acquire_dwave(datasets_root: Path) -> Path:
    """Download + materialize the 484-spin D-Wave train CSV.

    Target output: datasets/dwave/dwave_X_train.csv (CSV, no header, 0/1 int32).

    Strategy:
      1. Try the in-repo helper iqp_mmd.datasets.dwave.download_dwave with an
         explicit output_dir -- this writes the CSV as a side-effect.
      2. If it raises (network hiccup, missing requests, tar format change),
         fall through to a manual requests + tarfile extraction.
    """
    out_dir = datasets_root / "dwave"
    out_csv = out_dir / "dwave_X_train.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        _log(f"  [dwave] CSV already present at {out_csv}; skipping download")
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Primary path: in-repo helper (has known-good side-effect). ---
    try:
        from iqp_mmd.datasets.dwave import download_dwave
        _log("  [dwave] calling download_dwave(output_dir=...)")
        download_dwave(output_dir=out_dir)
        if out_csv.exists() and out_csv.stat().st_size > 0:
            return out_csv
        raise AcquisitionError(
            "download_dwave returned without writing dwave_X_train.csv"
        )
    except Exception as e:  # fall through to manual
        _log(f"  [dwave] download_dwave failed ({type(e).__name__}: {e}); "
             "falling back to manual requests+tarfile path")

    # --- Fallback: manual requests + tar extraction. ---
    import tarfile
    import tempfile
    try:
        import requests  # noqa: F401  (may be absent in some envs)
    except ImportError as e:
        raise AcquisitionError(f"requests module unavailable: {e}") from e
    import requests

    url = "https://zenodo.org/records/7250436/files/datasets.tar.gz?download=1"
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "datasets.tar.gz"
        _log(f"  [dwave] manual download: {url}")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmpdir)
        npy_path = (
            Path(tmpdir)
            / "datasets"
            / "484-z8-100mus"
            / "train-484spins-3nn-uniform-100mus.npy"
        )
        if not npy_path.exists():
            raise AcquisitionError(
                f"expected D-Wave npy not found in archive: {npy_path}"
            )
        X = np.load(npy_path)
        X = X.reshape(X.shape[0], -1)
        # D-Wave samples come in {-1,+1}; convert to {0,1}.
        if X.min() < 0:
            X = ((X + 1) // 2).astype(np.int32)
        else:
            X = X.astype(np.int32)
        np.savetxt(out_csv, X, delimiter=",", fmt="%d")
    return out_csv


def _acquire_genomic(datasets_root: Path) -> Path:
    """Download the 805-SNP genome haplotype file via urllib.

    Target output: datasets/genomic/805_SNP_1000G_real_train.csv
    (this is exactly what DatasetPaths.train_path('genomic-805') expects).

    We do NOT call iqp_mmd.datasets.genomic.download_genomic because it shells
    out to wget, which fails silently on this machine.
    """
    out_dir = datasets_root / "genomic"
    out_csv = out_dir / "805_SNP_1000G_real_train.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        _log(f"  [genomic-805] CSV already present at {out_csv}; skipping download")
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/"
        "29c1ef7cf242e842df4360abae2eebeec995f40e/1000G_real_genomes/"
        "805_SNP_1000G_real.hapt"
    )
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_hapt = Path(tmpdir) / "805_SNP_1000G_real.hapt"
        _log(f"  [genomic-805] urllib download: {url}")
        try:
            urllib.request.urlretrieve(url, str(tmp_hapt))
        except Exception as e:
            raise AcquisitionError(
                f"urlretrieve failed for {url}: {type(e).__name__}: {e}"
            ) from e

        # Parse .hapt: whitespace-separated, each row a haplotype.
        # The upstream _load_hapt drops the first two columns (sample IDs).
        try:
            import pandas as pd
        except ImportError as e:
            raise AcquisitionError(f"pandas unavailable: {e}") from e
        data = pd.read_csv(tmp_hapt, sep=r"\s+", header=None, engine="python")
        if data.shape[1] > 805:
            data = data.drop(columns=[0, 1])
        X = data.values.astype(np.int32)
        # Defensive: clamp to {0,1}.
        X = np.clip(X, 0, 1).astype(np.int32)
        np.savetxt(out_csv, X, delimiter=",", fmt="%d")
    return out_csv


def _acquire_mnist(datasets_root: Path) -> Path:
    """Download MNIST, binarize pixels, emit CSV.

    Target output: datasets/MNIST/x_train.csv (CSV, no header, 0/1 int32).

    Strategy:
      1. Try torchvision.datasets.MNIST (cached locally if available).
      2. Else fetch the raw IDX3 gzip from the ossci S3 mirror (yann.lecun.com
         has been unreliable).
    """
    out_dir = datasets_root / "MNIST"
    out_csv = out_dir / "x_train.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        _log(f"  [MNIST] CSV already present at {out_csv}; skipping download")
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Primary path: torchvision. ---
    try:
        import torchvision  # noqa: F401
        _log("  [MNIST] trying torchvision.datasets.MNIST ...")
        import torchvision.datasets as tvd
        raw = tvd.MNIST(root="/tmp/mnist_data", train=True, download=True)
        X = np.asarray(raw.data).reshape(len(raw.data), -1)
        X = (X > 127).astype(np.int32)
        np.savetxt(out_csv, X, delimiter=",", fmt="%d")
        return out_csv
    except Exception as e:
        _log(f"  [MNIST] torchvision path failed ({type(e).__name__}: {e}); "
             "falling back to raw IDX3 download")

    # --- Fallback: raw IDX3 from ossci mirror. ---
    import gzip
    import struct
    import tempfile
    urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    ]
    last_exc: Exception | None = None
    with tempfile.TemporaryDirectory() as tmpdir:
        gz_path = Path(tmpdir) / "train-images-idx3-ubyte.gz"
        for url in urls:
            try:
                _log(f"  [MNIST] urllib download: {url}")
                urllib.request.urlretrieve(url, str(gz_path))
                break
            except Exception as e:
                _log(f"  [MNIST] {url} failed: {type(e).__name__}: {e}")
                last_exc = e
        else:  # pragma: no cover -- all URLs failed
            raise AcquisitionError(
                f"all MNIST mirrors failed; last error: {last_exc}"
            )
        with gzip.open(gz_path, "rb") as f:
            magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise AcquisitionError(
                    f"MNIST IDX3 bad magic number: {magic} (expected 2051)"
                )
            buf = f.read(n_images * n_rows * n_cols)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_images, n_rows * n_cols)
            X = (arr > 127).astype(np.int32)
            np.savetxt(out_csv, X, delimiter=",", fmt="%d")
    return out_csv


def _acquire_spin_glass(datasets_root: Path) -> Path:
    raise SkipDataset(
        "numpyro_broken: cannot import IsingSpins from qml_benchmarks "
        "(numpyro incompat in this env; see Phase 3 RESEARCH)"
    )


def _acquire_scale_free(datasets_root: Path) -> Path:
    raise SkipDataset(
        "numpyro_broken: cannot import IsingSpins from qml_benchmarks "
        "(numpyro incompat in this env; see Phase 3 RESEARCH)"
    )


_ACQUIRE_DISPATCH = {
    "spin_glass": _acquire_spin_glass,
    "dwave": _acquire_dwave,
    "MNIST": _acquire_mnist,
    "genomic-805": _acquire_genomic,
    "scale_free": _acquire_scale_free,
}


def acquire_dataset(name: str, datasets_root: Path) -> Path:
    """Dispatch to the correct `_acquire_*` function. SkipDataset re-raised;
    other exceptions wrapped in AcquisitionError for uniform caller handling."""
    if name not in _ACQUIRE_DISPATCH:
        raise ValueError(f"unknown dataset: {name!r}")
    fn = _ACQUIRE_DISPATCH[name]
    try:
        return fn(datasets_root)
    except SkipDataset:
        raise
    except AcquisitionError:
        raise
    except Exception as e:
        raise AcquisitionError(
            f"{name} acquisition failed: {type(e).__name__}: {e}"
        ) from e


# --- Status JSON -----------------------------------------------------------
def _write_status(status: dict[str, dict[str, Any]]) -> None:
    STATUS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Ensure all 5 datasets always appear in the JSON (pending ones as 'pending').
    full: dict[str, dict[str, Any]] = {}
    for name in TRAIN_ORDER:
        if name in status:
            full[name] = status[name]
        else:
            full[name] = {
                "outcome": "pending",
                "reason": None,
                "csv_path": None,
                "checkpoint_path": None,
            }
    tmp_path = STATUS_JSON_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(full, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, STATUS_JSON_PATH)


def _load_status_if_exists() -> dict[str, dict[str, Any]]:
    if STATUS_JSON_PATH.exists():
        try:
            with open(STATUS_JSON_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


# --- Training stub (full body added in Task 2) -----------------------------
def train_dataset(
    name: str,
    csv_path: Path,
    n_iters: int = 500,
    seed: int = 666,
) -> dict[str, Any]:
    """Train IqpSimulator via iqpopt + save deterministic checkpoint.

    [Task 1 stub] The real implementation lands in Task 2 and will defer
    jax/iqpopt imports to call-time so that `--acquire-only` preflights don't
    pay startup cost. Raising NotImplementedError here makes the expected
    return contract unambiguous; --acquire-only paths do not reach this stub.

    Returns (when implemented): dict with fields: checkpoint_path, final_loss,
    initial_loss, n_iters_run, train_time_sec, tag.
    """
    raise NotImplementedError(
        "train_dataset is implemented in Task 2 of 03-01-PLAN.md; "
        "run with --acquire-only until then"
    )


# --- Dispatch wrappers -----------------------------------------------------
def _rel_to_repo(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(p).resolve())


def run_single(
    name: str,
    n_iters: int = 500,
    seed: int = 666,
    acquire_only: bool = False,
) -> None:
    """Acquire + (optionally) train one dataset; update status JSON."""
    status = _load_status_if_exists()
    _log(f"\n=== Dataset: {name} ===")
    try:
        csv_path = acquire_dataset(name, DATASETS_ROOT)
        status[name] = {
            "outcome": "acquired",
            "reason": None,
            "csv_path": _rel_to_repo(csv_path),
            "checkpoint_path": None,
        }
        _log(f"  ACQUIRED {name}: {csv_path}")
    except SkipDataset as e:
        status[name] = {
            "outcome": "skip",
            "reason": str(e),
            "csv_path": None,
            "checkpoint_path": None,
        }
        _log(f"  SKIP {name}: {e}")
        _write_status(status)
        return
    except AcquisitionError as e:
        status[name] = {
            "outcome": "acquisition_error",
            "reason": str(e),
            "csv_path": None,
            "checkpoint_path": None,
        }
        _log(f"  ACQUISITION ERROR {name}: {e}")
        _write_status(status)
        return

    if acquire_only:
        _write_status(status)
        return

    # Restart logic: if a checkpoint already exists, skip retraining.
    ckpt_path = RESULTS_ROOT / name / "checkpoint.npz"
    if ckpt_path.exists():
        _log(f"  [{name}] checkpoint already at {ckpt_path}; skipping training")
        status[name]["outcome"] = "success"
        status[name]["checkpoint_path"] = _rel_to_repo(ckpt_path)
        _write_status(status)
        return

    _open_progress_log(RESULTS_ROOT / name)
    try:
        result = train_dataset(name, Path(csv_path), n_iters=n_iters, seed=seed)
        status[name]["outcome"] = "success"
        status[name].update(result)
    except Exception as e:
        status[name]["outcome"] = "training_error"
        status[name]["reason"] = f"{type(e).__name__}: {e}"
        _log(f"  TRAINING ERROR {name}: {e}")
    finally:
        _close_progress_log()
    _write_status(status)


def run_all(
    n_iters: int = 500,
    seed: int = 666,
    acquire_only: bool = False,
) -> None:
    """Iterate TRAIN_ORDER, acquiring and (optionally) training each dataset."""
    status = _load_status_if_exists()
    _write_status(status)  # normalize + write pending slots

    for name in TRAIN_ORDER:
        _log(f"\n=== Dataset: {name} ===")

        # --- Acquire -----------------------------------------------------
        try:
            csv_path = acquire_dataset(name, DATASETS_ROOT)
            status[name] = {
                "outcome": "acquired",
                "reason": None,
                "csv_path": _rel_to_repo(csv_path),
                "checkpoint_path": None,
            }
            _log(f"  ACQUIRED {name}: {csv_path}")
        except SkipDataset as e:
            status[name] = {
                "outcome": "skip",
                "reason": str(e),
                "csv_path": None,
                "checkpoint_path": None,
            }
            _log(f"  SKIP {name}: {e}")
            _write_status(status)
            continue
        except AcquisitionError as e:
            status[name] = {
                "outcome": "acquisition_error",
                "reason": str(e),
                "csv_path": None,
                "checkpoint_path": None,
            }
            _log(f"  ACQUISITION ERROR {name}: {e}")
            _write_status(status)
            continue

        if acquire_only:
            _write_status(status)
            continue

        # --- Restart check ------------------------------------------------
        ckpt_path = RESULTS_ROOT / name / "checkpoint.npz"
        if ckpt_path.exists():
            _log(f"  [{name}] checkpoint already at {ckpt_path}; skipping training")
            status[name]["outcome"] = "success"
            status[name]["checkpoint_path"] = _rel_to_repo(ckpt_path)
            _write_status(status)
            continue

        # --- Train --------------------------------------------------------
        _open_progress_log(RESULTS_ROOT / name)
        try:
            result = train_dataset(name, Path(csv_path), n_iters=n_iters, seed=seed)
            status[name]["outcome"] = "success"
            status[name].update(result)
        except Exception as e:
            status[name]["outcome"] = "training_error"
            status[name]["reason"] = f"{type(e).__name__}: {e}"
            _log(f"  TRAINING ERROR {name}: {e}")
        finally:
            _close_progress_log()
        _write_status(status)


# --- CLI entrypoint --------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pauli_scale_pipeline",
        description=(
            "Phase 3: Acquire + train IQP pipeline for the 5 big-n datasets. "
            "Skips datasets that cannot be acquired (numpyro/qml_benchmarks "
            "broken) rather than crashing."
        ),
    )
    p.add_argument(
        "--dataset",
        choices=list(DATASET_HYPERPARAMS),
        default=None,
        help="Run a single dataset.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 datasets in smallest-n-first order.",
    )
    p.add_argument("--n-iters", type=int, default=500,
                   help="Training iterations per dataset (default: 500).")
    p.add_argument("--seed", type=int, default=666,
                   help="Random seed (default: 666).")
    p.add_argument(
        "--acquire-only",
        action="store_true",
        help="Acquire datasets only; do not start training. Used for preflight.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.all and not args.dataset:
        parser.print_help()
        return 1

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all:
        run_all(
            n_iters=args.n_iters,
            seed=args.seed,
            acquire_only=args.acquire_only,
        )
    else:
        run_single(
            args.dataset,
            n_iters=args.n_iters,
            seed=args.seed,
            acquire_only=args.acquire_only,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
