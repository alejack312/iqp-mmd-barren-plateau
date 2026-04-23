# Technology Stack

**Analysis Date:** 2026-04-22

## Languages

**Primary:**
- Python >=3.10 — All research code, training, experiments, analysis. Source under `src/iqp_bp/` and `src/iqp_mmd/`.

**Secondary:**
- Racket / Forge (Forge v5.2, `#lang forge`) — Formal modeling of IQP hypergraph plateau theory. Files under `forge/models/*.frg`.
- Bash — Dev-environment and experiment launch scripts under `scripts/` (e.g. `scripts/setup_dev.sh`, `scripts/run_scaling.sh`).
- YAML — Experiment configs under `configs/` and `configs/experiments/`.
- Jupyter notebooks — Exploratory analysis under `notebooks/` (`00_sanity_checks.ipynb`, `01_scaling_plots.ipynb`).

## Runtime

**Environment:**
- CPython 3.10+ (hard floor from `pyproject.toml`, `requires-python = ">=3.10"`)
- Racket 8.7+ (required only for Forge subprocess execution; see `src/iqp_bp/forge/runner.py`). Not vendored — installed once per machine at `~/forge` per README and user memory (`C:\Users\cuqui\cs1710\forge`).
- Java 11+ (Forge prerequisite per README).

**Package Manager:**
- `pip` with `venv`, bootstrapped by `scripts/setup_dev.sh` (creates `.venv/`, `pip install -r requirements.txt`, then `pip install -e ".[dev]"`).
- Lockfile: missing — only `requirements.txt` (unpinned lower bounds) and `pyproject.toml` dependency ranges.

## Frameworks

**Core:**
- `jax` / `jaxlib` >=0.4.20 — Autodiff through Monte Carlo; primary differentiation engine for MMD gradients and IQP training. Used throughout `src/iqp_mmd/training/`, `src/iqp_mmd/models/`, and `src/iqp_bp/rng.py`.
- `flax` >=0.8.0 — Neural network layers for `DeepGraphEBM` / `GraphEBM` in `src/iqp_mmd/models/graph_ebm.py` (`flax.linen as nn`).
- `numpyro` >=0.13 — MCMC sampling for Ising dataset generation (`src/iqp_mmd/datasets/ising.py`).
- `pennylane` >=0.33 — Quantum circuit integration (`src/iqp_mmd/circuits/pennylane.py`).
- `iqpopt` >=2024.7.0 — Xanadu IQP optimization kernel; used for sampling and MMD gradient estimation (`src/iqp_mmd/sampling/sampler.py`, `src/iqp_mmd/metrics/mmd_eval.py`, `src/iqp_mmd/training/iqp_trainer.py`).
- `networkx` >=3.0 — Hypergraph / graph construction (`src/iqp_bp/hypergraph/families.py`).

**Testing:**
- `pytest` >=7.0 — Primary test runner. ~20 test modules in `tests/`.
- `hypothesis` >=6.90 — Property-based testing (`tests/test_hypothesis.py`, `src/iqp_bp/hypergraph/hypothesis_strategies.py`). `.hypothesis/constants/` is committed-via-ignore state.

**Build/Dev:**
- `hatchling` — Build backend (`pyproject.toml` `[build-system]`).
- `ruff` >=0.3 — Linter/formatter. Configured in `pyproject.toml` (`line-length = 120`, `target-version = "py310"`).
- `jupyterlab` >=4.0 + `ipykernel` >=6.0 — Notebook runtime (from `requirements.txt`).

## Key Dependencies

**Critical:**
- `numpy` >=1.24 — Universal array backend across `src/iqp_bp/`.
- `scipy` >=1.11 — Scientific routines used alongside numpy.
- `pandas` >=2.0 — Results tables and CSV IO for experiments.
- `pyyaml` >=6.0 — Config loading (`src/iqp_bp/config.py` imports `yaml`).
- `matplotlib` >=3.7, `seaborn` >=0.13 — Plotting (notebooks, `scripts/plot_*.py`, `scripts/f*_confusion_figure.py`).
- `scikit-learn` >=1.3 — Classical baselines / utilities.

**Infrastructure:**
- `iqpopt` (Xanadu) — Core IQP circuit optimization; distinct from the local `iqp_bp.iqp` exact expectation implementation.
- `qml-benchmarks` (Xanadu, optional extra `benchmarks`) — Classical baselines (RBM, DeepEBM).

**Optional extras (declared in `pyproject.toml`):**
- `dev`: pytest, ruff, hypothesis.
- `datasets`: `torch` >=2.0, `torchvision` >=0.15, `requests` >=2.28, `joblib` >=1.3 (MNIST, genomic downloaders).
- `benchmarks`: `qml-benchmarks` >=0.1.
- `qiskit`: `qiskit` >=1.0, `qiskit-aer` >=0.14 (imported in `src/iqp_bp/qiskit/circuit_builder.py` and `src/iqp_bp/experiments/run_qiskit.py`).

## Configuration

**Environment:**
- No `.env` files detected. No runtime secrets.
- Virtualenv convention: `.venv/` at repo root (activated via `.venv/Scripts/activate` on Windows or `.venv/bin/activate` on Linux/macOS).
- Python version pinned via `pyproject.toml` `requires-python`; no `.python-version` file present.

**Key configs:**
- `configs/base.yaml` — Project-wide defaults: `experiment`, `circuit` (family, n_qubits sweep), `kernel` (gaussian/laplacian/multi-scale), `init`, `dataset`, `estimation`, `training`, `qiskit`, `forge`.
- `configs/hyperparameters.yaml` — Per-model hyperparameters from the reference paper.
- `configs/schema.yaml` — Config schema.
- `configs/experiments/*.yaml` — 20 named experiment configs (e.g. `forge_f4_manifests.yaml`, `scaling_ac_smoke.yaml`, `qiskit_validation.yaml`, `training_smoke.yaml`).

**Build:**
- `pyproject.toml` — PEP 517/518 project definition; hatchling targets `src/iqp_mmd` and `src/iqp_bp`.
- No Dockerfile, no CI workflows (`.github/` absent).
- No pre-commit config.

## Platform Requirements

**Development:**
- Python 3.10+ with `pip` and `venv`.
- Git Bash / WSL / VSCode terminal on Windows (required for Forge's Racket subprocess per README; `cmd`/PowerShell explicitly unsupported).
- For Forge-dependent experiments: Racket 8.7+ and Java 11+ installed out-of-tree.
- For `qiskit` extras: additional install via `pip install -e ".[qiskit]"`.

**Production:**
- Not deployed — this is a research codebase. Experiments run locally or on an HPC-style shell (`scripts/run_scaling.sh`). Results written under `results/`.

## Console Scripts

Declared in `pyproject.toml` `[project.scripts]`:
- `iqp-train` → `iqp_mmd.cli.train:main`
- `iqp-evaluate` → `iqp_mmd.cli.evaluate:main`
- `iqp-sample` → `iqp_mmd.cli.sample:main`
- `iqp-dataset` → `iqp_mmd.cli.generate_dataset:main`
- `iqp-bp` → `iqp_bp.cli:main`

---

*Stack analysis: 2026-04-22*
