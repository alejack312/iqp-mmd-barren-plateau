# External Integrations

**Analysis Date:** 2026-04-22

## APIs & External Services

This is a local-only scientific research codebase. It has no network-facing services, no third-party SaaS integrations, no auth providers, and no webhooks. All "integrations" are subprocess invocations of external toolchains or bundled SDKs for quantum/classical ML.

**Quantum toolchains (Python SDKs, in-process):**
- **IQPopt** (Xanadu) — IQP circuit optimization and sampling.
  - Package: `iqpopt>=2024.7.0`
  - Entry points: `src/iqp_mmd/sampling/sampler.py` (`import iqpopt as iqp`, `iqpopt.utils`), `src/iqp_mmd/training/iqp_trainer.py` (`iqpopt.gen_qml`), `src/iqp_mmd/metrics/mmd_eval.py`, `src/iqp_mmd/metrics/kgel.py`.
- **PennyLane** (Xanadu) — Variational quantum programming framework.
  - Package: `pennylane>=0.33`
  - Entry points: `src/iqp_mmd/circuits/pennylane.py` (`import pennylane as qml`).
- **Qiskit + Qiskit Aer** (IBM, optional extra) — Reference simulator for cross-validation of IQP MMD estimators.
  - Packages: `qiskit>=1.0`, `qiskit-aer>=0.14`
  - Entry points: `src/iqp_bp/qiskit/circuit_builder.py` (imports `qiskit.qasm2`), `src/iqp_bp/qiskit/mmd.py`, `src/iqp_bp/qiskit/noise.py`, `src/iqp_bp/qiskit/estimators.py`, `src/iqp_bp/experiments/run_qiskit.py`.
  - Install: `pip install -e ".[qiskit]"`.
- **qml-benchmarks** (Xanadu, optional extra) — Classical baselines (RBM, DeepEBM).
  - Package: `qml-benchmarks>=0.1`
  - Install: `pip install -e ".[benchmarks]"`.

**Formal methods toolchain (external subprocess):**
- **Forge** (Brown CSCI 1710, v5.2) — Alloy-family relational logic solver on top of Racket.
  - Not a Python package. Invoked via `racket <file.frg>` subprocess from `src/iqp_bp/forge/runner.py` (`run_racket`). Uses `shutil.which("racket")` detection and treats missing Racket as `status="skipped_no_racket"` rather than an error.
  - Models: `forge/models/hypergraph.frg`, `forge/models/hypergraph_examples.frg`.
  - Installed out-of-tree at `~/forge` (or `C:\Users\cuqui\cs1710\forge`) per README. Not vendored into repo.
  - Query templates emitted from `src/iqp_bp/forge/query_templates.py`; labels exported via `src/iqp_bp/forge/export_instances.py` and `src/iqp_bp/forge/experiment_emitter.py`; results parsed in `src/iqp_bp/forge/parser.py`.
  - VSCode extension: `siddharthaprasad.forge-fm` (optional).

**Autodiff / numerics:**
- **JAX / jaxlib** — CPU-only autodiff. Used pervasively for gradients through MMD Monte Carlo estimators. Key sites: `src/iqp_bp/rng.py` (`split_rng`, `derive_seed`), all `src/iqp_mmd/models/*.py`, all `src/iqp_mmd/training/*.py`, `src/iqp_mmd/observables/hamiltonian.py`.
- **Flax (`flax.linen`)** — Neural module layer for EBM models (`src/iqp_mmd/models/graph_ebm.py`).
- **NumPyro** — MCMC sampling for Ising lattice ground truth in `src/iqp_mmd/datasets/ising.py`.

## Data Storage

**Databases:**
- None. No ORM, no SQL, no key-value store.

**File Storage:**
- Local filesystem only. Experiment outputs written under `results/` (path configured via `experiment.output_dir` in `configs/base.yaml`, default `results/`).
- Per-run result formats:
  - `results.jsonl` — Line-delimited JSON records (one per `(n_qubits, seed, family, ...)` cell). Referenced by `configs/base.yaml` `forge.plateau_agreement.label_source`.
  - CSV files for samples / datasets (e.g. `./data/ising/train.csv`, `./output/samples.csv` per README).
  - Pickled model parameters (e.g. `./output/trained_parameters/params_*.pkl`).
  - Forge instance `.frg` files emitted at experiment time by `src/iqp_bp/forge/export_instances.py`.
- Training checkpoints: emitted by `src/iqp_mmd/checkpoint_export.py` and by `src/iqp_bp/experiments/run_training.py` (trajectory + marginal diagnostics per commit `9d10f43`).

**Caching:**
- `.pytest_cache/` — pytest state.
- `.hypothesis/constants/` — Hypothesis property-test database (many hash-named entries present; treat as generated state).
- Python `functools.lru_cache` used in-process (e.g. `src/iqp_bp/config.py`). No external cache (Redis/Memcached).

## External Datasets (one-way downloads)

Optional datasets are fetched from third-party sources when the `datasets` extra is installed (`torch`, `torchvision`, `requests`, `joblib`):

- **MNIST** — via `torchvision` (`src/iqp_mmd/datasets/mnist.py`, imports `torchvision` lazily).
- **D-Wave quantum-annealer samples** — Zenodo download (per README "Supported Datasets" table).
- **INRIA 1000 Genomes SNP data** — genomic SNP (805 / 10K-spin) datasets (per README).
- **2D Ising / scale-free Ising** — generated locally via NumPyro MCMC + networkx (Barabasi–Albert) in `src/iqp_mmd/datasets/ising.py`; no download.

These are developer-initiated, one-shot fetches — not ongoing integrations.

## Authentication & Identity

**Auth Provider:**
- None. No user accounts, no sessions, no API keys required.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, no Rollbar, etc.).

**Logs:**
- Python `logging` module (`experiment.log_level: INFO` in `configs/base.yaml`). Used in `src/iqp_bp/forge/label_loader.py`, `src/iqp_bp/experiments/run_validation.py`.
- `stdout`/`stderr` capture in `src/iqp_bp/forge/runner.py` (`ForgeResult.stdout`, `.stderr`) for Forge subprocess diagnostics.

## CI/CD & Deployment

**Hosting:**
- None — local-run research repo.

**CI Pipeline:**
- No `.github/workflows/`, no GitLab CI, no Travis, no Jenkins. `.github/` directory absent. Tests are developer-invoked (`python -m pytest tests/ -x -q`).

**Release:**
- No published package artifacts (version pinned at `0.1.0` in `pyproject.toml`). Editable install only.

## Environment Configuration

**Required env vars:**
- None. All configuration is file-driven via YAML in `configs/`.

**Secrets location:**
- No secrets. No `.env`, no `secrets.yaml`, no keyrings.

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server.

**Outgoing:**
- None. No outbound webhooks or callbacks.

## Subprocess Integrations

The only non-Python integration is the Forge/Racket subprocess:

- **Command:** `racket <path>.frg`
- **Wrapper:** `src/iqp_bp/forge/runner.py` → `run_racket(frg_path, timeout_sec=...)`
- **Detection:** `shutil.which("racket")` — missing binary yields `status="skipped_no_racket"` so CI/smoke paths degrade gracefully.
- **Status parsing:** `src/iqp_bp/forge/parser.py` (`detect_status`, `detect_plateau_agreement`, `detect_predicate_truth`) classifies stdout into `sat`/`unsat`/`agree`/`disagree`.
- **Timeout:** Per-query `timeout_sec` is mandatory; configured via `forge.timeout_sec` in `configs/base.yaml` (default 60s).

---

*Integration audit: 2026-04-22*
