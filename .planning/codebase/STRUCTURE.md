# Codebase Structure

**Analysis Date:** 2026-04-22

## Directory Layout

```
iqp-mmd-barren-plateau/
├── src/
│   ├── iqp_bp/                  # Barren-plateau study package (primary work)
│   │   ├── __init__.py
│   │   ├── cli.py               # argparse dispatcher; `iqp-bp` console script
│   │   ├── config.py            # YAML loader + schema validation + grid resolver
│   │   ├── rng.py               # Deterministic seed streams (BLAKE2b-based)
│   │   ├── distributions/       # Marginal + anti-concentration diagnostics
│   │   │   ├── marginals.py
│   │   │   └── marginal_metrics.py
│   │   ├── experiments/         # Experiment runners (one per CLI subcommand)
│   │   │   ├── data_factory.py
│   │   │   ├── marginal_artifacts.py
│   │   │   ├── run_forge.py
│   │   │   ├── run_qiskit.py
│   │   │   ├── run_scaling.py
│   │   │   ├── run_training.py
│   │   │   └── run_validation.py
│   │   ├── forge/               # Python bridge to the Forge/Racket model
│   │   │   ├── experiment_emitter.py
│   │   │   ├── export_instances.py
│   │   │   ├── label_loader.py
│   │   │   ├── parser.py
│   │   │   ├── query_templates.py
│   │   │   └── runner.py
│   │   ├── hypergraph/          # Circuit-family generator matrices
│   │   │   ├── families.py
│   │   │   └── hypothesis_strategies.py
│   │   ├── iqp/                 # IQP model + expectation kernels
│   │   │   ├── expectation.py
│   │   │   └── model.py
│   │   ├── mmd/                 # MMD kernel, loss, gradients, mixture helpers
│   │   │   ├── gradients.py
│   │   │   ├── kernel.py
│   │   │   ├── loss.py
│   │   │   └── mixture.py
│   │   ├── qiskit/              # Qiskit validation path (optional extra)
│   │   │   ├── circuit_builder.py
│   │   │   ├── estimators.py
│   │   │   ├── mmd.py
│   │   │   └── noise.py
│   │   └── training/            # MMD trainer + checkpoint trajectory
│   │       └── trainer.py
│   └── iqp_mmd/                 # Upstream toolkit (XanaduAI/scaling-gqml derived)
│       ├── __init__.py
│       ├── checkpoint_export.py
│       ├── circuits/pennylane.py
│       ├── cli/                 # train / evaluate / sample / generate_dataset
│       ├── config/              # paths, hyperparams
│       ├── datasets/            # blobs, dwave, genomic, ising, loaders, mnist
│       ├── metrics/             # covariance, kgel, mmd_eval
│       ├── models/              # graph_ebm, iqp_simulator
│       ├── observables/hamiltonian.py
│       ├── sampling/sampler.py
│       └── training/            # ebm_trainer, iqp_trainer, rbm_trainer, hyperparams
├── configs/
│   ├── base.yaml                # Defaults that every experiment merges over
│   ├── schema.yaml              # Type schema consumed by validate_config
│   ├── hyperparameters.yaml     # Best hyperparameters from the upstream paper
│   └── experiments/             # One YAML per experiment preset (forge_*, scaling_*, training_smoke, validation, qiskit_validation)
├── forge/
│   ├── models/
│   │   ├── hypergraph.frg       # Forge (Racket/Alloy) structural model
│   │   └── hypergraph_examples.frg
│   └── runs/                    # Generated per-run Forge artifacts
├── tests/                       # pytest + hypothesis
│   ├── test_anti_concentration.py
│   ├── test_config.py
│   ├── test_expectation_small_n.py / _streaming.py
│   ├── test_gradients_small_n.py
│   ├── test_hypergraph_families.py
│   ├── test_hypothesis.py
│   ├── test_iqp_mmd_checkpoint_export.py
│   ├── test_iqp_model_provenance.py
│   ├── test_iqp_probability_vector_small_n.py
│   ├── test_marginal_metrics.py / test_marginals.py
│   ├── test_mmd_exact_small_n.py / test_mmd_kernel.py / test_mmd_loss_details.py
│   ├── test_qiskit_circuit_builder.py / test_qiskit_mmd.py / test_qiskit_noise.py
│   ├── test_rng.py
│   ├── test_run_forge.py / test_run_scaling.py / test_run_training.py / test_run_validation.py / test_run_qiskit.py
│   ├── test_scaling_data_factory.py
│   ├── test_trainer.py
│   ├── test_experiment_emitter.py / test_export_instances.py / test_label_loader.py
│   └── test_imports.py
├── scripts/                     # One-off analysis, plotting, reruns
├── notebooks/                   # Jupyter (00_sanity_checks, 01_scaling_plots)
├── results/                     # Output root; one subdir per experiment name
├── docs/
│   ├── Background and Project Summary.md
│   ├── Proposal.md
│   ├── SMART-spec.md
│   ├── upstream-audit.md
│   ├── papers/                  # Reference PDFs
│   └── technical/               # anti-concentration, architecture, bandwidth-marginals, configuration, glossary, iqp-classical-sampling, learning-task, mmd-gaussian-fourier, p1-p5 notes
├── iqp-mmd-barren-plateau-vault/# Obsidian vault mirroring docs (Code MOC, Experiments MOC, etc.)
├── pyproject.toml               # hatchling build; ruff config; console scripts
├── requirements.txt
├── README.md
├── LICENSE
├── TODOS.md
├── ScopeLock.md
└── architecture.excalidraw      # Architecture diagram source
```

## Directory Purposes

**`src/iqp_bp/`:**
- Purpose: All code for the barren-plateau study. This is where new experiment work goes.
- Contains: CLI, config loader, experiment runners, numerical core (hypergraphs, IQP, MMD), Qiskit validation, Forge bridge, training.
- Key files: `cli.py`, `config.py`, `rng.py`, `iqp/model.py`, `mmd/loss.py`, `experiments/run_scaling.py`.

**`src/iqp_mmd/`:**
- Purpose: Upstream XanaduAI/scaling-gqml toolkit. Touch only when mirroring upstream changes or wiring `iqp_bp` to upstream checkpoints.
- Contains: PennyLane IQP simulator, RBM/EBM trainers, dataset loaders, upstream metrics.
- Key files: `cli/train.py`, `training/iqp_trainer.py`, `models/iqp_simulator.py`.

**`configs/`:**
- Purpose: All experiment configuration. No configs live in code.
- Contains: `base.yaml` (defaults), `schema.yaml` (type contract), `experiments/*.yaml` (one preset per run).
- Key files: `configs/base.yaml`, `configs/schema.yaml`, `configs/experiments/scaling_v1.yaml`.

**`forge/`:**
- Purpose: Forge (Racket/Alloy) structural model + generated run artifacts.
- Contains: `.frg` models and per-run emitted instances.
- Generated: `forge/runs/` is generated per invocation.
- Committed: `forge/models/` yes; `forge/runs/` artifacts typically no.

**`tests/`:**
- Purpose: Pytest + Hypothesis suite. Mirror package structure.
- Contains: One `test_<module>.py` per source module. Small-n exact-truth tests plus property-based strategies.
- Generated scratch dirs (`_tmp_*`, `_pytest_tmp`) are ignored.

**`scripts/`:**
- Purpose: One-off analysis, plotting, reruns. Not part of the importable package.
- Contains: Plotting (`plot_*.py`), confusion figures (`f3_confusion_figure.py`, `f4_confusion_figure.py`), verification helpers (`verify_frozen_predicate.py`, `_verify_checkpoint_faithful.py`), setup scripts (`setup_dev.sh`, `run_scaling.sh`).

**`notebooks/`:**
- Purpose: Sanity checks and visualization. Not on the experiment critical path.

**`results/`:**
- Purpose: Output root. Every experiment writes `results/<experiment.name>/{config.json, manifest.json, results.jsonl, checkpoints/, anti_concentration/, ...}`.
- Generated: Yes. Committed: No (see `.gitignore`).

**`docs/`:**
- Purpose: Prose documentation and reference PDFs.
- Key files: `docs/technical/architecture.md`, `docs/technical/configuration.md`, `docs/technical/anti-concentration.md`, `docs/SMART-spec.md`.

**`iqp-mmd-barren-plateau-vault/`:**
- Purpose: Obsidian vault mirroring `docs/` for linked note-taking.
- Generated: No (hand-maintained). Committed: Yes.

## Key File Locations

**Entry Points:**
- `src/iqp_bp/cli.py`: `iqp-bp` console script; dispatches the six experiment subcommands.
- `src/iqp_mmd/cli/train.py`: `iqp-train` console script.
- `src/iqp_mmd/cli/evaluate.py`: `iqp-evaluate` console script.
- `src/iqp_mmd/cli/sample.py`: `iqp-sample` console script.
- `src/iqp_mmd/cli/generate_dataset.py`: `iqp-dataset` console script.

**Configuration:**
- `configs/base.yaml`: Defaults merged into every experiment.
- `configs/schema.yaml`: Type schema enforced by `validate_config`.
- `configs/experiments/`: Per-experiment YAMLs (one per preset).
- `pyproject.toml`: Build, deps, console scripts, ruff config.

**Core Logic:**
- `src/iqp_bp/iqp/model.py`: `IQPModel` class.
- `src/iqp_bp/iqp/expectation.py`: IQP expectation and phase functions.
- `src/iqp_bp/hypergraph/families.py`: `make_hypergraph` dispatcher.
- `src/iqp_bp/mmd/loss.py`, `mmd/kernel.py`, `mmd/gradients.py`: MMD math.
- `src/iqp_bp/rng.py`: `derive_seed`, `experiment_stream_bundle`, stream constants.
- `src/iqp_bp/config.py`: `load_config`, `resolve_experiment_grid`, `persist_experiment_manifest`.

**Testing:**
- `tests/test_*.py`: Pytest modules (see layout above).
- `.hypothesis/`: Hypothesis example DB; not committed.
- `pytest-cache-files-*`: transient.

## Naming Conventions

**Files:**
- Pattern: `snake_case.py` for all Python modules (e.g. `run_scaling.py`, `circuit_builder.py`).
- Tests: `test_<module>.py` mirroring the source module name.
- Scripts: `snake_case.py` plus private helpers prefixed `_` (e.g. `scripts/_codex_parse.py`, `scripts/_smoke_probe.py`).
- Shell: `snake_case.sh` in `scripts/`.

**Directories:**
- Pattern: `snake_case` for packages (`iqp_bp`, `iqp_mmd`, `hypergraph`, `anti_concentration` subdirs under results).
- Experiment output dirs: match `experiment.name` from the YAML (e.g. `results/forge_f4_manifests/`).

**Configs:**
- Pattern: `snake_case.yaml`. Experiment presets live under `configs/experiments/` and often encode ablations with `_ablation.yaml` or holdouts with `_holdout_<thing>.yaml` suffixes.

**Constants:**
- Stream names: `STREAM_<UPPER>` (e.g. `STREAM_CIRCUIT`, `STREAM_FORGE`) in `src/iqp_bp/rng.py`.
- Validation enum tables: leading underscore + uppercase (`_VALID_FAMILIES`, `_SCALAR_ENUM_PATHS`) in `config.py`.

**Checkpoint filenames:**
- Pattern: `<family>_n<n>_<kernel>_<init_scheme>_seed<idx>.npz` (see `run_scaling._checkpoint_name`).

**Artifact stems:**
- Pattern: `<family>__n<n>__<kernel>__<init>__<dataset>[__sigma<x>][__theta<x>][__er<x>]` with `-` → `m` and `.` → `p` in floats (see `run_scaling._setting_stem` / `_format_scalar`).

## Where to Add New Code

**New experiment type (new CLI subcommand):**
- Runner: `src/iqp_bp/experiments/run_<name>.py` exposing `run(cfg: dict) -> None`.
- Wire in: `src/iqp_bp/cli.py` — add to the `for cmd in (...)` tuple and the dispatch `if/elif`.
- Config: `configs/experiments/<name>.yaml` extending `base.yaml`.
- Schema: extend `configs/schema.yaml` and the enum tables in `src/iqp_bp/config.py` if new fields are introduced.
- Tests: `tests/test_run_<name>.py`.

**New circuit family:**
- Implementation: add a branch in `src/iqp_bp/hypergraph/families.py::make_hypergraph`.
- Register: add the name to `_VALID_FAMILIES` in `src/iqp_bp/config.py`.
- Family kwargs: add a branch in each runner's `_extract_family_kwargs` (see `run_scaling.py` and `run_forge.py`).
- Tests: extend `tests/test_hypergraph_families.py` and `tests/test_hypothesis.py`.

**New kernel:**
- Implementation: `src/iqp_bp/mmd/kernel.py`.
- Register: `_VALID_KERNELS` in `config.py` and `_bandwidth_values_for_kernel` / `_get_kernel_params` in `run_scaling.py`.
- Tests: `tests/test_mmd_kernel.py` and `tests/test_mmd_exact_small_n.py`.

**New dataset type:**
- Implementation: `src/iqp_bp/experiments/data_factory.py::make_dataset` branch.
- Register: `_VALID_DATASETS` in `config.py` and validation in `_validate_dataset_config`.
- Tests: `tests/test_scaling_data_factory.py`.

**New Forge predicate/query:**
- Predicate: extend `forge/models/hypergraph.frg`.
- Query builder: add to `src/iqp_bp/forge/query_templates.py`.
- Parser: add detector to `src/iqp_bp/forge/parser.py`.
- Tests: `tests/test_run_forge.py`.

**New RNG consumer:**
- Never call `np.random.default_rng()` directly from experiment code. Use `derive_seed(base_seed, experiment_name, setting_key, STREAM_*, *extras)` from `src/iqp_bp/rng.py`, and add a new `STREAM_<NAME>` constant there if a new stream is needed.

**New script / plot:**
- `scripts/<name>.py` (not importable; use `python scripts/<name>.py`). Prefix private helpers with `_`.

**New Jupyter exploration:**
- `notebooks/<nn>_<topic>.ipynb` with a two-digit prefix.

**Shared utilities:**
- Prefer the existing module where the utility logically belongs. New cross-cutting helpers for `iqp_bp` go in a new module under `src/iqp_bp/` (e.g. `src/iqp_bp/<name>.py`) rather than a catch-all `utils.py`.

## Special Directories

**`results/`:**
- Purpose: Experiment outputs (config snapshot, manifest, results.jsonl, checkpoints, AC artifacts, figures).
- Generated: Yes. Committed: No.

**`.hypothesis/`:**
- Purpose: Hypothesis example database cache.
- Generated: Yes. Committed: No.

**`forge/runs/`:**
- Purpose: Per-run Forge `.frg` instance files emitted by the bridge.
- Generated: Yes. Committed: selectively (README present).

**`pytest-cache-files-*`, `tests/_tmp_*`, `tests/_pytest_tmp/`:**
- Purpose: Pytest scratch.
- Generated: Yes. Committed: No.

**`iqp-mmd-barren-plateau-vault/.obsidian/`:**
- Purpose: Obsidian workspace config for the vault.
- Generated: partial. Committed: Yes (used for shared note-taking).

---

*Structure analysis: 2026-04-22*
