# Architecture

**Analysis Date:** 2026-04-22

## Pattern Overview

**Overall:** Layered scientific-experiment pipeline with a CLI-driven orchestration layer over pure-numpy/JAX numerical kernels. Two sibling Python packages live under `src/`:

- `iqp_bp/` — the barren-plateau study package (gradient variance, anti-concentration, Forge structural modeling).
- `iqp_mmd/` — the upstream-derived training toolkit (IQP simulator, datasets, metrics, RBM/EBM baselines).

A third component is the **Forge formal model** at `forge/models/hypergraph.frg` (Racket/Alloy-based) driven through a Python bridge in `src/iqp_bp/forge/`.

**Key Characteristics:**
- Config-driven: a single YAML (deep-merged over `configs/base.yaml`) fully parameterizes every experiment run.
- Deterministic-by-construction: all randomness flows through labeled seed streams in `src/iqp_bp/rng.py`.
- Side effects happen only at the edges (experiment runners persist JSONL + manifests + checkpoints under `results/`).
- Numerical core (hypergraphs, IQP expectation, MMD kernel, gradients) is pure-numpy and unit-testable on small `n`.

## Layers

**CLI / Entry Layer:**
- Purpose: Dispatch a command name and one YAML path to the right runner.
- Location: `src/iqp_bp/cli.py`, `src/iqp_mmd/cli/`
- Contains: argparse wiring, lazy imports of experiment runners, exit-code management.
- Depends on: config loader, experiment runners.
- Used by: `pyproject.toml` console scripts (`iqp-bp`, `iqp-train`, `iqp-evaluate`, `iqp-sample`, `iqp-dataset`).

**Config Layer:**
- Purpose: Load, deep-merge, schema-validate, and expand scaling grids from YAML.
- Location: `src/iqp_bp/config.py`, `configs/base.yaml`, `configs/schema.yaml`, `src/iqp_mmd/config/`
- Contains: `load_config`, `validate_config`, `resolve_experiment_grid`, `persist_experiment_manifest`.
- Depends on: `yaml`, filesystem.
- Used by: every experiment runner.

**Experiment Orchestration Layer:**
- Purpose: For each resolved grid point, build a model, run numerical core, persist one JSONL record + sidecar artifacts.
- Location: `src/iqp_bp/experiments/`
- Contains: `run_scaling.py`, `run_training.py`, `run_qiskit.py`, `run_validation.py`, `run_forge.py`, plus shared helpers in `data_factory.py` and `marginal_artifacts.py`.
- Depends on: config layer, numerical core, RNG, Forge bridge.
- Used by: CLI.

**Numerical Core:**
- Purpose: Pure-numpy implementations of the hypergraph / IQP / MMD math.
- Location:
  - `src/iqp_bp/hypergraph/families.py` — circuit-family generator matrices (product_state, lattice, erdos_renyi, complete_graph, bounded_degree, dense, community, symmetric).
  - `src/iqp_bp/iqp/model.py` — `IQPModel` (G, theta, provenance).
  - `src/iqp_bp/iqp/expectation.py` — `iqp_expectation`, `iqp_expectation_exact`, `iqp_phase`.
  - `src/iqp_bp/mmd/kernel.py`, `mmd/loss.py`, `mmd/gradients.py`, `mmd/mixture.py` — MMD² estimator, analytic gradients, mixture helpers.
  - `src/iqp_bp/distributions/marginals.py`, `marginal_metrics.py` — marginal and anti-concentration diagnostics.
- Depends on: numpy only (JAX where explicitly used in training).
- Used by: experiment orchestration layer, Qiskit layer, Forge layer.

**Training Layer:**
- Purpose: Minimal MMD trainer with trajectory checkpoints.
- Location: `src/iqp_bp/training/trainer.py`
- Contains: `Trainer` class with `CheckpointCallback` protocol, SGD/Adam loops, marginal diagnostics.
- Depends on: numerical core, `run_validation.save_iqp_checkpoint`.
- Used by: `run_training.py`.

**Qiskit Validation Layer:**
- Purpose: Cross-check numpy results against Qiskit (statevector / Aer / IBM backends).
- Location: `src/iqp_bp/qiskit/`
- Contains: `circuit_builder.py`, `estimators.py`, `mmd.py`, `noise.py`.
- Depends on: numerical core, optional `qiskit`/`qiskit-aer`.
- Used by: `run_qiskit.py`.

**Forge Bridge Layer:**
- Purpose: Export a concrete IQP instance to a Forge `.frg` fact file, run Racket, parse the stdout for plateau/agreement verdicts.
- Location: `src/iqp_bp/forge/`
- Contains: `export_instances.export_to_forge`, `experiment_emitter.emit_experiment_instance`, `query_templates.build_query`, `runner.run_racket`, `parser.detect_plateau_agreement` / `detect_predicate_truth`, `label_loader.load_labeled_rows`.
- Depends on: external `racket` binary on PATH, the Forge model at `forge/models/hypergraph.frg`.
- Used by: `run_forge.py`.

**Upstream Toolkit (`iqp_mmd`):**
- Purpose: Full training toolkit from XanaduAI/scaling-gqml (IqpSimulator, RBM/EBM baselines, datasets, sampling, metrics).
- Location: `src/iqp_mmd/`
- Subpackages: `circuits/`, `cli/`, `config/`, `datasets/`, `metrics/`, `models/`, `observables/`, `sampling/`, `training/`, plus top-level `checkpoint_export.py`.
- Depends on: `iqpopt`, `pennylane`, `jax`, optional `torch`, `qml-benchmarks`.
- Used by: its own CLI scripts; `iqp_bp` imports only `checkpoint_export` artifacts indirectly.

## Data Flow

**`run-scaling` (the canonical flow):**

1. `cli.main` parses `run-scaling <cfg.yaml>` and calls `config.load_config` (deep-merged over `base.yaml`, schema-validated).
2. `run_scaling.run(cfg)` calls `resolve_experiment_grid(cfg)` to materialize the cartesian product of `(family, n_qubits, kernel, init_scheme, bandwidth, er_p_edge, small_angle_std)`.
3. `persist_experiment_manifest` writes `config.json` + `manifest.json` to `results/<experiment_name>/`.
4. For each setting:
   - `experiment_stream_bundle(base_seed, "run_scaling", setting_key)` derives per-stream seeds (`circuit`, `data`, `theta`, `kernel`, `estimation`).
   - `IQPModel.from_family(...)` builds a generator matrix `G` via `hypergraph.families.make_hypergraph`.
   - `make_dataset(dataset_cfg, n, seed)` synthesizes training data.
   - `theta_list` of `num_seeds` parameter vectors is drawn.
   - `_summarize_anti_concentration` optionally calls into `run_validation.evaluate_anti_concentration_from_model` and writes AC sidecar artifacts + an optional checkpoint.
   - For each of up to 5 param indices, `estimate_gradient_variance` runs with kernel-specific params.
   - One JSONL record per `(setting, param_idx)` is appended to `results/<name>/results.jsonl`.

**`run-forge` flow:**

1. `load_labeled_rows` reads a scaling `results.jsonl` and rehydrates G from the saved IQP checkpoints.
2. `export_to_forge` serializes G as a Forge hypergraph instance; `emit_experiment_instance` appends experiment metadata.
3. `build_query` renders a Forge `test-expect` body for the desired predicate.
4. `run_racket` shells out to `racket <.frg>`; stdout is captured in a `ForgeResult`.
5. `parser.detect_plateau_agreement` / `detect_predicate_truth` classify the stdout into verdicts persisted in the run's JSONL.

**State Management:**
- No in-process global state. Experiment state flows: `config dict → grid list → (per-setting) model + theta + data → JSONL record`.
- All seeds are content-derived via `derive_seed(base_seed, experiment_name, setting_key, stream, *extras)` so reruns are byte-identical.

## Key Abstractions

**`IQPModel`:**
- Purpose: Bundle `(G, theta, provenance)` with constructors from named families and JSON-safe provenance.
- Location: `src/iqp_bp/iqp/model.py`
- Pattern: Dataclass-like class with classmethod factory `from_family`.

**Circuit Families:**
- Purpose: Uniform constructor interface for generator matrices across all sweep-able families.
- Location: `src/iqp_bp/hypergraph/families.py`
- Pattern: Single dispatcher `make_hypergraph(family=..., n, m, rng, **kwargs)`.

**Seed Streams:**
- Purpose: Decouple independent sources of randomness so local changes do not shift unrelated draws.
- Location: `src/iqp_bp/rng.py`
- Pattern: Named string constants (`STREAM_CIRCUIT`, `STREAM_DATA`, `STREAM_THETA`, `STREAM_KERNEL`, `STREAM_ESTIMATION`, `STREAM_QISKIT`, `STREAM_FORGE`) combined via `derive_seed` (BLAKE2b-hashed JSON key).

**Config Grid Resolver:**
- Purpose: Lift list-valued YAML fields into a cartesian-product of scalar settings.
- Location: `src/iqp_bp/config.py::resolve_experiment_grid`
- Pattern: Pure function, `dict → list[dict]`.

**Forge `LabeledRow`:**
- Purpose: Pair a `results.jsonl` record with its checkpoint-derived G and AC labels so the Forge runner has a single input object.
- Location: `src/iqp_bp/forge/label_loader.py`

## Entry Points

**`iqp-bp` console script:**
- Location: `src/iqp_bp/cli.py::main`
- Triggers: `pip install -e .` exposes it via `pyproject.toml [project.scripts]`.
- Responsibilities: Dispatch to `grid-preview`, `run-scaling`, `run-training`, `run-qiskit`, `run-validation`, `run-forge`.

**`iqp-train` / `iqp-evaluate` / `iqp-sample` / `iqp-dataset`:**
- Location: `src/iqp_mmd/cli/train.py`, `evaluate.py`, `sample.py`, `generate_dataset.py`
- Triggers: Console scripts in `pyproject.toml`.
- Responsibilities: Upstream iqp_mmd workflows (training, evaluation, sampling, dataset generation).

**Scripts:**
- Location: `scripts/`
- Triggers: Direct `python scripts/<name>.py` invocations.
- Responsibilities: One-off analysis, plotting (`plot_ac_trajectory.py`, `plot_m_k_heatmap.py`, `f3_confusion_figure.py`, `f4_confusion_figure.py`), reruns, verification.

**Shell driver:**
- Location: `scripts/run_scaling.sh`, `scripts/setup_dev.sh`
- Triggers: Developer workflow.

## Error Handling

**Strategy:** Fail-fast at config validation; log-and-skip at per-setting numerical failures.

**Patterns:**
- `config.validate_config` raises `ValueError` with dotted-path messages (e.g. `"dataset.n_samples must be >= 1"`).
- Runners wrap the per-setting model build in `try/except ValueError` and log a warning with the skipped coordinate (see `run_scaling.run`).
- Forge subprocess failures return a `ForgeResult` with non-zero `returncode` and captured stderr; the parser classifies missing predicates as `"unknown"` rather than raising.
- Logging uses the stdlib `logging` module with a module-level `log = logging.getLogger(__name__)`.

## Cross-Cutting Concerns

**Logging:** stdlib `logging` per module. Log level is driven by `experiment.log_level` in YAML (`DEBUG`/`INFO`/`WARNING`).

**Validation:** Centralized in `config.validate_config` against `configs/schema.yaml`; enums enforced via `_SWEEPABLE_ENUM_PATHS` and `_SCALAR_ENUM_PATHS` tables.

**Reproducibility:** All RNG must flow through `iqp_bp.rng.derive_seed` / `experiment_stream_bundle`. Provenance dicts (family, n, m_requested, m_generated, rng_seed, kwargs) are attached to every `IQPModel` and serialized into checkpoints by `run_validation.save_iqp_checkpoint`.

**Persistence:** JSONL for records, JSON for manifests/configs, `.npz` for IQP checkpoints, PDF/PNG/SVG for figures. Output roots live under `results/<experiment_name>/`.

**Testing:** Pytest + Hypothesis (`tests/test_hypergraph_families.py`, `test_hypothesis.py`). Small-n exact tests (`test_expectation_small_n.py`, `test_mmd_exact_small_n.py`) pin the numerical core to ground truth; Qiskit cross-checks live in `test_qiskit_*.py`.

---

*Architecture analysis: 2026-04-22*
