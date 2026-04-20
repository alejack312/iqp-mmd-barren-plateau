# TODO Dependency Roadmap

Dependency-aware execution list derived from `docs/SMART-spec.md` and the `TODO:` markers in `src/iqp_bp`.

Audit status as of 2026-04-08:
- The full anti-concentration track (`AC1`-`AC6`) is complete; the Apr 8 10:30 critical-path deadline was met.
- Scaling grid expansion (`S6`) is complete; the runner sweeps the full Cartesian grid.
- Several foundation and validation TODOs are now complete.
- Many remaining TODOs still have partial scaffolding already present in the codebase.

Legend:
- `[x]` complete
- `[~]` partial
- `[ ]` open

How to read this file:
- `Ready now` means the task does not depend on any other current TODO in `src/iqp_bp`.
- `Depends on` means the task should be done after the listed TODO IDs.
- IDs are stable planning labels for this file only.
- Glossary links for recurring terms:
  [locked MMD^2 derivation](docs/technical/glossary.md#locked-mmd2-derivation),
  [ZZ lattice family](docs/technical/glossary.md#zz-lattice-family),
  [sparse Erdos-Renyi family](docs/technical/glossary.md#sparse-erdos-renyi-family),
  [generator matrix](docs/technical/glossary.md#generator-matrix),
  [generator](docs/technical/glossary.md#generator),
  [mask](docs/technical/glossary.md#mask),
  [parity](docs/technical/glossary.md#parity),
  [spectral weight](docs/technical/glossary.md#spectral-weight),
  [gradient concentration](docs/technical/glossary.md#gradient-concentration)

Implementation docs index:
- Hypothesis: [Hypothesis API](https://hypothesis.readthedocs.io/en/latest/reference/api.html), [Hypothesis strategies](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html), [Hypothesis NumPy](https://hypothesis.readthedocs.io/en/latest/numpy.html)
- JAX / NumPy: [JAX grad](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html), [JAX value_and_grad](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html), [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html), [JAX jit](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html), [JAX random](https://docs.jax.dev/en/latest/jax.random.html), [NumPy Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- Config / CLI / serialization: [PyYAML docs](https://pyyaml.org/wiki/PyYAMLDocumentation), [pathlib.Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [itertools.product](https://docs.python.org/3/library/itertools.html#itertools.product), [json module](https://docs.python.org/3/library/json.html), [pytest parametrize](https://docs.pytest.org/en/7.1.x/how-to/parametrize.html)
- Graph generation / fitting: [NetworkX grid_2d_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.lattice.grid_2d_graph.html), [NetworkX erdos_renyi_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html), [SciPy curve_fit](https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.optimize.curve_fit.html)
- Qiskit / Aer: [Qiskit QuantumCircuit](https://quantum.cloud.ibm.com/docs/api/qiskit/2.1/qiskit.circuit.QuantumCircuit), [Qiskit ParameterVector](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.0/qiskit.circuit.ParameterVector), [Qiskit qasm2](https://docs.quantum.ibm.com/api/qiskit/qasm2), [Qiskit transpile](https://quantum.cloud.ibm.com/docs/api/qiskit/0.39/qiskit.compiler.transpile), [Qiskit primitives](https://quantum.cloud.ibm.com/docs/api/qiskit/dev/primitives), [Qiskit SparsePauliOp](https://docs.quantum.ibm.com/api/qiskit/0.24/qiskit.quantum_info.SparsePauliOp), [AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html), [Aer NoiseModel](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html), [Aer depolarizing_error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.depolarizing_error.html), [Aer ReadoutError](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.ReadoutError.html), [Aer amplitude_damping_error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.amplitude_damping_error.html), [Aer thermal_relaxation_error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html)
- Forge: [Forge docs](https://forge-fm.github.io/forge-documentation/), [Forge constraints](https://forge-fm.github.io/forge-documentation/building-models/constraints/constraints.html), [Forge options](https://forge-fm.github.io/forge-documentation/running-models/options.html), [Forge model intuition](https://forge-fm.github.io/book/chapters/solvers/bounds_booleans_how_forge_works.html)

## Ready Now

### Foundation and Theory

- `[x] T1` [Week 1] [Confirm the exact Gaussian spectral normalization used by the locked MMD^2 derivation](src/iqp_bp/mmd/kernel.py#L34)
  Current state: the Gaussian kernel, spectral weights, and Z-word sampling path now all use the same locked convention `k(x,y) = exp(-H(x,y)/(2 sigma^2))` with Walsh decay `tau = tanh(1/(4 sigma^2))`. See [locked MMD^2 derivation](docs/technical/glossary.md#locked-mmd2-derivation) and [spectral weight](docs/technical/glossary.md#spectral-weight).
  Docs to read first: [NumPy Generator]. Treat this as a theory-first task; no external API should be changed until the local derivation is locked.

- `[~] T2` [Week 1] [Keep the Laplacian kernel path as an explicit stub until its MMD^2 decomposition is derived and checked](src/iqp_bp/mmd/kernel.py#L92)
  Current state: there is an approximate Laplacian implementation, but it is not yet a clearly validated or theory-locked final path.
  Docs to read first: [NumPy Generator]. Keep this implementation minimal until the local derivation is complete.

- `[x] T3` [Week 1] [Replace the generic 2D patch sampler with the exact nearest-neighbour ZZ lattice family in scope](src/iqp_bp/hypergraph/families.py#L80)
  Current state: the 2D lattice family is now the exact open-boundary square-grid nearest-neighbour ZZ construction, with square-only validation and deterministic row generation.
  Docs to read first: [NetworkX grid_2d_graph], [NumPy Generator].

### Reproducibility and Core Pipeline

- `[~] P1` [Week 1] [Add config validation and a persisted resolved experiment grid for reproducible reruns](src/iqp_bp/config.py#L27)
  Current state: config loading and deep-merge are implemented, and the scaling runner already knows how to resolve scalar settings, but schema validation, persisted grid manifests, and a real preview surface are still missing.
  Docs to read first: [PyYAML docs], [pathlib.Path], [itertools.product], [json module].
  Implementation checklist:
  - `[x]` [src/iqp_bp/config.py](src/iqp_bp/config.py): add `validate_config(cfg)` against [configs/schema.yaml](configs/schema.yaml), failing loudly on missing required sections, wrong scalar-vs-list shapes, and unsupported enum values used by current runners.
  - `[x]` [src/iqp_bp/config.py](src/iqp_bp/config.py): add a config-owned helper that materializes the resolved scalar experiment grid for `run-scaling`, instead of leaving grid resolution discoverable only through the runner internals.
  - `[x]` [src/iqp_bp/config.py](src/iqp_bp/config.py): add persistence helpers that write the merged config plus resolved grid manifest to JSON in `experiment.output_dir` before long runs start.
  - `[x]` [src/iqp_bp/cli.py](src/iqp_bp/cli.py): add a real `grid-preview` or `validate-config` subcommand wired into argparse; remove the current unreachable branch and make preview print the resolved scalar settings rather than only the merged YAML.
  - `[x]` [src/iqp_bp/experiments/run_scaling.py](src/iqp_bp/experiments/run_scaling.py): consume the config-owned resolved-grid helper and persist the manifest path into run artifacts so reruns can be traced to one exact grid.
  - `[x]` [tests/test_config.py](tests/test_config.py): add tests for valid merge+validation, invalid enum/scalar shape rejection, and persisted manifest contents.
  - `[x]` [tests/test_run_scaling.py](tests/test_run_scaling.py): add a smoke test that the scaling run writes a resolved-grid/config manifest before records are emitted.

- `[x] P2` [Week 1] [Reserve named RNG streams for circuit, data, theta, kernel, and Qiskit sampling](src/iqp_bp/rng.py#L25)
  Current state: `derive_seed()` and `named_seed_streams()` exist, and `run_scaling` already uses named seeds for `circuit` and `data`, but there is still no single enforced stream contract for `theta`, kernel sampling, and Qiskit/noise sampling.
  Docs to read first: [NumPy Generator], [JAX random].
  Implementation checklist:
  - `[x]` [src/iqp_bp/rng.py](src/iqp_bp/rng.py): define the canonical stream names and responsibilities for `circuit`, `data`, `theta`, `kernel`, `estimation`, and `qiskit`.
  - `[x]` [src/iqp_bp/rng.py](src/iqp_bp/rng.py): add a small public helper for requesting the standard stream bundle for a given experiment coordinate so callers do not re-invent labels ad hoc.
  - `[x]` [src/iqp_bp/experiments/run_scaling.py](src/iqp_bp/experiments/run_scaling.py): replace inline `derive_seed(..., "theta", idx)` and estimation-specific seed construction with the canonical named stream helpers.
  - `[x]` [src/iqp_bp/mmd/loss.py](src/iqp_bp/mmd/loss.py): split kernel-word sampling and IQP expectation sampling into separate RNG streams so Monte Carlo draws remain stable under refactors.
  - `[x]` [src/iqp_bp/mmd/gradients.py](src/iqp_bp/mmd/gradients.py): stop sharing one mutable RNG across `sample_a`, `iqp_expectation`, and gradient estimators; use deterministic substreams by responsibility.
  - `[x]` [src/iqp_bp/experiments/run_qiskit.py](src/iqp_bp/experiments/run_qiskit.py) and [src/iqp_bp/qiskit/noise.py](src/iqp_bp/qiskit/noise.py): thread the same named-stream policy into shot sampling and noise-model randomness.
  - `[x]` [tests/test_rng.py](tests/test_rng.py): add tests for stable named seeds, stream separation, and loop-order invariance of resolved experiment coordinates.
  - `[x]` [tests/test_run_scaling.py](tests/test_run_scaling.py): add a rerun reproducibility test proving identical outputs for the same config/seed.

- `[x] P3` [Week 1] [Add stable batching or streaming to the classical IQP expectation engine](src/iqp_bp/iqp/expectation.py#L69)
  Current state: the expectation engine is vectorized, but it does not yet batch or stream to control memory explicitly.
  Docs to read first: [NumPy Generator].
  Implementation checklist:
  - `[x]` [src/iqp_bp/iqp/expectation.py](src/iqp_bp/iqp/expectation.py): add a `batch_size` or `streaming` argument to `iqp_expectation()` and compute the Monte Carlo estimate over bounded chunks of `z`.
  - `[x]` [src/iqp_bp/iqp/expectation.py](src/iqp_bp/iqp/expectation.py): accumulate the mean and variance online so the public return value stays `(estimate, stderr)` without materializing all cosine samples at once.
  - `[x]` [src/iqp_bp/mmd/loss.py](src/iqp_bp/mmd/loss.py): pass the new batching control through the MMD estimator so large `num_z_samples` runs can opt into bounded-memory behavior.
  - `[x]` [src/iqp_bp/mmd/gradients.py](src/iqp_bp/mmd/gradients.py): thread the same batching control through analytic gradient estimation paths.
  - `[x]` [tests/test_expectation_small_n.py](tests/test_expectation_small_n.py): add parity tests showing batched and unbatched estimates match for a fixed seed within floating-point tolerance.
  - `[x]` [tests/test_expectation_streaming.py](tests/test_expectation_streaming.py): add deterministic chunking tests covering different batch sizes and exact-vs-MC agreement.

- `[x] P4` [Week 1] [Preserve family and generation metadata on IQP models for experiment and Qiskit provenance](src/iqp_bp/iqp/model.py#L41)
  Current state: `IQPModel` wraps `G` and `theta`, but it does not preserve family or provenance metadata.
  Docs to read first: [json module], [pathlib.Path]. The main learning target here is the repo's own model/provenance conventions.
  Implementation checklist:
  - `[x]` [src/iqp_bp/iqp/model.py](src/iqp_bp/iqp/model.py): extend `IQPModel` to store JSON-serializable provenance metadata alongside `G` and `theta`.
  - `[x]` [src/iqp_bp/iqp/model.py](src/iqp_bp/iqp/model.py): make `from_family()` populate provenance with at least `family`, requested `n`, requested/generated `m`, family kwargs, and generation seed labels.
  - `[x]` [src/iqp_bp/experiments/run_scaling.py](src/iqp_bp/experiments/run_scaling.py): build models through a provenance-preserving path for anti-concentration and checkpoint export, rather than reconstructing metadata externally.
  - `[x]` [src/iqp_bp/experiments/run_validation.py](src/iqp_bp/experiments/run_validation.py): include model provenance in checkpoint save/load paths so validation runs can reconstruct the originating circuit family without parallel bookkeeping.
  - `[x]` [tests/test_iqp_model_provenance.py](tests/test_iqp_model_provenance.py): add tests that `from_family()` captures provenance and preserves it through checkpoint round-trips.
  - `[x]` [tests/test_run_scaling.py](tests/test_run_scaling.py): strengthen the existing checkpoint assertions to verify stored provenance fields, not just `family`.

- `[x] P5` [Week 1] [Expose per-observable contributions and confidence diagnostics in the MMD^2 estimator](src/iqp_bp/mmd/loss.py#L46)
  Current state: the MMD^2 estimator exists, but it only returns a scalar and does not expose contribution-level diagnostics.
  Docs to read first: [NumPy Generator], [json module].
  Implementation checklist:
  - `[x]` [src/iqp_bp/mmd/loss.py](src/iqp_bp/mmd/loss.py): add a non-breaking detailed path such as `return_details=False` that exposes sampled observables, `exp_p`, `exp_q`, per-observable squared contributions, and aggregate Monte Carlo diagnostics.
  - `[x]` [src/iqp_bp/mmd/loss.py](src/iqp_bp/mmd/loss.py): report estimator uncertainty consistently, including point estimate, sample std, stderr, and sample count for the observable mixture.
  - `[x]` [src/iqp_bp/mmd/gradients.py](src/iqp_bp/mmd/gradients.py): reuse the detailed MMD path where it helps keep gradient debugging and MMD debugging numerically aligned.
  - `[x]` [src/iqp_bp/experiments/run_scaling.py](src/iqp_bp/experiments/run_scaling.py): decide whether detailed diagnostics stay in-memory only, or are written as sidecar JSON per setting; if persisted, keep `results.jsonl` compact and store only stable pointers.
  - `[x]` [tests/test_hypothesis.py](tests/test_hypothesis.py): keep the scalar contract unchanged for existing callers.
  - `[x]` [tests/test_mmd_loss_details.py](tests/test_mmd_loss_details.py): add tests for returned diagnostics shape, deterministic seeded outputs, and exact reconstruction of the scalar MMD estimate from per-observable contributions.

### Anti-Concentration Track

Deadline for `AC1`-`AC6`: `2026-04-08 10:30` local time.

- `[x] AC1` [Due Apr 8 10:30] [Write the anti-concentration technical note and lock the repo's finite-n decision rule](docs/technical/anti-concentration.md)
  Current state: the anti-concentration technical note is locked and comprehensive, covering threshold and second-moment definitions and the deterministic implementation targets.
  Docs to read first: local paper `docs/papers/2512.24801v1.pdf`, [locked MMD^2 derivation](docs/technical/glossary.md#locked-mmd2-derivation), [the learning task](docs/technical/learning-task.md).

- `[x] AC2` [Due Apr 8 10:30] [Add exact small-n IQP output-probability extraction and normalization checks](src/iqp_bp/iqp/model.py#L41)
  Current state: the exact probability extraction path `probability_vector_exact` (and its alias) is implemented in `IQPModel` using an in-place Walsh-Hadamard transform.
  Docs to read first: [NumPy Generator], [itertools.product], local paper `docs/papers/2512.24801v1.pdf`. Keep this path exact and small-n only; do not blur it with sample-based approximations.

### Scaling Inputs

- `[x] S1` [Weeks 3-4] [Calibrate the sparse Erdos-Renyi family to the SMART bounded-degree regime](src/iqp_bp/hypergraph/families.py#L48)
  Current state: `erdos_renyi` now samples a sparse pairwise graph with bounded expected degree using `p = min(1, c / n)` and returns one weight-2 generator row per sampled edge. See [sparse Erdos-Renyi family](docs/technical/glossary.md#sparse-erdos-renyi-family).
  Docs to read first: [NetworkX erdos_renyi_graph], [NumPy Generator].

- `[x] S2` [Weeks 3-4] [Implement the Ising-like synthetic target and the structured real or binary-mixture target](src/iqp_bp/experiments/run_scaling.py#L130)
  Current state: dataset config placeholders exist, but only product Bernoulli data is actually implemented.
  Docs to read first: [NumPy Generator], [json module]. The key learning task is to mirror the SMART dataset definitions exactly.
  Implementation checklist:
  - `[x]` [src/iqp_bp/experiments/data_factory.py](src/iqp_bp/experiments/data_factory.py): make the dataset factory the only runtime entry point for scaling datasets, and harden the `ising` and `binary_mixture` branches with explicit config validation, deterministic seeding, and JSON-serializable dataset provenance.
  - `[x]` [src/iqp_bp/experiments/data_factory.py](src/iqp_bp/experiments/data_factory.py): tighten the `ising` generator contract so `grid_2d` rejects non-square `n`, `erdos_renyi` records the realized graph statistics, and returned metadata contains all hyperparameters needed to reproduce one dataset draw exactly.
  - `[x]` [src/iqp_bp/experiments/data_factory.py](src/iqp_bp/experiments/data_factory.py): tighten the `binary_mixture` contract so mode count, thresholding, noise scale, and latent-center generation policy are all recorded in metadata and validated for sane ranges.
  - `[x]` [src/iqp_bp/config.py](src/iqp_bp/config.py) and [configs/schema.yaml](configs/schema.yaml): extend schema validation so `dataset.type in {"product_bernoulli","ising","binary_mixture"}` is enforced and nested `dataset.ising.*` / `dataset.binary_mixture.*` keys are shape-checked before long runs begin.
  - `[x]` [src/iqp_bp/experiments/run_scaling.py](src/iqp_bp/experiments/run_scaling.py): keep dataset generation delegated to `make_dataset(...)`, but promote `dataset_metadata` into the resolved-setting artifact path so every JSONL record and checkpoint sidecar can be traced back to one concrete target distribution.
  - `[x]` [tests/test_scaling_data_factory.py](tests/test_scaling_data_factory.py): extend the existing factory tests to cover invalid `ising` shapes, invalid topology names, reproducible `erdos_renyi` topology metadata, and the binary-mixture metadata contract.
  - `[x]` [tests/test_run_scaling.py](tests/test_run_scaling.py): add an end-to-end smoke test proving one scaling run succeeds for `dataset.type=ising` and one for `dataset.type=binary_mixture`, and that the emitted records carry the expected `dataset_metadata`.

- `[~] S3` [Weeks 3-4] [Add cached parity statistics and structured target-data helpers on the data side of MMD](src/iqp_bp/mmd/mixture.py#L39)
  Current state: batched dataset parity expectations are implemented, but caching and structured target-data helpers are missing.
  Docs to read first: [NumPy Generator], [pathlib.Path], [json module].

### Qiskit and Forge Prep

- `[x] Q1` [Week 5] [Split measured vs unmeasured Qiskit builders and emit QASM and transpilation metadata](src/iqp_bp/qiskit/circuit_builder.py#L55)
  Current state: `circuit_builder.py` now exposes explicit measured and unmeasured builders, returns a `CircuitSpec` carrying qubit/generator counts plus QASM text, and provides JSON-serializable transpilation metadata; `run_qiskit.py` consumes that split API and the new builder tests cover the emitted metadata deterministically.
  Docs to read first: [Qiskit QuantumCircuit], [Qiskit ParameterVector], [Qiskit qasm2], [Qiskit transpile].
  Implementation checklist:
  - `[x]` [src/iqp_bp/qiskit/circuit_builder.py](src/iqp_bp/qiskit/circuit_builder.py): split the current builder into explicit unmeasured and measured paths so statevector estimation never depends on measurement-stripping while shot-based estimation still gets a measurement-ready circuit.
  - `[x]` [src/iqp_bp/qiskit/circuit_builder.py](src/iqp_bp/qiskit/circuit_builder.py): return a circuit-spec metadata bundle alongside the circuit, including qubit count, generator count, bound-vs-parameterized status, and QASM export text for the exact circuit that was executed.
  - `[x]` [src/iqp_bp/qiskit/circuit_builder.py](src/iqp_bp/qiskit/circuit_builder.py): add a helper for transpiling one circuit under an explicit backend/options contract and persist the resulting depth, size, and basis-gate metadata in a JSON-serializable form.
  - `[x]` [src/iqp_bp/qiskit/__init__.py](src/iqp_bp/qiskit/__init__.py): re-export the new measured/unmeasured builder entry points so downstream code stops importing one ambiguous builder name.
  - `[x]` [src/iqp_bp/experiments/run_qiskit.py](src/iqp_bp/experiments/run_qiskit.py): consume the split builder API and store the returned circuit/QASM/transpile metadata in every validation artifact, rather than reconstructing those facts ad hoc inside the runner.
  - `[x]` [configs/experiments/qiskit_validation.yaml](configs/experiments/qiskit_validation.yaml): add any missing transpilation/export options needed by the new builder contract, such as optimization level, basis-gate pinning, or whether raw QASM should be saved.
  - `[x]` [tests/test_qiskit_circuit_builder.py](tests/test_qiskit_circuit_builder.py): add tests proving measured and unmeasured builders produce the expected measurement split and that QASM/transpile metadata are emitted deterministically for a fixed config.

- `[~] Q2` [Week 6] [Add amplitude-damping and backend-inspired Qiskit noise presets](src/iqp_bp/qiskit/noise.py#L64)
  Current state: depolarizing, readout, and combined noise models exist, but amplitude damping and backend-like presets do not.
  Docs to read first: [Aer NoiseModel], [Aer depolarizing_error], [Aer ReadoutError], [Aer amplitude_damping_error], [Aer thermal_relaxation_error].

- `[x] F1` [Week 7] [Emit overlap-graph, degree-constraint, and threshold facts in Forge exports](src/iqp_bp/forge/export_instances.py#L34)
  Current state: Forge export exists for qubits, generators, and containment, but not for overlap-graph or threshold facts.
  Docs to read first: [Forge docs], [Forge constraints], [Forge model intuition].

## Depends On Other TODOs

### Week 1 Follow-Through

- `[~] P6` [Week 1] [Add a config-validation or grid-preview CLI subcommand](src/iqp_bp/cli.py#L21)
  Depends on: `P1`
  Current state: the CLI already has a `--dry-run` flag, but not a dedicated grid-preview or validation command.
  Docs to read first: [PyYAML docs], [pathlib.Path], [json module].

### Validation Layer

- `[x] V1` [Week 2] [Add Hypothesis strategies for the four SMART circuit families](src/iqp_bp/hypergraph/hypothesis_strategies.py#L1)
  Depends on: `T3`, `S1`
  Current state: the Hypothesis layer now samples the four SMART families only, and the family-specific tests cover the exact lattice and sparse ER structural invariants.
  Docs to read first: [Hypothesis API], [Hypothesis strategies], [Hypothesis NumPy], [pytest parametrize], [NetworkX grid_2d_graph], [NetworkX erdos_renyi_graph].

- `[~] V2` [Week 2] [Wire the exact IQP expectation path into automated Monte Carlo vs exact validation plots](src/iqp_bp/iqp/expectation.py#L97)
  Depends on: `P3`
  Current state: the exact IQP expectation path exists, but there is no automated validation harness or plotting.
  Docs to read first: [pytest parametrize], [pathlib.Path], [json module].

- `[~] V3` [Week 2] [Implement the JAX autodiff gradient estimator and compare it to the analytic path](src/iqp_bp/mmd/gradients.py#L45)
  Depends on: `P5`
  Current state: analytic gradients and finite differences exist, and JAX support is hinted at in RNG utilities, but autodiff is not implemented here.
  Docs to read first: [JAX grad], [JAX value_and_grad], [JAX vmap], [JAX jit], [JAX random].

- `[x] V4` [Week 2] [Add an exact small-n MMD^2 path for brute-force kernel validation](src/iqp_bp/mmd/loss.py#L62)
  Depends on: `T1`, `T2`, `P5`
  Current state: `mmd2_exact_small_n(...)` is implemented, reuses the Monte Carlo spectral-weight convention, relies on exact model and dataset-side expectation helpers, and is covered by dedicated exact-path and MC-vs-exact regression tests.
  Docs to read first: [pytest parametrize], [itertools.product], [NumPy Generator].
  Implementation checklist:
  - `[x]` [src/iqp_bp/mmd/loss.py](src/iqp_bp/mmd/loss.py): add a dedicated exact small-`n` entry point, such as `mmd2_exact_small_n(...)` or `mmd2(..., exact_small_n=True)`, that enumerates the full observable support instead of sampling `a`.
  - `[x]` [src/iqp_bp/mmd/loss.py](src/iqp_bp/mmd/loss.py): make the exact path reuse the same kernel spectral-weight convention as the Monte Carlo path, so validation is testing one formula in two execution modes rather than two separate derivations.
  - `[x]` [src/iqp_bp/iqp/model.py](src/iqp_bp/iqp/model.py): reuse `expectation_exact(...)` / `probability_vector_exact(...)` to provide the model-side exact `Z_a` expectations values needed by the brute-force MMD computation without introducing a second exact-IQP implementation.
  - `[x]` [src/iqp_bp/mmd/mixture.py](src/iqp_bp/mmd/mixture.py): add a helper for exact empirical `Z_a` expectations evaluation over the full observable batch so the dataset side of the exact path stays vectorized and numerically aligned with the sampled path.
  - `[x]` [src/iqp_bp/mmd/kernel.py](src/iqp_bp/mmd/kernel.py): expose or refactor the exact per-observable spectral weights needed by the exact small-`n` path, especially for Gaussian and any explicitly supported non-Gaussian kernels.
  - `[x]` [tests/test_mmd_loss_details.py](tests/test_mmd_loss_details.py): add regression tests that compare sampled `mmd2(return_details=True)` against the exact small-`n` result for fixed seeds and confirm the detailed contribution bookkeeping reconstructs the exact scalar.
  - `[x]` [tests/test_expectation_small_n.py](tests/test_expectation_small_n.py) or a new [tests/test_mmd_exact_small_n.py](tests/test_mmd_exact_small_n.py): add brute-force validation cases over tiny `n` where exact MMD^2 can be checked against hand-constructed toy distributions and exact IQP models.

- `[~] V5` [Week 2] [Add Hypothesis coverage for data-dependent init and the full small-angle sweep](src/iqp_bp/hypergraph/hypothesis_strategies.py#L69)
  Depends on: `S4`
  Current state: uniform and small-angle strategies exist, but the full sweep and data-dependent init are missing.
  Docs to read first: [Hypothesis API], [Hypothesis strategies], [Hypothesis NumPy], [pytest parametrize].

- `[x] AC3` [Due Apr 8 10:30] [Implement the deterministic anti-concentration checker over exact probability vectors and sample histograms](src/iqp_bp/experiments/run_validation.py#L1)
  Depends on: `AC1`, `AC2`
  Current state: the anti-concentration checker `check_anti_concentration` and associated evaluators are implemented in `run_validation.py`, supporting both exact vectors and empirical histograms.
  Docs to read first: [NumPy Generator], [json module], local paper `docs/papers/2512.24801v1.pdf`. The exact small-n path is the primary claim; histogram mode should be explicitly labeled as a secondary diagnostic.

- `[x] AC4` [Due Apr 8 10:30] [Extend the validation runner to load trained IQP checkpoints or generated bitstrings and serialize anti-concentration artifacts](src/iqp_bp/experiments/run_validation.py#L1)
  Depends on: `AC2`, `AC3`, `P4`
  Current state: the validation runner supports checkpoint loading (`.npz`), sample loading, and serializes results to JSON and CSV artifacts via `write_anti_concentration_artifacts`.
  Docs to read first: [pathlib.Path], [json module]. Preserve provenance: family, n, training seed, checkpoint path, and whether the result came from exact probabilities or sampled counts.

- `[x] AC5` [Due Apr 8 10:30] [Add deterministic tests for the anti-concentration pass/fail boundary and the sample-to-exact convergence path](tests/test_expectation_small_n.py#L1)
  Depends on: `AC2`, `AC3`
  Current state: comprehensive tests exist in `test_anti_concentration.py`, `test_iqp_probability_vector_small_n.py`, and `test_run_validation.py`.
  Docs to read first: [pytest parametrize], [NumPy Generator]. Include at least: uniform-distribution pass, delta-distribution fail, and a toy histogram that approaches the exact checker as sample size grows.

### Scaling v1

- `[~] S4` [Weeks 3-4] [Replace the data-dependent init stub with the covariance-informed initializer](src/iqp_bp/experiments/run_scaling.py#L146)
  Depends on: `S2`
  Current state: a stubbed data-dependent init branch exists, but it is still just a small Gaussian draw.
  Docs to read first: [NumPy Generator], [JAX random]. The key learning task is the covariance logic in the repo's data path, not an external package.

- `[~] S5` [Weeks 3-4] [Enforce the primary four-family sweep and comparable parameter-count policies centrally](src/iqp_bp/hypergraph/families.py#L240)
  Depends on: `T3`, `S1`
  Current state: the four primary families are present and used in configs, but the policy is not centrally enforced.
  Docs to read first: [itertools.product], [json module].

- `[x] S6` [Weeks 3-4] [Expand the scaling runner to sweep the full Cartesian grid of experiment axes](src/iqp_bp/experiments/run_scaling.py#L39)
  Depends on: `P1`, `P2`, `S2`, `S4`, `S5`
  Current state: `resolve_scaling_settings` expands the full Cartesian product over family, kernel, init, `n`, per-kernel bandwidth, per-family ER degree, and per-init small-angle std via `itertools.product`, and writes one JSONL record per resolved scalar setting and parameter index.
  Docs to read first: [itertools.product], [pathlib.Path], [json module].

- `[~] S7` [Weeks 3-4] [Extend gradient-variance summaries with aggregate norm proxies and heavy-tail diagnostics](src/iqp_bp/mmd/gradients.py#L164)
  Depends on: `V3`
  Current state: mean, variance, std, and median are reported, but the richer plateau diagnostics are missing.
  Docs to read first: [NumPy Generator], [SciPy curve_fit].

- `[~] S8` [Weeks 3-4] [Fit polynomial vs exponential scaling and emit the summary artifacts for the interim memo](src/iqp_bp/experiments/run_scaling.py#L86)
  Depends on: `S6`, `S7`
  Current state: raw JSONL records are written, but no fitting or memo-oriented summaries are produced.
  Docs to read first: [SciPy curve_fit], [pathlib.Path], [json module].

- `[x] AC6` [Due Apr 8 10:30] [Emit anti-concentration summaries and checkpoint plots alongside the scaling outputs](src/iqp_bp/experiments/run_scaling.py#L39)
  Depends on: `AC4`, `S6`
  Current state: `run_scaling.py` calls `_summarize_anti_concentration` for every setting with `n <= anti_concentration.max_n`, serializes per-setting summary JSON, thresholds CSV, and threshold/diagnostics plots via `write_anti_concentration_artifacts`, optionally exports `.npz` checkpoints through `save_iqp_checkpoint`, and copies compact `ac_*` fields onto each gradient-variance JSONL row while keeping the artifact schema separate. Landed in commit `da9db4a` ("feat: Implement scaling dataset factory and finish AC6 anti-concentration artifacts").
  Docs to read first: [pathlib.Path], [json module], [SciPy curve_fit]. Keep the anti-concentration output schema separate from the gradient-scaling schema so downstream analysis can distinguish them cleanly.

- `[~] AC7` [Week 2] [Implement marginal computation helpers for exact probability vectors and sampled bitstrings](src/iqp_bp/distributions/marginals.py)
  Depends on: none
  Current state: the repo now has `exact_marginal`, `sample_marginal`, exact/sample Walsh coefficient helpers, subset enumeration with capped uniform sampling, and probability-vector sampling for exact-to-sampled bridging. The remaining work is scaling validation on longer runs, not the core math API.

- `[~] AC8` [Week 2] [Add per-order marginal mismatch metrics tied to the Gaussian `tau^|S|` weight](src/iqp_bp/distributions/marginal_metrics.py)
  Depends on: `AC7`
  Current state: TV, chi-square, Fourier squared error, and `summarize_by_order(...)` are implemented. For Gaussian runs the summary reports `tau_power`, `order_weight`, and `weighted_mmd2_contribution`, so the per-order table reconstructs the scalar Gaussian MMD total when all subsets are included.

- `[~] AC9` [Week 2] [Implement a minimal MMD trainer with per-step checkpoint trajectories](src/iqp_bp/training/trainer.py)
  Depends on: none
  Current state: `Trainer` now supports SGD/Adam, deterministic step-seeded loss/gradient evaluation, trajectory JSONL persistence, and `.npz` checkpoints at step `0`, every `checkpoint_every`, and the final step. The exact small-`n` loss mode also uses exact small-`n` gradients so smoke tests stay deterministic.

- `[~] AC10` [Week 2] [Emit anti-concentration and marginal diagnostics along the training trajectory](src/iqp_bp/experiments/run_training.py)
  Depends on: `AC7`, `AC8`, `AC9`
  Current state: `run-training` is wired into the CLI, persists one run directory per resolved setting, and appends `ac_*` plus `marginal_orders` fields to every persisted trajectory row. Marginal sidecars are also written under `runs/<setting>/marginals/`.

- `[~] AC11` [Week 3] [Run learned-distribution anti-concentration cells in both exact small-`n` and sampled larger-`n` modes](configs/experiments/ghosh_kim_small_n.yaml)
  Depends on: `AC10`
  Current state: the reproducible configs and runner paths now exist for `ghosh_kim_small_n.yaml` and `ghosh_kim_large_n_sampled.yaml`. The code path is implemented; the long experiment results and written findings still need to be generated from actual runs.

- `[~] AC12` [Week 3] [Sweep Gaussian bandwidth and compare empirical per-order matching against Rudolph `tau^|S|`](configs/experiments/bandwidth_marginal_sweep.yaml)
  Depends on: `AC10`, `AC11`
  Current state: the Gaussian sigma-sweep config exists, the trajectory rows already expose the exact per-order `tau`-weighted contribution, and `docs/technical/bandwidth-marginals.md` documents how to interpret the artifacts. The remaining step is to execute the sweep and write the empirical conclusion.

### Qiskit Validation

- `[~] Q3` [Week 5] [Lift the Qiskit observable-level estimator primitives to full MMD^2 and gradient-SNR comparisons](src/iqp_bp/qiskit/estimators.py#L97)
  Depends on: `Q1`, `V4`
  Current state: statevector expectation, shot-based expectation, and parameter-shift are implemented at the observable level only.
  Docs to read first: [Qiskit primitives], [Qiskit SparsePauliOp], [AerSimulator], [Aer NoiseModel].

- `[x] Q4` [Week 5] [Implement the Qiskit validation runner for classical, statevector, shots, and noise comparisons](src/iqp_bp/experiments/run_qiskit.py#L44)
  Depends on: `Q1`, `Q3`, `P2`, `P4`
  Current state: `run_qiskit.py` now expands the validation grid over family, `n`, shot count, and noise rate, reuses one IQP model per coordinate under the named-stream RNG policy, computes classical/statevector/shot/noisy comparisons for one shared observable set, and relies on batch estimator helpers plus a config-name noise-model factory; smoke coverage exists in `tests/test_run_qiskit.py`.
  Docs to read first: [Qiskit QuantumCircuit], [Qiskit qasm2], [Qiskit transpile], [AerSimulator], [Aer NoiseModel], [json module].
  Implementation checklist:
  - `[x]` [src/iqp_bp/experiments/run_qiskit.py](src/iqp_bp/experiments/run_qiskit.py): replace the stub loop with a real coordinate expansion over family, `n`, shot count, and enabled noise settings, while preserving the named-stream RNG policy from `P2`.
  - `[x]` [src/iqp_bp/experiments/run_qiskit.py](src/iqp_bp/experiments/run_qiskit.py): for each coordinate, build the IQP model once, compute the classical exact reference, then run statevector, shot-based, and noisy-Qiskit comparisons against that same observable set and parameter vector.
  - `[x]` [src/iqp_bp/qiskit/estimators.py](src/iqp_bp/qiskit/estimators.py): lift the current observable-level primitives into runner-friendly helpers that can evaluate batches of observables and return runner-facing arrays and diagnostics.
  - `[x]` [src/iqp_bp/qiskit/noise.py](src/iqp_bp/qiskit/noise.py): expose a stable noise-model factory API that `run_qiskit.py` can call by config name when the validation grid includes noisy runs.
  - `[x]` [src/iqp_bp/config.py](src/iqp_bp/config.py) and [configs/schema.yaml](configs/schema.yaml): validate the `qiskit` block used by `run_qiskit`, including shot arrays, `max_n`, backend names, and noise configuration shapes, before the runner starts.
  - `[x]` [configs/experiments/qiskit_validation.yaml](configs/experiments/qiskit_validation.yaml): align the example config with the runner's real axes and artifact outputs so the checked-in config is runnable, not aspirational.
  - `[x]` [tests/test_run_qiskit.py](tests/test_run_qiskit.py): add a smoke test for the runner control flow with small `n`, asserting that comparison records are emitted with the expected coordinate fields and reproducible file outputs.

- `[x] Q5` [Week 5] [Build the Qiskit circuits and store the raw cross-check data](src/iqp_bp/experiments/run_qiskit.py#L53)
  Depends on: `Q4`
  Current state: the Qiskit runner now writes one JSONL summary row per validation coordinate, emits deterministic `setting_id`-based raw JSON sidecars plus QASM exports, includes raw shot counts when requested, and points each summary row at its saved sidecar artifacts.
  Docs to read first: [Qiskit QuantumCircuit], [Qiskit qasm2], [AerSimulator], [json module], [pathlib.Path].
  Implementation checklist:
  - `[x]` [src/iqp_bp/experiments/run_qiskit.py](src/iqp_bp/experiments/run_qiskit.py): persist one compact summary JSONL row per validation coordinate plus sidecar raw artifacts containing observable masks, classical expectations, Qiskit expectations, absolute errors, and shot-count histograms.
  - `[x]` [src/iqp_bp/experiments/run_qiskit.py](src/iqp_bp/experiments/run_qiskit.py): standardize output paths and filenames so raw cross-check data, circuit exports, and summary records can be joined by one stable setting identifier.
  - `[x]` [src/iqp_bp/qiskit/circuit_builder.py](src/iqp_bp/qiskit/circuit_builder.py): make the emitted QASM/transpile metadata available to the runner so saved circuit text can be debugged from sidecar outputs.
  - `[x]` [src/iqp_bp/qiskit/estimators.py](src/iqp_bp/qiskit/estimators.py): return raw counts or other low-level execution outputs when requested, so the stored cross-check artifacts capture enough evidence to explain disagreement between exact, statevector, and shot/noise regimes.
  - `[x]` [src/iqp_bp/iqp/model.py](src/iqp_bp/iqp/model.py) and [src/iqp_bp/experiments/run_validation.py](src/iqp_bp/experiments/run_validation.py): reuse existing provenance and checkpoint conventions where helpful so Qiskit artifacts encode the originating circuit family and deterministic seeds consistently with the classical validation path.
  - `[x]` [tests/test_run_qiskit.py](tests/test_run_qiskit.py): extend the runner tests to assert that raw artifact files are written, filenames are deterministic for one setting, and summary rows contain pointers to the saved raw data and circuit exports.

### Scaling v2 and Phase-2 Kernels

- `[~] K1` [Week 6] [Validate the multi-scale Gaussian kernel against the exact mixture formula](src/iqp_bp/mmd/kernel.py#L202)
  Depends on: `T1`, `V4`
  Current state: multi-scale Gaussian sampling and kernel evaluation exist, but they are not yet exact-validated.
  Docs to read first: [NumPy Generator], [pytest parametrize].

- `[~] K2` [Week 6] [Promote the multi-scale Gaussian config into the phase-2 experiment sweep](src/iqp_bp/experiments/run_scaling.py#L160)
  Depends on: `K1`, `S6`
  Current state: the scaling runner has a `multi_scale_gaussian` parameter branch, but phase-2 sweeps are not actually wired end-to-end.
  Docs to read first: [itertools.product], [pathlib.Path], [json module].

### Forge Sprint

- `[x] F2` [Week 7] [Replace export-only Forge mode with automated structural searches and machine-readable results](src/iqp_bp/experiments/run_forge.py#L37)
  Depends on: `F1`, `P4`
  Current state: the Forge runner exports instances and logs them, but does not yet perform automated structural searches.
  Docs to read first: [Forge docs], [Forge constraints], [Forge options], [json module], [pathlib.Path].

## Completed

- `T1`, `T3`, `S1`, `V1`, `Q1`, `Q4`, `Q5`

## Suggested Execution Waves

### Wave A - Parallelizable immediately

`T1`, `T2`, `T3`, `P1`, `P2`, `P3`, `P4`, `P5`, `AC1`, `AC2`, `S1`, `S2`, `S3`, `Q1`, `Q2`, `F1`

### Apr 8 10:30am critical path

`AC1`, `AC2`, `AC3`, `AC4`, `AC5`, `AC6`

### Wave B - Unlocked after Wave A

`P6`, `V1`, `V2`, `V3`, `V4`, `AC3`, `AC5`, `S4`, `S5`

### Wave C - Core experiment enablement

`V5`, `AC4`, `S6`, `S7`, `Q3`

### Wave D - Reporting and cross-check runs

`S8`, `AC6`, `Q4`, `K1`, `F2`

### Wave E - Final phase-2 execution tasks

`Q5`, `K2`

### Learned-Distribution AC Extension

Wave A:
`AC7`, `AC9`

Wave B:
`AC8`

Wave C:
`AC10`

Wave D:
`AC11`, `AC12`

[Hypothesis API]: https://hypothesis.readthedocs.io/en/latest/reference/api.html
[Hypothesis strategies]: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
[Hypothesis NumPy]: https://hypothesis.readthedocs.io/en/latest/numpy.html
[JAX grad]: https://docs.jax.dev/en/latest/_autosummary/jax.grad.html
[JAX value_and_grad]: https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html
[JAX vmap]: https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
[JAX jit]: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
[JAX random]: https://docs.jax.dev/en/latest/jax.random.html
[NumPy Generator]: https://numpy.org/doc/stable/reference/random/generator.html
[PyYAML docs]: https://pyyaml.org/wiki/PyYAMLDocumentation
[pathlib.Path]: https://docs.python.org/3/library/pathlib.html#pathlib.Path
[itertools.product]: https://docs.python.org/3/library/itertools.html#itertools.product
[json module]: https://docs.python.org/3/library/json.html
[pytest parametrize]: https://docs.pytest.org/en/7.1.x/how-to/parametrize.html
[NetworkX grid_2d_graph]: https://networkx.org/documentation/stable/reference/generated/networkx.generators.lattice.grid_2d_graph.html
[NetworkX erdos_renyi_graph]: https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html
[SciPy curve_fit]: https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.optimize.curve_fit.html
[Qiskit QuantumCircuit]: https://quantum.cloud.ibm.com/docs/api/qiskit/2.1/qiskit.circuit.QuantumCircuit
[Qiskit ParameterVector]: https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.0/qiskit.circuit.ParameterVector
[Qiskit qasm2]: https://docs.quantum.ibm.com/api/qiskit/qasm2
[Qiskit transpile]: https://quantum.cloud.ibm.com/docs/api/qiskit/0.39/qiskit.compiler.transpile
[Qiskit primitives]: https://quantum.cloud.ibm.com/docs/api/qiskit/dev/primitives
[Qiskit SparsePauliOp]: https://docs.quantum.ibm.com/api/qiskit/0.24/qiskit.quantum_info.SparsePauliOp
[AerSimulator]: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
[Aer NoiseModel]: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html
[Aer depolarizing_error]: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.depolarizing_error.html
[Aer ReadoutError]: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.ReadoutError.html
[Aer amplitude_damping_error]: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.amplitude_damping_error.html
[Aer thermal_relaxation_error]: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html
[Forge docs]: https://forge-fm.github.io/forge-documentation/
[Forge constraints]: https://forge-fm.github.io/forge-documentation/building-models/constraints/constraints.html
[Forge options]: https://forge-fm.github.io/forge-documentation/running-models/options.html
[Forge model intuition]: https://forge-fm.github.io/book/chapters/solvers/bounds_booleans_how_forge_works.html
