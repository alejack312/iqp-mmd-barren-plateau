# TODO Dependency Roadmap

Dependency-aware execution list derived from `docs/SMART-spec.md` and the `TODO:` markers in `src/iqp_bp`.

Audit status as of 2026-03-29:
- No TODOs are fully complete yet.
- Many TODOs have partial scaffolding already present in the codebase.

Legend:
- `[x]` complete
- `[~]` partial
- `[ ]` open

How to read this file:
- `Ready now` means the task does not depend on any other current TODO in `src/iqp_bp`.
- `Depends on` means the task should be done after the listed TODO IDs.
- IDs are stable planning labels for this file only.

## Ready Now

### Foundation and Theory

- `[ ] T1` [Week 1] [Confirm the exact Gaussian spectral normalization used by the locked MMD^2 derivation](src/iqp_bp/mmd/kernel.py#L34)
  Current state: Gaussian weights and sampling are implemented, but the derivation-to-code normalization has not been explicitly locked down.

- `[~] T2` [Week 1] [Keep the Laplacian kernel path as an explicit stub until its MMD^2 decomposition is derived and checked](src/iqp_bp/mmd/kernel.py#L92)
  Current state: there is an approximate Laplacian implementation, but it is not yet a clearly validated or theory-locked final path.

- `[ ] T3` [Week 1] [Replace the generic 2D patch sampler with the exact nearest-neighbour ZZ lattice family in scope](src/iqp_bp/hypergraph/families.py#L93)
  Current state: a 2D lattice-style generator exists, but it is still a generic patch sampler rather than the exact scope-locked family.

### Reproducibility and Core Pipeline

- `[~] P1` [Week 1] [Add config validation and a persisted resolved experiment grid for reproducible reruns](src/iqp_bp/config.py#L27)
  Current state: config loading and deep-merge are implemented, but schema validation and resolved-grid persistence are missing.

- `[~] P2` [Week 1] [Reserve named RNG streams for circuit, data, theta, kernel, and Qiskit sampling](src/iqp_bp/rng.py#L25)
  Current state: deterministic seed splitting exists, but streams are not named or reserved by responsibility.

- `[~] P3` [Week 1] [Add stable batching or streaming to the classical IQP expectation engine](src/iqp_bp/iqp/expectation.py#L69)
  Current state: the expectation engine is vectorized, but it does not yet batch or stream to control memory explicitly.

- `[ ] P4` [Week 1] [Preserve family and generation metadata on IQP models for experiment and Qiskit provenance](src/iqp_bp/iqp/model.py#L41)
  Current state: `IQPModel` wraps `G` and `theta`, but it does not preserve family or provenance metadata.

- `[~] P5` [Week 1] [Expose per-observable contributions and confidence diagnostics in the MMD^2 estimator](src/iqp_bp/mmd/loss.py#L46)
  Current state: the MMD^2 estimator exists, but it only returns a scalar and does not expose contribution-level diagnostics.

### Scaling Inputs

- `[~] S1` [Weeks 3-4] [Calibrate the sparse Erdos-Renyi family to the SMART bounded-degree regime](src/iqp_bp/hypergraph/families.py#L58)
  Current state: an Erdos-Renyi generator exists with configurable `p_edge`, but it is not yet calibrated to the SMART bounded-expected-degree regime.

- `[~] S2` [Weeks 3-4] [Implement the Ising-like synthetic target and the structured real or binary-mixture target](src/iqp_bp/experiments/run_scaling.py#L130)
  Current state: dataset config placeholders exist, but only product Bernoulli data is actually implemented.

- `[~] S3` [Weeks 3-4] [Add cached parity statistics and structured target-data helpers on the data side of MMD](src/iqp_bp/mmd/mixture.py#L39)
  Current state: batched dataset parity expectations are implemented, but caching and structured target-data helpers are missing.

### Qiskit and Forge Prep

- `[~] Q1` [Week 5] [Split measured vs unmeasured Qiskit builders and emit QASM and transpilation metadata](src/iqp_bp/qiskit/circuit_builder.py#L55)
  Current state: a Qiskit circuit builder exists, but it always measures and does not emit export metadata.

- `[~] Q2` [Week 6] [Add amplitude-damping and backend-inspired Qiskit noise presets](src/iqp_bp/qiskit/noise.py#L64)
  Current state: depolarizing, readout, and combined noise models exist, but amplitude damping and backend-like presets do not.

- `[~] F1` [Week 7] [Emit overlap-graph, degree-constraint, and threshold facts in Forge exports](src/iqp_bp/forge/export_instances.py#L34)
  Current state: Forge export exists for qubits, generators, and containment, but not for overlap-graph or threshold facts.

## Depends On Other TODOs

### Week 1 Follow-Through

- `[~] P6` [Week 1] [Add a config-validation or grid-preview CLI subcommand](src/iqp_bp/cli.py#L21)
  Depends on: `P1`
  Current state: the CLI already has a `--dry-run` flag, but not a dedicated grid-preview or validation command.

### Validation Layer

- `[~] V1` [Week 2] [Add Hypothesis strategies for the four SMART circuit families](src/iqp_bp/hypergraph/hypothesis_strategies.py#L27)
  Depends on: `T3`, `S1`
  Current state: generic and bounded-degree Hypothesis strategies exist, but not the full SMART family set.

- `[~] V2` [Week 2] [Wire the exact IQP expectation path into automated Monte Carlo vs exact validation plots](src/iqp_bp/iqp/expectation.py#L97)
  Depends on: `P3`
  Current state: the exact IQP expectation path exists, but there is no automated validation harness or plotting.

- `[~] V3` [Week 2] [Implement the JAX autodiff gradient estimator and compare it to the analytic path](src/iqp_bp/mmd/gradients.py#L45)
  Depends on: `P5`
  Current state: analytic gradients and finite differences exist, and JAX support is hinted at in RNG utilities, but autodiff is not implemented here.

- `[~] V4` [Week 2] [Add an exact small-n MMD^2 path for brute-force kernel validation](src/iqp_bp/mmd/loss.py#L62)
  Depends on: `T1`, `T2`, `P5`
  Current state: MMD^2 estimation exists, but only through sampled observables and sampled expectations.

- `[~] V5` [Week 2] [Add Hypothesis coverage for data-dependent init and the full small-angle sweep](src/iqp_bp/hypergraph/hypothesis_strategies.py#L69)
  Depends on: `S4`
  Current state: uniform and small-angle strategies exist, but the full sweep and data-dependent init are missing.

### Scaling v1

- `[~] S4` [Weeks 3-4] [Replace the data-dependent init stub with the covariance-informed initializer](src/iqp_bp/experiments/run_scaling.py#L146)
  Depends on: `S2`
  Current state: a stubbed data-dependent init branch exists, but it is still just a small Gaussian draw.

- `[~] S5` [Weeks 3-4] [Enforce the primary four-family sweep and comparable parameter-count policies centrally](src/iqp_bp/hypergraph/families.py#L240)
  Depends on: `T3`, `S1`
  Current state: the four primary families are present and used in configs, but the policy is not centrally enforced.

- `[~] S6` [Weeks 3-4] [Expand the scaling runner to sweep the full Cartesian grid of experiment axes](src/iqp_bp/experiments/run_scaling.py#L39)
  Depends on: `P1`, `P2`, `S2`, `S4`, `S5`
  Current state: the scaling runner loops over families, kernels, inits, and `n`, but it still collapses several axis lists to a single value.

- `[~] S7` [Weeks 3-4] [Extend gradient-variance summaries with aggregate norm proxies and heavy-tail diagnostics](src/iqp_bp/mmd/gradients.py#L164)
  Depends on: `V3`
  Current state: mean, variance, std, and median are reported, but the richer plateau diagnostics are missing.

- `[~] S8` [Weeks 3-4] [Fit polynomial vs exponential scaling and emit the summary artifacts for the interim memo](src/iqp_bp/experiments/run_scaling.py#L86)
  Depends on: `S6`, `S7`
  Current state: raw JSONL records are written, but no fitting or memo-oriented summaries are produced.

### Qiskit Validation

- `[~] Q3` [Week 5] [Lift the Qiskit observable-level estimator primitives to full MMD^2 and gradient-SNR comparisons](src/iqp_bp/qiskit/estimators.py#L97)
  Depends on: `Q1`, `V4`
  Current state: statevector expectation, shot-based expectation, and parameter-shift are implemented at the observable level only.

- `[~] Q4` [Week 5] [Implement the Qiskit validation runner for classical, statevector, shots, and noise comparisons](src/iqp_bp/experiments/run_qiskit.py#L44)
  Depends on: `Q1`, `Q3`, `P2`, `P4`
  Current state: the runner file and imports exist, but the run path is still a stub.

- `[~] Q5` [Week 5] [Build the Qiskit circuits and store the raw cross-check data](src/iqp_bp/experiments/run_qiskit.py#L53)
  Depends on: `Q4`
  Current state: the ingredients exist, but the runner does not yet execute the comparisons or persist real records.

### Scaling v2 and Phase-2 Kernels

- `[~] K1` [Week 6] [Validate the multi-scale Gaussian kernel against the exact mixture formula](src/iqp_bp/mmd/kernel.py#L202)
  Depends on: `T1`, `V4`
  Current state: multi-scale Gaussian sampling and kernel evaluation exist, but they are not yet exact-validated.

- `[~] K2` [Week 6] [Promote the multi-scale Gaussian config into the phase-2 experiment sweep](src/iqp_bp/experiments/run_scaling.py#L160)
  Depends on: `K1`, `S6`
  Current state: the scaling runner has a `multi_scale_gaussian` parameter branch, but phase-2 sweeps are not actually wired end-to-end.

### Forge Sprint

- `[~] F2` [Week 7] [Replace export-only Forge mode with automated structural searches and machine-readable results](src/iqp_bp/experiments/run_forge.py#L37)
  Depends on: `F1`, `P4`
  Current state: the Forge runner exports instances and logs them, but does not yet perform automated structural searches.

## Completed

- None confirmed complete in the current audit.

## Suggested Execution Waves

### Wave A - Parallelizable immediately

`T1`, `T2`, `T3`, `P1`, `P2`, `P3`, `P4`, `P5`, `S1`, `S2`, `S3`, `Q1`, `Q2`, `F1`

### Wave B - Unlocked after Wave A

`P6`, `V1`, `V2`, `V3`, `V4`, `S4`, `S5`

### Wave C - Core experiment enablement

`V5`, `S6`, `S7`, `Q3`

### Wave D - Reporting and cross-check runs

`S8`, `Q4`, `K1`, `F2`

### Wave E - Final phase-2 execution tasks

`Q5`, `K2`

