# P1-P5 Implementation Notes

This note explains what was implemented for `P1` through `P5`, what each change is responsible for in the codebase, and why the implementation was done that way.

The audience for this document is a future maintainer who opens the code and wants to answer three questions quickly:

- What did `P1`-`P5` actually add?
- Why do the implementations look the way they do?
- Which files are the canonical places to inspect when something in this area needs to change?

The short version:

- `P1` made the scaling pipeline reproducible at the config level.
- `P2` made randomness reproducible at the stream level.
- `P3` made the classical expectation engine bounded-memory.
- `P4` made circuit provenance explicit and serializable.
- `P5` made MMD estimates inspectable instead of opaque.

Together, these changes move the repo from "the experiments run" to "the experiments can be rerun, debugged, and defended."

## Why these five items matter together

These tasks are tightly coupled even though they live in different files.

If a scaling run cannot be reconstructed from one exact config and one exact resolved grid, the output is not really reproducible. That is `P1`.

If the config is fixed but one refactor silently changes which random numbers get consumed by kernel sampling, theta initialization, or expectation estimation, the output is still not reproducible. That is `P2`.

If the expectation engine allocates arrays proportional to the full Monte Carlo budget, the classical regime stops scaling long before the theory does. That is `P3`.

If a checkpoint or validation artifact cannot tell us which family produced the circuit, we lose traceability between training, validation, and Qiskit paths. That is `P4`.

If an MMD estimate returns only one scalar, we cannot tell whether a change affected the kernel-word mixture, the dataset expectation path, or the IQP expectation path. That is `P5`.

So the implementation strategy was not to add isolated features. It was to tighten the whole experiment loop until each run had:

- a validated input contract
- deterministic randomness by responsibility
- bounded-memory estimation
- explicit circuit provenance
- inspectable estimator diagnostics

## P1: Config Validation And Persisted Resolved Grids

Primary code:

- [src/iqp_bp/config.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/config.py)
- [src/iqp_bp/cli.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/cli.py)
- [src/iqp_bp/experiments/run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_scaling.py)
- [configs/schema.yaml](/C:/Users/cuqui/iqp-mmd-barren-plateau/configs/schema.yaml)

Tests:

- [tests/test_config.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_config.py)
- [tests/test_run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_run_scaling.py)

### What was implemented

`P1` added four concrete capabilities:

1. Config loading now validates the merged config instead of only deep-merging YAML.
2. The scaling grid is resolved in config space by `resolve_experiment_grid(...)`, not only inside runner internals.
3. The merged config and resolved grid are persisted to `config.json` and `manifest.json` before the long run begins.
4. The CLI preview path now prints the resolved scalar settings, which is the actual experiment surface, not just the merged nested YAML.

The most important current behavior is:

- `load_config(...)` validates the merged config against the repo contract.
- `resolve_experiment_grid(...)` produces the Cartesian product of the actual sweep axes.
- `persist_experiment_manifest(...)` writes the exact merged config and exact resolved grid used for the run.
- `run_scaling.py` writes `manifest_path` into output records so a result row can be traced back to one manifest.

### Why it was implemented this way

The point of `P1` is not "have validation." The point is "make the experiment contract inspectable and reproducible."

The implementation deliberately keeps validation in `config.py`, not in `run_scaling.py`, because config validity is a property of the configuration layer, not of one runner. That keeps runners from re-implementing partial checks.

Validation is schema-informed but not schema-naive. The YAML schema is used as the structural source of truth, but the code still has explicit enum sets and runner-aware checks. That design is intentional:

- the YAML file expresses the public config surface
- the Python code enforces the parts that matter operationally
- the runner does not have to guess whether a field is a scalar, list, enum, or nullable list

Another important design choice is that sweep axes such as `circuit.family`, `kernel.type`, and `init.scheme` are accepted as either scalars or lists, while truly scalar config like `dataset.type` remains scalar. That matches the actual experiment model:

- some fields define the sweep
- some fields define one chosen target or mode for the whole run

The resolved grid is persisted before any records are emitted because reproducibility should not depend on a run finishing successfully. If a run crashes halfway through, we still want:

- the exact merged config
- the exact scalar grid that was intended

### Audit note

During the audit, `P1` turned out to be the one item that was not actually complete. The shipped `scaling_v1.yaml` used list-valued sweep axes, but the validation path treated `circuit.family` as a scalar enum and crashed on load. That is now fixed in [src/iqp_bp/config.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/config.py).

## P2: Named RNG Streams By Responsibility

Primary code:

- [src/iqp_bp/rng.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/rng.py)
- [src/iqp_bp/experiments/run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_scaling.py)
- [src/iqp_bp/mmd/loss.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/loss.py)
- [src/iqp_bp/mmd/gradients.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/gradients.py)
- [src/iqp_bp/qiskit/noise.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/qiskit/noise.py)

Tests:

- [tests/test_rng.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_rng.py)
- [tests/test_run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_run_scaling.py)

### What was implemented

`P2` formalized the repo's canonical random streams:

- `circuit`
- `data`
- `theta`
- `kernel`
- `estimation`
- `qiskit`

The core helpers are:

- `derive_seed(...)`
- `named_seed_streams(...)`
- `experiment_stream_bundle(...)`
- `split_rng(...)`

The scaling runner now derives stream seeds from the experiment coordinate rather than from loop order. The MMD code splits kernel-word sampling and IQP expectation sampling into separate RNG streams. Gradient estimation also uses child generators rather than one mutable shared generator.

### Why it was implemented this way

The main failure mode being prevented here is hidden coupling through RNG consumption order.

Without named streams, a harmless refactor can change results by accident:

- adding a diagnostic draw
- changing a loop nesting order
- changing `num_a_samples`
- changing how many gradient parameters are inspected

Those are not scientific changes, but they can change the random draws if one mutable RNG is reused everywhere.

That is why `P2` uses two layers of control:

1. named integer seeds from experiment coordinates
2. local splitting into child generators for independent sub-responsibilities

This gives two benefits:

- reruns of the same experiment coordinate are stable
- one estimator path can change its local sampling pattern without shifting unrelated paths

The naming is by responsibility, not by module, on purpose. The stable thing is not "which file asked for the seed." The stable thing is "what the randomness is used for."

## P3: Stable Batching And Streaming In The Classical IQP Expectation Engine

Primary code:

- [src/iqp_bp/iqp/expectation.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/iqp/expectation.py)
- [src/iqp_bp/mmd/loss.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/loss.py)
- [src/iqp_bp/mmd/gradients.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/gradients.py)

Tests:

- [tests/test_expectation_small_n.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_expectation_small_n.py)
- [tests/test_expectation_streaming.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_expectation_streaming.py)

### What was implemented

`iqp_expectation(...)` now supports `batch_size`. When `batch_size` is provided, the estimator:

- samples `z` in chunks
- computes cosine samples chunk by chunk
- accumulates the mean and variance online
- returns the same public contract: `(estimate, stderr)`

The same batching control is threaded through:

- `mmd2(...)`
- `grad_expectation_analytic(...)`
- `grad_mmd2_analytic(...)`
- `grad_mmd2_finite_diff(...)`
- `estimate_gradient_variance(...)`

I also added guardrails during the audit so that invalid inputs such as `num_z_samples <= 0` or `batch_size <= 0` fail explicitly instead of drifting into bad behavior.

### Why it was implemented this way

The core design decision was: keep the estimator interface stable while changing the memory behavior.

That is why the public return value stayed `(estimate, stderr)` instead of introducing a separate "streaming estimator object" or a different result shape. Callers care about the expectation and its uncertainty, not about whether the samples were accumulated in one array or in chunks.

The implementation uses online accumulation because the bottleneck is not the phase formula itself. The bottleneck is holding all Monte Carlo samples in memory at once for large `num_z_samples`.

The chosen strategy preserves:

- identical semantics for the estimate
- deterministic behavior for a fixed RNG and batch policy
- bounded memory proportional to `batch_size * n`

This is exactly the right tradeoff for the classical path in this repo. The goal is not to invent a new estimator. The goal is to make the current estimator usable at larger sample budgets.

## P4: Model Provenance For Experiments, Validation, And Qiskit

Primary code:

- [src/iqp_bp/iqp/model.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/iqp/model.py)
- [src/iqp_bp/experiments/run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_scaling.py)
- [src/iqp_bp/experiments/run_validation.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_validation.py)

Tests:

- [tests/test_iqp_model_provenance.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_iqp_model_provenance.py)
- [tests/test_run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_run_scaling.py)

### What was implemented

`IQPModel` now carries a `provenance` dict alongside `G` and `theta`.

`IQPModel.from_family(...)` records fields such as:

- `family`
- `n`
- `m_requested`
- `m_generated`
- `family_kwargs`
- `rng_seed` when provided

Checkpoint save/load paths in validation serialize this provenance as JSON so a reloaded model still knows where it came from.

The scaling runner uses provenance-preserving model construction rather than reconstructing metadata manually later.

### Why it was implemented this way

The main design choice here is that provenance belongs on the model object, not beside it.

If provenance is tracked separately, it always drifts:

- one code path forgets to pass it through
- one artifact writes only a subset
- one validation path reconstructs it differently

Putting provenance on `IQPModel` makes the object self-describing. That is especially important because the repo now has multiple downstream consumers of the same logical circuit:

- scaling experiments
- anti-concentration validation
- checkpoint export/import
- Qiskit-side validation

The provenance format is JSON-serializable by design. That sounds obvious, but it matters because many upstream values are NumPy scalars, arrays, or other objects that are not safe to dump directly. The implementation coerces provenance into plain Python types before storing it. That avoids the very common failure mode where metadata exists in memory but cannot be cleanly written to disk.

The presence of both `m_requested` and `m_generated` is also intentional. Some families have intrinsic row count. If we only record one `m`, we lose the distinction between:

- what the runner asked for
- what the family actually generated

That distinction is useful for debugging and for explaining why exact families sometimes ignore caller-provided `m`.

## P5: Detailed MMD Diagnostics Instead Of Scalar-Only Estimates

Primary code:

- [src/iqp_bp/mmd/loss.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/loss.py)
- [src/iqp_bp/mmd/gradients.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/gradients.py)
- [src/iqp_bp/experiments/run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_scaling.py)

Tests:

- [tests/test_mmd_loss_details.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_mmd_loss_details.py)
- [tests/test_hypothesis.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_hypothesis.py)

### What was implemented

`mmd2(...)` still returns a scalar by default, but it now supports `return_details=True`.

The detailed path returns:

- `mmd2`
- `a_samples`
- `exp_p`
- `exp_q`
- `contributions`
- `mc_diagnostics`

The Monte Carlo diagnostics include:

- point estimate
- sample standard deviation
- standard error
- number of sampled observables

Gradient code can now reuse precomputed `a_samples` and `exp_p` so gradient debugging and MMD debugging can stay aligned on the same sampled observables.

The scaling runner explicitly keeps these detailed arrays in memory only and does not dump them into `results.jsonl`.

### Why it was implemented this way

The most important design constraint for `P5` was non-breaking introspection.

Existing callers expected `mmd2(...)` to return a float. That contract was preserved, because the scalar path is still the main path used by training and sweeps.

The detailed path exists to answer debugging questions that the scalar cannot answer:

- Which sampled observables contributed most?
- Did `exp_p` shift, or did `exp_q` shift?
- Did the kernel-word mixture change?
- Is the Monte Carlo uncertainty acceptable?

The detailed return shape is estimator-centered rather than file-centered. It returns exactly the objects needed to reconstruct the scalar:

```text
mmd2 = mean((exp_p - exp_q)^2)
```

That makes the diagnostics mechanically trustworthy instead of "summary stats that hopefully match the scalar."

Another deliberate choice was to keep detailed diagnostics out of `results.jsonl`. Those arrays scale with `num_a_samples`, so persisting them inline would make the main results stream noisy and heavy. The repo chose compact result rows plus optional in-memory diagnostics, which is the right default for large sweeps.

## Why these implementations are conservative

Across all five tasks, the implementation style is intentionally conservative:

- preserve existing public contracts where possible
- move logic to the layer where it conceptually belongs
- make determinism explicit
- add diagnostics without bloating the main artifact path

This repo is doing experiment work, not just library work. That changes what "good implementation" means.

In a pure library, it can be fine to rely on callers to compose reproducibility correctly. In an experiment repo, the code has to make the correct thing the default thing. That is why `P1` through `P5` emphasize:

- validated config boundaries
- deterministic seed derivation
- self-describing models
- inspectable estimators

## Current Status After Audit

As of the latest audit:

- `P2` through `P5` were already substantively implemented and verified.
- `P1` had one real defect: the validation path did not accept the repo's real sweep-axis config style used by `scaling_v1.yaml`.
- that defect is now fixed, and the `grid-preview` path for the shipped scaling config works again

So the current interpretation is:

- `P1` is now complete in the intended sense
- `P2` is complete
- `P3` is complete
- `P4` is complete
- `P5` is complete

## Where To Start If You Need To Change These Later

If you need to modify one of these areas, start here:

- Config contract or sweep behavior: [src/iqp_bp/config.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/config.py)
- Stream policy or reproducibility bugs: [src/iqp_bp/rng.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/rng.py)
- Classical expectation scaling or memory behavior: [src/iqp_bp/iqp/expectation.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/iqp/expectation.py)
- Provenance or checkpoint metadata: [src/iqp_bp/iqp/model.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/iqp/model.py) and [src/iqp_bp/experiments/run_validation.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_validation.py)
- MMD diagnostics or debugging alignment: [src/iqp_bp/mmd/loss.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/loss.py) and [src/iqp_bp/mmd/gradients.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/gradients.py)

If you only want the terse implementation checklist, see [TODOS.md](/C:/Users/cuqui/iqp-mmd-barren-plateau/TODOS.md).

If you want the broader design reasoning around the scaling pipeline, also read [corrected-scaling-pipeline-plan.md](/C:/Users/cuqui/iqp-mmd-barren-plateau/docs/technical/corrected-scaling-pipeline-plan.md) and [implementation-choices.md](/C:/Users/cuqui/iqp-mmd-barren-plateau/docs/technical/implementation-choices.md).
