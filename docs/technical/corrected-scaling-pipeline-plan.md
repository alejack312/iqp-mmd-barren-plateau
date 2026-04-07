# Corrected Scaling Pipeline Plan

This plan supersedes the external draft for `S2`, `S4`, `S6`, and `AC6`.

It preserves the current config contract in [configs/schema.yaml](/C:/Users/cuqui/iqp-mmd-barren-plateau/configs/schema.yaml) and makes an explicit reuse decision for the existing dataset code under [src/iqp_mmd/datasets](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_mmd/datasets).

## Goals

1. Implement the missing structured datasets required by the scaling runner.
2. Expand the scaling runner to sweep the full documented experiment axes.
3. Add anti-concentration outputs to the scaling artifacts without changing the schema of existing gradient-variance records in a confusing way.
4. Avoid introducing a second, inconsistent dataset API.

## Guardrails

- Preserve the existing config surface:
  - `dataset.type` remains a scalar string, not a sweep axis.
  - `dataset.ising` continues to use `beta`, `coupling_std`, and `topology`.
  - `dataset.binary_mixture` continues to use `n_modes` and `noise`.
  - `init.scheme` remains one of `uniform`, `small_angle`, `data_dependent`.
- Keep dataset generation on-the-fly at runtime for the scaling runner. No file inputs.
- Do not reimplement dataset logic inline inside `run_scaling.py`.
- Do not silently strengthen the meaning of `data_dependent` init in the same patch unless the docs, tests, and Hypothesis layer are updated together.

## Reuse Decision

### Decision Summary

Introduce a small `iqp_bp` dataset factory module and have the scaling runner call that module. Reuse the `iqp_mmd.datasets` package selectively:

- Reuse [src/iqp_mmd/datasets/ising.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_mmd/datasets/ising.py) as a reference implementation and optional cross-check target in tests.
- Do not call `iqp_mmd.datasets.ising.generate_ising_*()` directly from the scaling runner in the first pass.
- Do not reuse [src/iqp_mmd/datasets/blobs.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_mmd/datasets/blobs.py) for `binary_mixture`.

### Why

`iqp_mmd.datasets` is real code and should not be ignored, but it does not line up cleanly with the `iqp_bp` scaling contract:

- [src/iqp_mmd/datasets/ising.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_mmd/datasets/ising.py) depends on optional `qml-benchmarks` internals at call time.
- Its API is training-package oriented: train/test split containers, `temperature`, `periodic`, `random_weights`, and chain settings.
- The scaling runner contract is experiment-package oriented: one binary sample matrix, `beta`, `coupling_std`, and `topology`, generated from the experiment seed.

So the correct reuse strategy is:

- keep `iqp_mmd.datasets` as the package-facing dataset layer,
- build a narrow `iqp_bp` runtime dataset factory for scaling experiments,
- add tests that compare small-n moment behavior against the existing `iqp_mmd` Ising implementation when optional dependencies are present.

`blobs.py` is intentionally not reused because it is a spin-blob generator, not the documented binarized Gaussian mixture.

## Implementation Plan

### Phase 0: Prerequisite alignment

Before touching `S2`/`S6`, close the practical gaps that those tasks already depend on:

- `P1`: resolved-config validation and persisted grid preview.
- `P2`: named seed streams for circuit, data, theta, kernel, and anti-concentration/checkpoint work.
- `S5`: central enforcement of the four-family primary sweep policy.

These can be minimal but should exist before the full Cartesian expansion lands, otherwise reproducibility and record provenance stay muddy.

### Phase 1: Add a dedicated scaling dataset factory

Add a new module, for example:

- [src/iqp_bp/experiments/data_factory.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/data_factory.py)

Public shape:

```python
def make_dataset(dataset_cfg: dict[str, Any], n: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    ...
```

Return:

- `data`: binary array of shape `(n_samples, n)`, dtype `uint8`
- `metadata`: JSON-serializable dataset provenance for result records

Supported dataset types:

#### `product_bernoulli`

- Native implementation in `iqp_bp`
- Uses `np.random.default_rng(seed)`
- Returns fair Bernoulli samples and metadata with `type`, `n_samples`, and seed information

#### `ising`

- Native implementation in `iqp_bp`, not inlined in `run_scaling.py`
- Must respect the existing config contract:
  - `beta`
  - `coupling_std`
  - `topology` in `{"grid_2d", "erdos_renyi"}`
- Sampling method:
  - use a simple native MCMC sampler suitable for runtime generation
  - multiple short chains are acceptable
  - seed handling must be deterministic through the named `data` RNG stream
- Coupling construction:
  - sample couplings with the documented scaling `J_ij ~ N(0, coupling_std^2 / n)`
  - build the interaction graph from `topology`
- Metadata:
  - include `beta`, `coupling_std`, `topology`, chain count, burn-in, and effective sample count

Reuse of `iqp_mmd.datasets.ising`:

- add optional tests that compare low-order statistics from the native `iqp_bp` Ising generator to [generate_ising_lattice](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_mmd/datasets/ising.py#L29) on small lattice cases when `qml-benchmarks` is installed
- do not make the scaling runner import that module at runtime

#### `binary_mixture`

- Native implementation in `iqp_bp`
- Must match the documented semantics:
  - sample `K = n_modes` Gaussian centers in `R^n`
  - sample Gaussian noise with standard deviation `noise`
  - binarize per coordinate to obtain `{0,1}^n`
- Do not implement this as "binary centers plus bit flips"
- Metadata:
  - include `n_modes`, `noise`, seed information, and any binarization threshold used

Non-goal:

- Do not reuse [src/iqp_mmd/datasets/blobs.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_mmd/datasets/blobs.py). It is a different target distribution.

### Phase 2: Refactor `run_scaling.py` to use explicit scalar settings

Update [src/iqp_bp/experiments/run_scaling.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_scaling.py) so it no longer reaches into list-valued config fields and takes `[0]`.

Explicit scalar axes to sweep:

- `circuit.family`
- `circuit.n_qubits`
- `kernel.type`
- `kernel.bandwidth` for `gaussian` and `laplacian`
- `init.scheme`
- `init.small_angle.std` when `scheme == "small_angle"`
- `circuit.erdos_renyi.p_edge` when `family == "erdos_renyi"`

Important non-axis:

- `dataset.type` remains a single chosen target for the run

Implementation shape:

- build a resolved grid using `itertools.product`
- materialize one fully explicit scalar setting object per coordinate
- pass scalar values into helper functions instead of letting helpers inspect list-valued config blocks

Refactor helpers accordingly:

- `_make_G(...)` should accept an explicit ER `p_edge` scalar when relevant
- `_get_kernel_params(...)` should accept an explicit `sigma` scalar when relevant
- `_make_theta(...)` should accept an explicit small-angle `std` scalar when relevant
- `_make_data(...)` should disappear in favor of the dataset factory

Record-keeping:

- every JSONL row must include the exact scalar coordinates used
- include dataset metadata in each row, or store a per-setting sidecar artifact and reference it from the row

### Phase 3: Keep `data_dependent` init stable in this change set

Do not replace the current `data_dependent` formula in this patch.

Current behavior:

```text
theta_j = scale * E_x[(-1)^(x·g_j)]
```

Why this stays:

- it is the behavior currently documented in [docs/technical/configuration.md](/C:/Users/cuqui/iqp-mmd-barren-plateau/docs/technical/configuration.md)
- it is mirrored in [src/iqp_bp/hypergraph/hypothesis_strategies.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/hypergraph/hypothesis_strategies.py)
- the stronger representative-`a` variant is still a design choice, not just an implementation detail

Corrected `S4` scope for this plan:

- keep the current parity-based initializer
- make it deterministic under the named theta/data RNG scheme
- emit init metadata into records
- if a stronger covariance initializer is desired later, add it as a separate, explicitly named init variant after docs and tests are updated together

This avoids mixing a research choice with the dataset and sweep refactor.

### Phase 4: Add AC6 artifacts without polluting the gradient-scaling schema

Extend the scaling runner so anti-concentration outputs are saved alongside scaling results, but kept structurally separate.

Output shape:

- `results.jsonl` remains the per-parameter gradient-variance record stream
- add a separate anti-concentration artifact path per setting, for example:
  - `anti_concentration/<setting-key>.json`
  - `anti_concentration/<setting-key>.csv`
  - optional checkpoint plots under `anti_concentration/plots/`

Each anti-concentration artifact should include:

- dataset provenance
- family
- `n`
- kernel
- init scheme
- the explicit scalar settings for bandwidth, ER `p_edge`, and small-angle `std` when applicable
- whether the anti-concentration result came from exact probabilities or sampled counts
- any checkpoint path used

Rows in `results.jsonl` should only include:

- compact summary fields
- a stable pointer to the richer anti-concentration artifact

This keeps downstream analysis from conflating gradient-scaling data with anti-concentration payloads.

## Verification Plan

### Automated tests

Add tests around the new dataset factory instead of testing dataset generation through `run_scaling.py` internals.

Recommended test files:

- [tests/test_scaling_data_factory.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_scaling_data_factory.py)
- [tests/test_run_scaling_grid.py](/C:/Users/cuqui/iqp-mmd-barren-plateau/tests/test_run_scaling_grid.py)

Coverage:

- `product_bernoulli`
  - shape
  - dtype
  - reproducibility from seed
- `ising`
  - shape and dtype
  - reproducibility from seed
  - non-trivial pairwise correlations on small grid cases
  - optional statistical cross-check against `iqp_mmd.datasets.ising` when its optional backend is available
- `binary_mixture`
  - shape and dtype
  - reproducibility from seed
  - mode structure sanity checks consistent with a binarized Gaussian mixture
- sweep resolution
  - exact count of resolved settings
  - ER `p_edge` only expands when `family == "erdos_renyi"`
  - small-angle `std` only expands when `init.scheme == "small_angle"`
  - dataset type is not expanded as a grid axis
- AC6 output
  - gradient rows contain stable artifact references
  - anti-concentration artifacts are written with exact scalar-setting metadata

### Manual verification

Do not use `--dry-run` to inspect JSONL output, because `--dry-run` only prints merged config and exits.

Instead:

1. run a very small real scaling job with:
   - 1-2 qubit sizes
   - 2 bandwidths
   - ER and non-ER families
   - `small_angle` and `uniform`
2. verify:
   - resolved run count matches expectation
   - JSONL rows contain scalar coordinates, not list-valued defaults
   - anti-concentration sidecar artifacts exist and match the row references

## Out of Scope

- Changing the meaning of `data_dependent` init to the representative-`a` formula
- Promoting `dataset.type` itself into a sweep axis
- Reworking the public `iqp_mmd` dataset API
- Replacing the runtime dataset factory with file-backed datasets

## Recommended Execution Order

1. `P1` and `P2`
2. add `iqp_bp` dataset factory
3. wire `run_scaling.py` to the factory and explicit scalar settings
4. land AC6 sidecar artifacts
5. only then revisit whether a stronger `data_dependent` initializer deserves a separate plan
