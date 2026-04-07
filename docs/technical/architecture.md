# Code architecture

The code lives under `src/iqp_bp/`. This note is the map. If you want the reasoning behind the recent SMART-family, Gaussian-kernel, and Hypothesis changes, read [implementation-choices.md](./implementation-choices.md) next.

## Package layout

```text
src/iqp_bp/
|-- config.py                       # YAML loading and config merging
|-- cli.py                          # Entry point: three subcommands
|-- rng.py                          # Seeding utilities
|-- hypergraph/
|   |-- families.py                 # G matrix generators
|   `-- hypothesis_strategies.py    # Property-based test strategies
|-- iqp/
|   |-- model.py                    # IQPModel wrapper
|   `-- expectation.py              # Classical <Z_a> formula
|-- mmd/
|   |-- kernel.py                   # Kernel helpers and Z-word samplers
|   |-- loss.py                     # MMD^2 estimator
|   |-- mixture.py                  # Dataset parity expectations
|   `-- gradients.py                # Analytical gradients + variance estimation
|-- qiskit/
|   `-- circuit_builder.py          # G matrix -> Qiskit QuantumCircuit
`-- experiments/
    |-- run_scaling.py              # Main scaling sweep
    |-- run_qiskit.py               # Qiskit validation
    `-- run_forge.py                # Forge structural modeling
```

## How a run works

Everything starts from a config file. The CLI loads it, merges it with `configs/base.yaml`, and dispatches to one of three runners.

```bash
python -m iqp_bp.cli run-scaling configs/experiments/scaling_v1.yaml
```

Inside `run_scaling.py`, the computation is a nested loop over `(family, kernel, init, n)`. For each combination:

1. `config.py:load_config()` deep-merges the experiment YAML over `configs/base.yaml`.
2. `hypergraph/families.py:make_hypergraph(family, n, m)` builds the binary generator matrix `G`.
3. A dataset of binary strings is generated on the fly.
4. `_make_theta(...)` builds `theta` for the chosen init scheme.
5. `mmd/gradients.py:estimate_gradient_variance(...)` computes variance over the `theta` ensemble.
6. Each gradient evaluation calls `mmd/kernel.py:sample_a(...)` for Z-word masks, `mmd/mixture.py:dataset_expectations_batch(...)` for the data side, and `iqp/expectation.py:iqp_expectation(...)` for the model side.
7. Results are written as JSONL records to `results/`.

## The G matrix

Most of the project reduces to two cheap binary operations on `G`:

```python
(G @ a) % 2   # shape (m,): which generators have odd overlap with Z-word a
(G @ z) % 2   # shape (m,): which generators have odd overlap with random bitstring z
```

Those two overlap patterns drive the cosine formula for `<Z_a>_{q_theta}` and therefore the MMD estimator and its gradients.

## Modules

**`config.py`** loads YAML, deep-merges over defaults, and returns a plain Python dict. The schema lives at `configs/schema.yaml`.

**`hypergraph/families.py`** contains the primary SMART families (`product_state`, `lattice`, `erdos_renyi`, `complete_graph`) plus older legacy families. The dispatcher is `make_hypergraph(family, n, m, **kwargs)`. Some families determine their own row count, so downstream code should use `G.shape[0]` instead of assuming the requested `m` survived unchanged.

**`iqp/expectation.py`** has the Monte Carlo and exact small-`n` paths for `<Z_a>_{q_theta}`. `iqp_expectation(...)` returns `(mean, stderr)`. `iqp_expectation_exact(...)` enumerates all bitstrings and is only practical for small systems.

**`mmd/kernel.py`** does two jobs. It exposes direct kernel evaluators such as `gaussian_kernel(...)` and `multi_scale_gaussian_kernel(...)`, and it exposes the Z-word samplers used by the Monte Carlo MMD path through `sample_a(...)`. Keeping both in one module makes it easier to keep the documented Gaussian convention, the direct kernel formula, and the sampling path in sync.

**`mmd/mixture.py`** computes `<Z_a>_p` from dataset samples. `dataset_expectations_batch(data, a_batch)` is the vectorized data-side path.

**`mmd/loss.py`** estimates `MMD^2(p, q_theta)` by sampling Z-words from the kernel-induced distribution, then comparing data-side and model-side expectations.

**`mmd/gradients.py`** contains the analytic gradient estimator and the outer variance summary. `grad_mmd2_analytic(...)` computes one parameter gradient. `estimate_gradient_variance(...)` wraps that over a `theta` ensemble.

**`qiskit/circuit_builder.py`** maps a binary generator matrix to a Qiskit circuit. It is only used in the Qiskit validation path.

**`rng.py`** centralizes seeded NumPy generators and deterministic seed splitting.

## The three runners

**`run_scaling.py`** is the main experiment runner. It sweeps the configured grid and writes one JSONL record per `(family, kernel, init, n, param_idx)`.

**`run_qiskit.py`** is the small-system cross-check runner. It compares the classical formula against Qiskit statevector, shot-based, and noisy estimates.

**`run_forge.py`** exports or searches small structural instances for Forge-based analysis.

## Testing

`hypergraph/hypothesis_strategies.py` provides both generic and SMART-scoped strategies. The current validation layer uses SMART-only family sampling through `smart_family_instance()` and `mmd_instance()`, so Hypothesis exercises the exact 2D lattice, sparse pairwise Erdos-Renyi, product-state, and complete-graph families rather than arbitrary placeholder matrices.

The test suite focuses on invariants, not frozen numeric outputs. That includes:
- expectation values staying in `[-1, 1]`
- exact lattice edge structure
- pairwise ER structure with no duplicate edges
- `theta.shape == G.shape[0]`
- consistency between the Monte Carlo cosine path and the exact path on small `n`

That style of testing is deliberate. The implementation still moves, but the mathematics should not.
