# Code architecture

The codebase is organized as a Python package at `src/iqp_bp/`. Here's the full layout and how the pieces fit together.

## Package layout

```
src/iqp_bp/
├── config.py                       # YAML loading and config merging
├── cli.py                          # Entry point: three subcommands
├── rng.py                          # Seeding utilities
├── hypergraph/
│   ├── families.py                 # G matrix generators
│   └── hypothesis_strategies.py   # Property-based test strategies
├── iqp/
│   ├── model.py                    # IQPModel wrapper
│   └── expectation.py              # Classical ⟨Z_a⟩ formula
├── mmd/
│   ├── kernel.py                   # Z-word samplers per kernel
│   ├── loss.py                     # MMD² estimator
│   ├── mixture.py                  # Dataset parity expectations
│   └── gradients.py                # Analytical gradients + variance estimation
├── qiskit/
│   └── circuit_builder.py          # G matrix → Qiskit QuantumCircuit
└── experiments/
    ├── run_scaling.py              # Main scaling sweep
    ├── run_qiskit.py               # Qiskit validation
    └── run_forge.py                # Forge structural modeling
```

## How a run works

Everything starts from a config file. The CLI loads it, merges it with `configs/base.yaml`, and dispatches to one of three runners.

```bash
python -m iqp_bp.cli run-scaling configs/experiments/scaling_v1.yaml
```

Inside `run_scaling.py`, the computation is a nested loop over (family × kernel × init × n). For each combination:

1. `config.py:load_config()` deep-merges the experiment YAML over `configs/base.yaml`.
2. `hypergraph/families.py:make_hypergraph(family, n, m)` builds the binary generator matrix G of shape (m, n).
3. A dataset of N binary strings is generated on-the-fly (no file I/O).
4. `_make_theta(init_scheme, m, init_cfg, seed)` draws m parameter values. This repeats `num_seeds` times to build the θ ensemble.
5. `mmd/gradients.py:estimate_gradient_variance(G, data, param_idx, theta_seeds, ...)` computes variance over the ensemble.
6. Each gradient evaluation calls `mmd/kernel.py:sample_a(...)` for Z-word masks, `mmd/mixture.py:dataset_expectations_batch(...)` for the data side, and `iqp/expectation.py:iqp_expectation(...)` for the model side.
7. Results are written as JSONL records to `results/`.

## The G matrix

Everything interesting reduces to two operations on G:

```python
(G @ a) % 2   # shape (m,): which generators have odd overlap with Z-word a
(G @ z) % 2   # shape (m,): which generators have odd overlap with random bitstring z
```

Cheap binary arithmetic, independent of n and m scale. The entire cosine formula for ⟨Z_a⟩_{q_θ} is built from these two dot-products.

## Modules

**`config.py`** — loads YAML, deep-merges over defaults, returns a plain Python dict. The schema lives at `configs/schema.yaml` and documents every field.

**`hypergraph/families.py`** — four primary G matrix generators (`product_state`, `lattice`, `erdos_renyi`, `complete_graph`) plus four legacy ones. The dispatcher is `make_hypergraph(family, n, m, **kwargs)`. New families get registered in the `FAMILIES` dict — the runner picks them up automatically.

**`iqp/expectation.py`** — two functions. `iqp_expectation(theta, G, a, num_z_samples)` runs the Monte Carlo cosine formula and returns a (mean, stderr) tuple. `iqp_expectation_exact(theta, G, a)` does the full 2^n sum for validation, but only works up to n ≈ 20. Both delegate the phase computation to `iqp_phase(theta, G, z, a)`.

**`mmd/kernel.py`** — Z-word samplers, one per kernel type. The public interface is `sample_a(kernel, n, B, **kernel_params)`. Note that this module does NOT evaluate k(x, y) directly — it only samples the Z-word masks needed for the Monte Carlo estimator of MMD². The connection is in `docs/technical/mmd-gaussian-fourier.md`.

**`mmd/mixture.py`** — computes ⟨Z_a⟩_p from dataset samples. `dataset_expectations_batch(data, a_batch)` handles a whole batch of masks in one vectorized pass via parity arithmetic.

**`mmd/gradients.py`** — the outer loop spends most of its time here. `grad_mmd2_analytic(theta, G, data, param_idx, ...)` computes the analytical gradient of MMD² for one parameter. `estimate_gradient_variance(G, data, param_idx, theta_seeds, ...)` wraps it to estimate variance over the θ ensemble.

**`qiskit/circuit_builder.py`** — takes a G matrix and builds a Qiskit `QuantumCircuit` with H layers and parameterized MultiRZ gates. Only used by `run_qiskit.py` for validation against the classical formula.

**`rng.py`** — `make_rng(seed)` returns a NumPy Generator. `split_seeds(base_seed, n)` derives n independent child seeds deterministically from a single base, so each (family, kernel, init, n) combination gets reproducible but independent randomness.

## The three runners

**`run_scaling.py`** is the main experiment. It sweeps the full grid and writes one JSONL record per (family, kernel, init, n, param_idx) with mean, variance, std, and median of the gradient.

**`run_qiskit.py`** validates the classical formula against Qiskit circuit simulation on small circuits (n ≤ 20 or so). It compares exact classical, statevector, shot-based, and noisy results. Run this first when debugging — if classical and statevector disagree, something is wrong in the expectation formula before you trust any large-n results.

**`run_forge.py`** takes small G matrices (n ≤ 12) and runs structural modeling via Forge to find minimal hypergraph patterns that induce gradient concentration.

## Testing

`hypergraph/hypothesis_strategies.py` provides Hypothesis strategies for property-based tests:
- `hypergraph_matrix(n_min, n_max, m_min, m_max)` — random binary G matrices with no all-zero rows
- `iqp_parameters(n_params, scheme, std)` — θ vectors under uniform or small-angle init

Tests check mathematical invariants — that expectation values stay in [−1, 1], that the gradient formula is consistent with finite differences, that the cosine formula matches the exact formula for small n — rather than specific numerical outputs. This makes them resilient to implementation changes.
