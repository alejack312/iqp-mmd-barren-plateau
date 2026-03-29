# Configuration guide

This covers the four things you'll most often need to change: what data the model trains on, what circuit topology it uses, which kernel defines the loss, and where parameters start. Everything is controlled through YAML files — no code changes required for standard experiments.

If a term here is project-specific â€” for example `ZZ lattice family`, `sparse Erdős–Rényi`, or `locked MMD^2 derivation` â€” see [glossary.md](./glossary.md).

## How configs work

`configs/base.yaml` holds defaults for all experiments. Files in `configs/experiments/` override specific values. When you run an experiment, `load_config()` deep-merges your file over the base:

```bash
python -m iqp_bp.cli run-scaling configs/experiments/scaling_v1.yaml
```

To see the fully merged config without running anything:

```bash
python -m iqp_bp.cli run-scaling configs/experiments/scaling_v1.yaml --dry-run
```

Any key not set in your experiment file falls back to the base config. This means you only need to specify what you're changing.

---

## Setting up the problem and data

The training target is controlled by the `dataset` block. The default is `product_bernoulli`:

```yaml
dataset:
  type: product_bernoulli
  n_samples: 10000
```

Under this distribution each qubit is an independent fair coin, so every Z-word expectation ⟨Z_a⟩_p = 0 for |a| ≥ 1. The loss gradient then measures purely how well the model reproduces uncorrelated marginals — the cleanest possible baseline for the barren plateau study.

Two alternatives are available for structured targets:

```yaml
# Ising-like correlations
dataset:
  type: ising
  ising:
    beta: 1.0           # inverse temperature
    coupling_std: 0.5   # disorder strength
    topology: grid_2d   # or: erdos_renyi

# Binarized Gaussian mixture
dataset:
  type: binary_mixture
  binary_mixture:
    n_modes: 4
    noise: 0.1
```

Data is generated on-the-fly at runtime from `experiment.seed`. You don't supply files.

Qubit count sweeps are set as a list in `circuit.n_qubits`:

```yaml
circuit:
  n_qubits: [16, 24, 32, 48, 64, 96]
```

The runner loops over each value independently. Each (n, family, kernel, init) combination gets its own dataset draw of size n_samples.

---

## Changing circuit connectivity

The `circuit.family` field picks the hypergraph generator that builds the matrix G. Each generator returns a binary (m, n) matrix where row j encodes which qubits participate in the gate exp(i θ_j X^{g_j}).

Four families are in the primary sweep:

```yaml
circuit:
  family: product_state   # or: lattice | erdos_renyi | complete_graph
```

**product_state** is the no-entanglement baseline. G is the n×n identity: single-qubit Z rotations only. If this family shows gradient concentration it would be a red flag — it shouldn't.

**lattice** puts generators on a 1D or 2D grid:

```yaml
circuit:
  family: lattice
  lattice:
    dimension: 2    # 1 or 2
    range: 1        # range=1 is nearest-neighbor; range=2 adds next-nearest
```

**erdos_renyi** draws random connectivity: each qubit is included in each generator independently with probability `p_edge`. Higher values mean denser interactions.

```yaml
circuit:
  family: erdos_renyi
  erdos_renyi:
    p_edge: [0.1, 0.2]   # list → sweeps both values
```

**complete_graph** gives all-to-all ZZ interactions. No extra parameters. It produces the densest possible connectivity and is where we expect gradient concentration to be most severe.

To sweep multiple families in one run, use a list:

```yaml
circuit:
  family: [product_state, lattice, erdos_renyi, complete_graph]
```

The number of generators m comes from `n_generators`:

```yaml
circuit:
  n_generators: "n"          # m = n
  n_generators: "2*n*log(n)" # formula, evaluated at runtime
  n_generators: 64           # fixed integer
```

For `product_state` and `complete_graph`, m is determined by the family definition and this field is ignored.

### Adding a custom topology

Define a function in `src/iqp_bp/hypergraph/families.py` that takes `(n, m, rng, **kwargs)` and returns a uint8 array of shape (m, n):

```python
def my_topology(n: int, m: int, rng=None, **kwargs) -> np.ndarray:
    ...

FAMILIES["my_topology"] = my_topology
```

Then set `family: my_topology` in the config. The runner picks it up automatically via the `FAMILIES` dispatcher.

---

## Changing the kernel

The kernel determines the spectral decomposition of the MMD loss — specifically, which Z-word observables the estimator samples and how they are weighted.

```yaml
kernel:
  type: gaussian       # or: laplacian | multi_scale_gaussian
  bandwidth: [0.1, 0.5, 1.0, 2.0, 5.0]
```

**gaussian** uses `k(x, y) = exp(-H(x, y) / σ²)` where H is Hamming distance. Small σ concentrates weight on low-Hamming-weight observables (local structure). Large σ spreads weight toward higher-weight terms (global correlations). This is the primary kernel in the scaling sweep.

**laplacian** uses `k(x, y) = exp(-√H(x, y) / σ)`. It has heavier tails than the Gaussian — more sensitive to distant bitstrings. The same bandwidth list applies.

**multi_scale_gaussian** is a mixture of Gaussians. You specify component bandwidths and optional weights:

```yaml
kernel:
  type: multi_scale_gaussian
  multi_scale_gaussian:
    sigmas: [0.5, 1.0, 2.0]
    weights: null            # null → uniform (1/K per component)
    # weights: [0.5, 0.3, 0.2]   # or explicit
```

To sweep multiple kernels, use a list:

```yaml
kernel:
  type: [gaussian, laplacian]
  bandwidth: [0.5, 1.0, 2.0]
```

The `bandwidth` list applies to both Gaussian and Laplacian. Multi-scale Gaussian uses its own `sigmas` list.

Changing the kernel affects nothing about the circuit or the dataset — only what the loss function measures and which Z-words the gradient estimator samples.

---

## Controlling initialization

`init.scheme` sets how each parameter vector θ is drawn for each trial in the variance ensemble.

**uniform** draws θ_j ~ Uniform[low, high]:

```yaml
init:
  scheme: uniform
  uniform:
    low: -3.14159265   # -π
    high: 3.14159265   # +π
```

This is the stress-test regime. Parameters are spread across the full range, which is where barren plateaus are most likely to appear if they exist.

**small_angle** draws θ_j ~ Normal(0, σ_θ²):

```yaml
init:
  scheme: small_angle
  small_angle:
    std: [0.01, 0.1, 0.3]   # sweep over σ_θ values
```

The idea is that near-zero parameters keep the circuit close to the identity, where gradients may be better behaved. σ_θ = 0.01 is nearly linear; σ_θ = 0.3 starts showing nonlinear effects. This is the main hypothesis under test.

**data_dependent** initializes from the training data's covariance structure:

```yaml
init:
  scheme: data_dependent
  data_dependent:
    dataset: product_bernoulli
```

This scheme is not yet implemented — the runner raises `NotImplementedError`. It is planned for a later phase.

To sweep multiple schemes:

```yaml
init:
  scheme: [uniform, small_angle]
```

The number of independent θ draws per (family, kernel, n) combination is `estimation.num_seeds`. Seeds are derived from `experiment.seed` via `split_seeds()`, so results are reproducible.

---

## Estimation parameters

These affect estimator accuracy and runtime:

```yaml
estimation:
  num_a_samples: 512    # Z-word samples per gradient estimate (B)
  num_z_samples: 1024   # bitstring samples per ⟨Z_a⟩ evaluation
  num_seeds: 50         # θ vectors in the variance ensemble
  use_jax: true         # JAX autodiff, faster for large batches
  batch_size: 10        # θ vectors per JAX vmap call
```

`num_a_samples` and `num_z_samples` trade off variance against runtime — both are linear. `num_seeds = 50` gives a rough picture of variance scaling; 200+ is more appropriate for publication results.

Global reproducibility is controlled by:

```yaml
experiment:
  seed: 42
```

All circuit generation, dataset generation, and parameter seeds are derived from this single value.
