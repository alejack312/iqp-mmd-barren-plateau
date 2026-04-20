---
title: Data Factory
tags:
  - code
  - iqp_bp
  - datasets
---

# `iqp_bp.experiments.data_factory`

Runtime dataset generation for the scaling and validation runners. Materializes binary datasets on the fly from config, without baking specific dataset files into the repo.

**File:** [`src/iqp_bp/experiments/data_factory.py`](../src/iqp_bp/experiments/data_factory.py)

## Public API

```python
def make_dataset(dataset_cfg, *, n, seed) -> tuple[np.ndarray, dict]
```

Returns `(data, metadata)` where `data` has shape `(n_samples, n)` with values in $\{0, 1\}$ and `metadata` is a JSON-serializable provenance dict.

## Supported Dataset Types

Selected via `dataset.type` in config:

### `product_bernoulli` — Baseline

```python
data = rng.integers(0, 2, size=(n_samples, n), dtype=np.uint8)
```

Each bit is independent Bernoulli(0.5). See [[Product Bernoulli Dataset]].

### `ising` — Structured pairwise correlations

Sparse Ising model with sequential Gibbs sampling:

- `topology`: `grid_2d` (requires perfect-square `n`) or `erdos_renyi` (bounded degree)
- `beta`: inverse temperature
- `coupling_std`: Gaussian coupling scale, normalized by $1/\sqrt{n}$
- `burn_in_sweeps`: default `max(20n, 100)`
- `thinning`: default 2
- `num_chains`: default 4

See [[Ising Dataset]].

### `binary_mixture` — Multi-modal binarized mixture

Sample Gaussian centers, perturb, then threshold:

```python
centers = rng.normal(0, 1, size=(n_modes, n))
assignments = rng.integers(0, n_modes, size=n_samples)
latent = centers[assignments] + rng.normal(0, noise, size=(n_samples, n))
data = (latent > threshold).astype(np.uint8)
```

See [[Binary Mixture Dataset]].

## Internal Helpers

- `_make_binary_mixture_dataset(n, n_samples, cfg, rng)`
- `_make_ising_dataset(n, n_samples, cfg, rng)`
- `_make_ising_topology(n, topology, coupling_std, rng)`
- `_sample_ising_binary(adjacency, beta, n_samples, burn_in_sweeps, thinning, num_chains, rng)`

## Metadata Shape

All dataset types return `metadata` containing at minimum:

```json
{
  "type": "ising",
  "n_samples": 10000,
  "seed": 12345,
  "beta": 1.0,
  "coupling_std": 1.0,
  "topology": "grid_2d",
  "grid_side": 4,
  "num_edges": 24
}
```

This is persisted on every JSONL row in the scaling output so dataset provenance travels with each result.

## Related

- [[Datasets]]
- [[Product Bernoulli Dataset]]
- [[Ising Dataset]]
- [[Binary Mixture Dataset]]
- [[Scaling Runner]]
