---
title: Binary Mixture Dataset
tags:
  - dataset
  - mixture
---

# D3 — Binarized Gaussian Mixture

Multi-modal structured dataset: sample Gaussian centers in $\mathbb{R}^n$, perturb with noise, then threshold to $\{0,1\}^n$.

## Target Distribution

Effectively:

$$
p(x) = \frac{1}{K}\sum_{k=1}^K q_k(x)
$$

where each $q_k$ is the result of thresholding $\mathcal{N}(\mu_k, \epsilon^2 I)$ component-wise at a fixed threshold (default 0).

## Generation

```python
centers = rng.normal(0, 1, size=(n_modes, n))
assignments = rng.integers(0, n_modes, size=n_samples)
latent = centers[assignments] + rng.normal(0, noise, size=(n_samples, n))
data = (latent > threshold).astype(np.uint8)
```

## Config

```yaml
dataset:
  type: binary_mixture
  n_samples: 10000
  binary_mixture:
    n_modes: 4
    noise: 0.1
    threshold: 0.0
```

## Why Mixture

- **Multi-modal** — exercises the tails of the MMD² mixture distribution
- **Harder for shallow IQP** — the model has to represent $K$ distinct clusters, not a smooth unimodal distribution
- **Closer to real data structure** — real datasets usually have mode structure, not independent bits

From [[Scope Lock#5. Dataset Plan]]:

> "Purpose: multi-modal target; closer to real data; hardest for MMD² to capture with shallow IQP."

## Metadata

```json
{
  "type": "binary_mixture",
  "n_samples": 10000,
  "seed": 12345,
  "n_modes": 4,
  "noise": 0.1,
  "threshold": 0.0
}
```

## Related

- [[Datasets]]
- [[Data Factory]]
- [[Product Bernoulli Dataset]]
- [[Ising Dataset]]
