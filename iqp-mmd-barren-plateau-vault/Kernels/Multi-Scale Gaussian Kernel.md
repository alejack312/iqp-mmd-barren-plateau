---
title: Multi-Scale Gaussian Kernel
tags:
  - kernel
  - mmd
  - phase2
---

# Multi-Scale Gaussian Kernel

A mixture-of-Gaussians kernel that aggregates several bandwidths into one loss. Phase-2 kernel.

## Definition

$$
k_{MSG}(x, y) = \sum_i w_i \exp\!\left(-\frac{H(x, y)}{2\sigma_i^2}\right)
$$

with weights $\sum_i w_i = 1$. Default is $K = 3$ components with $\sigma \in \{0.5, 1.0, 2.0\}$ and uniform weights.

## Spectral Weights (Mixture)

By linearity of the Walsh/Fourier decomposition, the spectral weight at mode $a$ is a weighted sum over components:

$$
w_{MSG}(a; \{\sigma_i\}, \{w_i\}) = \sum_i w_i \cdot \tau_i^{|a|}, \quad \tau_i = \tanh\!\left(\frac{1}{4\sigma_i^2}\right)
$$

## Sampling

The Monte Carlo sampler is a straightforward mixture:

1. Pick a component $i \sim \text{Categorical}(w_1, \ldots, w_K)$
2. Sample $a$ from `gaussian_sample_a` at the chosen $\sigma_i$

See `multi_scale_gaussian_sample_a` in [`mmd/kernel.py`](../src/iqp_bp/mmd/kernel.py).

## Why It Might Matter

- **Captures multi-scale correlations** — a single Gaussian only puts mass at one "scale," but real data often has structure at multiple scales.
- **Tradeoff hypothesis** (from [[Research Questions|Outcome C]]) — MSG might avoid BP at some bandwidth combinations but introduce bandwidth-dependent trade-offs.

## Status

Implemented, but:

- The MMD² exact decomposition is not yet validated against brute force on small $n$
- Component sweep is fixed at a default rather than being part of the experiment grid
- See `TODO D8.1` in `kernel.py`

## Config

```yaml
kernel:
  type: multi_scale_gaussian
  multi_scale_gaussian:
    sigmas: [0.5, 1.0, 2.0]
    weights: null  # null → uniform 1/K
```

## Related

- [[Gaussian Kernel]]
- [[Laplacian Kernel]]
- [[Kernel Module]]
- [[TODO Roadmap]]
