---
title: Kernel Module
tags:
  - code
  - iqp_bp
  - mmd
---

# `iqp_bp.mmd.kernel`

The module that exposes kernel evaluators, spectral weights, and Z-word samplers for each supported kernel.

**File:** [`src/iqp_bp/mmd/kernel.py`](../src/iqp_bp/mmd/kernel.py)

## Public API

### Gaussian (Primary)

- `gaussian_kernel(x, y, sigma)` — direct evaluation
- `gaussian_spectral_weights(n, sigma)` — $\tau^w$ for $w = 0, \ldots, n$
- `gaussian_sample_a(n, num_a_samples, sigma, rng)` — Z-word sampler
- `_gaussian_tau(sigma)` — the locked $\tau = \tanh(1/(4\sigma^2))$

### Laplacian (Phase 2, finite-cube locked)

- `laplacian_kernel(x, y, sigma)`
- `laplacian_sample_a(n, num_a_samples, sigma, rng)`
- `laplacian_spectral_weights(n, sigma)` — validated per-Hamming-weight Walsh coefficients
- `_laplacian_spectral_weight(n, w, sigma)`

### Multi-Scale Gaussian (Phase 2)

- `multi_scale_gaussian_kernel(x, y, sigmas, weights)`
- `multi_scale_gaussian_sample_a(n, num_a_samples, sigmas, weights, rng)`

### Polynomial (Legacy)

- `polynomial_kernel(x, y, degree, constant)`
- `polynomial_sample_a(n, num_a_samples, degree, constant, rng)`

### Linear (Legacy)

- `linear_kernel(x, y)`
- `linear_sample_a(n, num_a_samples, rng)`

## Dispatcher

```python
KERNEL_SAMPLERS = {
    "gaussian": gaussian_sample_a,
    "laplacian": laplacian_sample_a,
    "multi_scale_gaussian": multi_scale_gaussian_sample_a,
    "polynomial": polynomial_sample_a,  # legacy
    "linear": linear_sample_a,           # legacy
}

def sample_a(kernel, n, num_a_samples, rng, **kernel_params):
    return KERNEL_SAMPLERS[kernel](n=n, num_a_samples=num_a_samples, rng=rng, **kernel_params)
```

Used by both [[MMD Loss Module|`mmd.loss`]] and [[Gradients Module|`mmd.gradients`]].

## Helper Functions

- `_log_binom(n, k)` — stable log binomial via `lgamma`
- `_krawtchouk(k, x, n)` — Krawtchouk polynomial used by the Laplacian path
- `_poly_coeff(w, degree, constant)` — polynomial kernel coefficient via `comb`

## Convention Discipline

> [!important] Change all three together
> If you change any Gaussian formula — kernel, spectral weights, or sampler — update all three in the same commit. These were previously inconsistent and caused silent MMD² estimator bugs. See [[Implementation Choices#The Gaussian convention]] and [[Gaussian Convention]].

## Open TODOs in This Module

- **D8.1** — validate multi-scale Gaussian against exact mixture and add component sweep to the experiment grid

## Related

- [[Kernels MOC]]
- [[Gaussian Kernel]]
- [[Laplacian Kernel]]
- [[Multi-Scale Gaussian Kernel]]
- [[MMD Loss Module]]
- [[Gradients Module]]
