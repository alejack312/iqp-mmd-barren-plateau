---
title: Gaussian Kernel
tags:
  - kernel
  - mmd
  - primary
---

# Gaussian Kernel

The **primary kernel** for Phase 1 of the study. Fully locked in convention and implementation.

## Definition

$$
k_G(x, y) = \exp\!\left(-\frac{H(x, y)}{2\sigma^2}\right)
$$

where $H(x, y)$ is Hamming distance (with binary $\{0,1\}^n$ encoding; equivalent to $\|\cdot\|^2/2$ in the $\pm 1$ encoding).

**Bandwidth** $\sigma$ is the only parameter. The study sweeps over `[0.1, 0.5, 1.0, 2.0, 5.0]` by default.

## Spectral Weights

From the [[Gaussian Convention|locked]] Walsh/Fourier decomposition:

$$
w_G(a; \sigma) \propto \tau^{|a|}, \quad \tau = \tanh\!\left(\frac{1}{4\sigma^2}\right)
$$

where $|a|$ is the Hamming weight of the Z-word mask $a$. So higher-weight modes are suppressed exponentially in $|a|$, with suppression rate controlled by $\sigma$:

- Small $\sigma$ → $\tau \to 1$ → all modes weighted similarly
- Large $\sigma$ → $\tau \to 0$ → only low-weight (low-order) modes matter

See [[Spectral Weight]] and [[Kernel Spectral Decomposition]].

## Z-Word Sampling

The Monte Carlo sampler $a \sim P_G$ proceeds in two steps (`gaussian_sample_a`):

1. Sample Hamming weight $w \sim P(w) \propto \binom{n}{w} \tau^w$
2. Pick a uniformly random subset of size $w$ from the $n$ qubits

This samples from the weight-stratified marginal of $P_G$.

## Code Path

**File:** [`src/iqp_bp/mmd/kernel.py`](../src/iqp_bp/mmd/kernel.py)

| Function | Purpose |
|---|---|
| `gaussian_kernel(x, y, sigma)` | Direct kernel evaluation |
| `gaussian_spectral_weights(n, sigma)` | Array of $\tau^w$ for $w = 0, 1, \ldots, n$ |
| `gaussian_sample_a(n, num_a_samples, sigma, rng)` | Z-word sampler |
| `_gaussian_tau(sigma)` | The locked $\tau = \tanh(1/(4\sigma^2))$ |

## The Convention Lock Story

Earlier versions of the repo had three different definitions of `sigma` floating between `gaussian_kernel`, `gaussian_spectral_weights`, and `gaussian_sample_a`. The current lock is:

```text
k(x, y) = exp(-H(x, y) / (2 sigma^2))
↓
q = exp(-1 / (2 sigma^2))  # per-bit agreement factor
↓
tau = (1 - q) / (1 + q) = tanh(1 / (4 sigma^2))
```

All three functions now share the same $\sigma$ meaning. See [[Implementation Choices#The Gaussian convention]].

> [!warning] Why the lock matters
> If the direct kernel formula and the Z-word sampler disagree about what $\sigma$ means, the Monte Carlo estimator is no longer estimating the MMD it claims to estimate. The gradient variance numbers would be silently wrong.

## Gradient Flow

With Gaussian spectral weights, the MMD² gradient reshapes into a weighted sum where most contribution comes from low-weight Z-words. This means:

- Parameters whose generator $g_i$ is **disjoint from many sampled low-weight `a`** will have small gradients (parity gate from [[Gradient Derivation]])
- In the product-state family, gradients are well-behaved because every generator is weight-1
- In the complete-graph family, gradients dilute across many parameters

## Related

- [[Laplacian Kernel]]
- [[Multi-Scale Gaussian Kernel]]
- [[Gaussian Convention]]
- [[Kernel Spectral Decomposition]]
- [[Kernel Module]]
- [[MMD Loss]]
- [[References#Scaling GQML]] — source of the convention
