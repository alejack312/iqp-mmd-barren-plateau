---
title: Laplacian Kernel
tags:
  - kernel
  - mmd
  - phase2
---

# Laplacian Kernel

Phase-2 kernel with heavier tails than the Gaussian. Spectral decomposition via Krawtchouk polynomials.

## Definition

$$
k_L(x, y) = \exp\!\left(-\frac{\sqrt{H(x, y)}}{\sigma}\right)
$$

Unlike the Gaussian, the Laplacian decay in Hamming distance is **sub-exponential** ($\sqrt{H}$, not $H$), so high-distance pairs contribute more to the kernel.

## Spectral Weights

The exact Walsh/Fourier decomposition uses Krawtchouk polynomials:

$$
w_L(a; \sigma) = \frac{1}{2^n} \sum_{h=0}^{n} K_{|a|}(h; n) \cdot e^{-\sqrt{h}/\sigma}
$$

where $K_k(x; n) = \sum_j (-1)^j \binom{x}{j}\binom{n-x}{k-j}$ is the Krawtchouk polynomial.

In the code (`_laplacian_spectral_weight`):

```python
def _laplacian_spectral_weight(n, w, sigma):
    total = 0.0
    for h in range(n + 1):
        total += _krawtchouk(w, h, n) * np.exp(-np.sqrt(h) / sigma)
    return total / (2**n)
```

## Sampling Status

> [!warning] Explicit stub
> The current Laplacian sampler is an **approximate path** — see the `TODO T2` in `mmd/kernel.py`. It computes spectral weights via the Krawtchouk sum above, then samples Hamming weights proportional to $\binom{n}{w} \cdot w_L(n, w, \sigma)$.
>
> The MMD² decomposition with Laplacian kernel is not yet derived to the same level of rigor as the Gaussian. The implementation is marked as a stub and should not be trusted for final scaling claims until the derivation is locked.

See [[TODO Roadmap|T2]] and [[Kernel Module]].

## Why It's Interesting

- **Heavier spectral tails** mean the MMD pays more attention to intermediate-weight Z-words.
- This could either help (by exposing the model to more modes) or hurt (by spreading gradient signal thin).
- A clean comparison against Gaussian in the same families answers whether kernel choice alone controls BP behavior (Research Q2).

## Study Position

Phase 2 of the sweep — added after the Gaussian regime is fully characterized. See [[SMART Spec]].

## Related

- [[Gaussian Kernel]]
- [[Kernel Module]]
- [[Kernel Spectral Decomposition]]
- [[TODO Roadmap]] — T2 is the open derivation work
