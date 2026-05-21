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
w_L(a; \sigma) = \frac{1}{2^n} \sum_{h=0}^{n} K_h(|a|; n) \cdot e^{-\sqrt{h}/\sigma}
$$

where $K_h(k; n) = \sum_j (-1)^j \binom{k}{j}\binom{n-k}{h-j}$ is the Krawtchouk polynomial that sums the character of a fixed weight-$k$ observable over all bitstrings at Hamming distance $h$.

In the code (`_laplacian_spectral_weight`):

```python
def _laplacian_spectral_weight(n, w, sigma):
    total = 0.0
    for h in range(n + 1):
        total += _krawtchouk(h, w, n) * np.exp(-np.sqrt(h) / sigma)
    return total / (2**n)
```

## Sampling Status

> [!success] Locked finite-cube path
> The Laplacian sampler now uses the same Krawtchouk/Walsh coefficients as the exact MMD path. It samples Hamming weights proportional to $\binom{n}{w} \cdot w_L(n, w, \sigma)$, then samples a uniform mask of that weight.
>
> The implementation no longer clips negative coefficients or takes absolute values. Coefficients must be finite and non-negative up to numerical tolerance; otherwise the kernel path raises `ValueError` instead of producing an approximate MMD under the Laplacian name.

See [[TODO Roadmap|T2]] and [[Kernel Module]].

## Validation Contract

The code validates the Laplacian path in three ways:

- direct kernel reconstruction on small Boolean cubes:
  $k_L(x,y) = \sum_a w_L(a;\sigma)\chi_a(x)\chi_a(y)$
- exact small-$n$ MMD agreement between direct pairwise kernel MMD and the spectral observable sum
- sampler agreement with the normalized per-order spectral mass

This locks the implementation convention for finite-$n$ experiments. It does not automatically promote Laplacian results into the primary Gaussian-only claims.

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
- [[TODO Roadmap]] — T2 is complete
