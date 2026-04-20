---
title: Linear Kernel
tags:
  - kernel
  - legacy
---

# Linear Kernel

Degenerate baseline kernel. Legacy — not in the primary sweep.

## Definition

$$
k_{\text{lin}}(x, y) = \frac{x \cdot y}{n}
$$

in the $\pm 1$ encoding.

## Spectral Weights

$$
w_{\text{lin}}(a) = \begin{cases} 1/n & \text{if } |a| = 1 \\ 0 & \text{otherwise} \end{cases}
$$

So the MMD² reduces to

$$
\mathrm{MMD}^2_{\text{lin}}(p, q_\theta) = \frac{1}{n}\sum_{i=1}^{n} (\langle Z_i \rangle_p - \langle Z_i \rangle_{q_\theta})^2
$$

a sum of single-qubit marginal mismatches.

## Why Useful as a Baseline

- **Simplest possible kernel.** Any BP behavior here is purely driven by single-qubit observables.
- **Gradient = single-qubit expectation difference.** Very small variance, easy to measure.
- **Control for kernel complexity.** Comparing linear against Gaussian in the same family isolates the effect of multi-scale spectral support.

## Sampling

`linear_sample_a` draws one qubit index uniformly and places it in the mask:

```python
def linear_sample_a(n, num_a_samples, rng):
    result = np.zeros((num_a_samples, n), dtype=np.uint8)
    qubits = rng.integers(0, n, size=num_a_samples)
    result[np.arange(num_a_samples), qubits] = 1
    return result
```

## Related

- [[Kernel Module]]
- [[Polynomial Kernel]]
