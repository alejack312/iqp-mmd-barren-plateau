---
title: Polynomial Kernel
tags:
  - kernel
  - legacy
---

# Polynomial Kernel

Legacy kernel, not part of the primary sweep. Included in code for reference.

## Definition

$$
k_P(x, y) = \left(\frac{x \cdot y}{n} + c\right)^d
$$

where $x \cdot y$ is the $\pm 1$ inner product and $d$ is the degree.

## Spectral Support

The polynomial kernel has a remarkable **sparse spectrum**: only Z-words of weight $|a| \le d$ have nonzero weight. So:

- $d = 1$ → only single-qubit observables (reduces to linear)
- $d = 2$ → single-qubit + pairwise observables
- $d = 3$ → up to triples
- $d$ → captures interaction structure up to order $d$

This creates a **threshold effect** that could interact with circuit families in interesting ways.

## In the Code

```python
def polynomial_kernel(x, y, degree, constant=1.0):
    inner = float(np.dot(x, y)) / len(x)
    return float((inner + constant) ** degree)
```

The sampler `polynomial_sample_a` enumerates coefficients of $(z + c)^d$ via binomial expansion and samples a weight from the resulting distribution, then uniformly picks qubits within that weight.

## Why Legacy

From [[Scope Lock]]: the primary sweep was narrowed to Gaussian, Laplacian, multi-scale Gaussian after the SMART spec. Polynomial stays in the repo as an auxiliary kernel.

## Related

- [[Kernel Module]]
- [[Linear Kernel]] — the $d = 1$ degenerate case
