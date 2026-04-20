---
title: Walsh-Hadamard Transform
aliases:
  - FWHT
  - Hadamard transform
tags:
  - theory
  - math
  - transform
---

# Walsh-Hadamard Transform

The unnormalized Walsh-Hadamard transform (FWHT) is the bridge that converts an IQP circuit's diagonal phase vector into computational-basis amplitudes. It is the math behind `IQPModel.probability_vector_exact`.

## Definition

For a length-$2^n$ vector $v$ indexed by $z \in \{0,1\}^n$:

$$
w[x] = \sum_{z \in \{0,1\}^n} (-1)^{x \cdot z} \, v[z]
$$

with $x \in \{0,1\}^n$ and $x \cdot z = \sum_i x_i z_i \bmod 2$.

The inverse is the same transform divided by $2^n$.

## Butterfly Implementation

The FWHT can be computed in $O(n 2^n)$ time (vs naive $O(2^{2n})$) via a butterfly loop:

```python
h = 1
while h < len(values):
    for start in range(0, len(values), 2 * h):
        top = values[start : start + h]
        bottom = values[start + h : start + 2*h]
        values[start : start + h] = top + bottom
        values[start + h : start + 2*h] = top - bottom
    h *= 2
```

This is the code in `IQPModel._fwht_inplace` in [`src/iqp_bp/iqp/model.py`](../src/iqp_bp/iqp/model.py).

## Why IQP Needs It

An IQP circuit acts as $H^{\otimes n} D_\theta H^{\otimes n}$ where the middle block is diagonal in the computational basis:

$$
D_\theta |z\rangle = e^{-i\phi(z)} |z\rangle, \quad \phi(z) = \sum_j \theta_j (-1)^{z \cdot g_j}
$$

Applying the first Hadamard layer to $|0^n\rangle$ gives a uniform superposition. After $D_\theta$ we have the diagonal phase vector $d(z) = e^{-i\phi(z)}$ (up to $2^{-n/2}$ normalization). The final Hadamard layer is exactly an FWHT (divided by $2^{n/2}$):

$$
\text{amplitude}(x) = 2^{-n}\sum_z (-1)^{x \cdot z} d(z)
$$

And the Born probabilities $p(x) = |\text{amplitude}(x)|^2$ give the **exact IQP output distribution**.

## In the Codebase

**File:** [`src/iqp_bp/iqp/model.py`](../src/iqp_bp/iqp/model.py)

- `IQPModel._fwht_inplace(values)` — the butterfly itself
- `IQPModel.probability_vector_exact(max_qubits=20)` — enumerate $z$, build $d(z)$, run FWHT, square, normalize
- `IQPModel.output_probabilities_exact` — alias for the above

## Cost and Cap

Exponential in $n$: enumerating $z$ is $O(2^n)$ and the transform is $O(n 2^n)$. The default `max_qubits=20` cap is **not** a correctness bound — it is a runtime/memory guard. This is the reason the [[Anti-Concentration]] check is a *small-n* diagnostic.

## Related

- [[Anti-Concentration]]
- [[IQP Circuits]]
- [[IQP Expectation]] — uses the same phase $\phi(z)$ (just without the final transform)
- [[IQP Model]]
