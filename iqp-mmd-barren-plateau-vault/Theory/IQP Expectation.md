---
title: IQP Expectation
tags:
  - theory
  - iqp
  - estimator
---

# IQP Expectation ⟨Z_a⟩

The classical estimator for Pauli-Z observables on IQP circuit outputs. This is the computational core of the entire project — every MMD² estimate is built out of these.

## The Formula

For an observable $Z_a = \bigotimes_i Z_i^{a_i}$ where $a \in \{0,1\}^n$ is a [[Z-Word|Z-word]] mask:

$$
\langle Z_a \rangle_{q_\theta} = \mathbb{E}_{z \sim U(\{0,1\}^n)}\!\left[\cos\!\big(\Phi(\theta, z, a)\big)\right]
$$

with the phase

$$
\Phi(\theta, z, a) = 2 \sum_{j=1}^{m} \theta_j \cdot (a \cdot g_j \bmod 2) \cdot (-1)^{z \cdot g_j}
$$

This is efficient: only $O(m \cdot n \cdot B)$ work per observable for $B$ Monte Carlo samples. No $2^n$ state vector is ever materialized.

## Why This Works — Sketch

For the IQP state $|\psi(\theta)\rangle = H^{\otimes n} D_\theta H^{\otimes n}|0\rangle^n$ with diagonal block $D_\theta = \exp(i \sum_j \theta_j X^{g_j})$:

1. Pauli Z observables in the computational basis pick out parity signs $(-1)^{a \cdot x}$.
2. The Hadamard sandwich converts this into a double sum over bitstrings $z, z'$ of a phase that depends on $\Phi$.
3. Cancellation from uniform $z$ sampling turns $\langle Z_a \rangle$ into $\mathbb{E}_z[\cos \Phi]$.

The **den Nest-style** classical estimator exploits exactly this: cosines are bounded, so Monte Carlo over $z$ gives a bounded-variance estimator that **does not scale exponentially in $n$**.

> [!info] Why the factor of 2
> The factor of 2 inside $\Phi$ comes from the $e^{i\theta X}$ form of each gate contributing $2\theta$ to the phase when hit twice by the Hadamard sandwich.

## Key Properties

- $|\langle Z_a \rangle_{q_\theta}| \le 1$ always, since it's a cosine expectation.
- If $a \cdot g_j \bmod 2 = 0$ for all $j$, then $\Phi \equiv 0$ and $\langle Z_a \rangle = 1$.
- Changing $\theta_j$ only affects the expectation if $a \cdot g_j \bmod 2 = 1$ — this is the **parity gate** at the heart of [[Gradient Derivation|gradient sparsity]].

## In the Codebase

**File:** [`src/iqp_bp/iqp/expectation.py`](../src/iqp_bp/iqp/expectation.py)

Two paths:

| Function | Purpose | Cost |
|---|---|---|
| `iqp_phase(theta, G, z, a)` | Compute $\Phi$ for a batch of $z$ samples | $O(B m n)$ |
| `iqp_expectation(theta, G, a, num_z_samples, rng)` | Monte Carlo estimate + standard error | $O(B m n)$ per call |
| `iqp_expectation_exact(theta, G, a)` | Exact brute-force over all $2^n$ strings | $O(2^n m n)$, only $n \le 20$ |

The exact path is used for validation tests against the Monte Carlo estimator. See [[Tests]].

## Batch Structure

```python
a_dot_g = (G @ a) % 2          # shape (m,)
z_dot_G = z @ G.T              # shape (B, m)
sign = 1 - 2 * (z_dot_G % 2)   # (-1)^{z·g_j}, shape (B, m)
weighted = theta * a_dot_g     # (m,)
phases = 2.0 * (sign @ weighted)  # (B,)
```

Every operation is either a small matmul or a cheap mod-2 reduction. The heavy work is Bernoulli sampling of $z$ and the final cosine reduction.

## Related

- [[Generator Matrix]] — defines $G$
- [[MMD Loss]] — consumes $\langle Z_a \rangle$ inside the mixture
- [[Gradient Derivation]] — derives $\partial_{\theta_i} \langle Z_a \rangle$
- [[Walsh-Hadamard Transform]] — the exact small-$n$ alternative via probability-vector extraction
- [[IQP Model]]

## Implementation Notes

- The module has an open `TODO` to add **stable batching/streaming** so very large $n$ does not blow memory (P3 in the [[TODO Roadmap]]).
- A **JAX autodiff estimator** is planned to cross-check the analytic gradient.
- The exact $n \le 20$ path is the on-ramp for the [[Anti-Concentration]] checker.
