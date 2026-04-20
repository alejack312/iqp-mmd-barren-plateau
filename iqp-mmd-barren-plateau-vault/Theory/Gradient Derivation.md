---
title: Gradient Derivation
tags:
  - theory
  - gradient
  - mmd
---

# Analytic Gradient of MMD²

The per-parameter gradient of the [[MMD Loss]] under the [[IQP Expectation|classical IQP estimator]]. This is the quantity whose **variance** the project measures to diagnose [[Barren Plateaus]].

## Step 1 — Gradient of ⟨Z_a⟩

From the phase formula $\Phi(\theta, z, a) = 2 \sum_j \theta_j (a \cdot g_j \bmod 2)(-1)^{z \cdot g_j}$:

$$
\frac{\partial \Phi}{\partial \theta_i} = 2 \cdot (a \cdot g_i \bmod 2) \cdot (-1)^{z \cdot g_i}
$$

So differentiating the cosine:

$$
\boxed{\;\partial_{\theta_i} \langle Z_a \rangle_{q_\theta} = -2 \cdot (a \cdot g_i \bmod 2) \cdot \mathbb{E}_{z \sim U}\!\left[\sin(\Phi(\theta,z,a)) \cdot (-1)^{z \cdot g_i}\right]\;}
$$

> [!info] Parity gate
> The factor $(a \cdot g_i \bmod 2)$ is **0 unless $g_i$ overlaps oddly with $a$**. So $\partial_{\theta_i}\langle Z_a\rangle$ is literally zero when the Z-word $a$ is blind to generator $g_i$. This is why sparse `G`, sparse kernel support, and the structure of `G @ a` mod 2 all interact: they control how many parameters even have nonzero gradient at each Z-word mode.

## Step 2 — Gradient of MMD²

Differentiating $(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta})^2$ and averaging over $a \sim P_k$:

$$
\boxed{\;\partial_{\theta_i} \mathrm{MMD}^2(p, q_\theta) = -2 \cdot \mathbb{E}_{a \sim P_k}\!\left[\big(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta}\big) \cdot \partial_{\theta_i}\langle Z_a\rangle_{q_\theta}\right]\;}
$$

Only the second term depends on $\theta$, giving a single negative coefficient times the covariance between residual and gradient.

## In the Codebase

**File:** [`src/iqp_bp/mmd/gradients.py`](../src/iqp_bp/mmd/gradients.py)

| Function | What it computes |
|---|---|
| `grad_expectation_analytic(theta, G, a, param_idx, ...)` | $\partial_{\theta_i} \langle Z_a \rangle$ for one $(a, i)$ |
| `grad_mmd2_analytic(theta, G, data, param_idx, ...)` | $\partial_{\theta_i} \mathrm{MMD}^2$ for one parameter |
| `grad_mmd2_finite_diff(...)` | Reference finite-difference estimator (validation only) |
| `estimate_gradient_variance(G, data, param_idx, theta_seeds, ...)` | Variance over a $\theta$ ensemble — the primary BP metric |

## Early-Exit Optimization

In `grad_expectation_analytic`:

```python
a_dot_gi = int((a @ g_i) % 2)
if a_dot_gi == 0:
    return 0.0  # generator g_i doesn't contribute to Z_a
```

This is the same parity gate from the theory, promoted to an early return. In sparse regimes it skips most of the work.

## Validation Against Finite Differences

`grad_mmd2_finite_diff` re-seeds both calls to `mmd2` with the **same** RNG seed so that Monte Carlo noise is subtracted out between $\theta \pm \epsilon$ evaluations. This is essential — otherwise the finite-difference signal is buried in sampling noise.

From [[TODO Roadmap|V2]]: a proper analytic-vs-finite-difference regression test on small $n$ is still open work.

## What Variance Is Measured Over

The gradient variance integrates:

- $\theta \sim \theta_{\text{dist}}$ (init scheme) — **primary source**
- $z \sim U(\{0,1\}^n)$ (IQP MC) — secondary
- $a \sim P_k$ (Z-word MC) — secondary
- Circuit randomness (fixed hypergraph per seed, or averaged over random instances)

See [[Gradient Variance]] for the measurement protocol and statistical subtleties.

## Related

- [[MMD Loss]]
- [[IQP Expectation]]
- [[Gradient Variance]]
- [[Barren Plateaus]]
- [[Initialization Schemes]]
