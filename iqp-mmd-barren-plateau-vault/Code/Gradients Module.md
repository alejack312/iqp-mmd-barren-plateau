---
title: Gradients Module
tags:
  - code
  - iqp_bp
  - mmd
  - gradient
---

# `iqp_bp.mmd.gradients`

Analytic gradient estimators and the variance summary that drives the scaling runner.

**File:** [`src/iqp_bp/mmd/gradients.py`](../src/iqp_bp/mmd/gradients.py)

## Function Map

| Function | Returns | Role |
|---|---|---|
| `grad_expectation_analytic(theta, G, a, param_idx, ...)` | scalar | $\partial_{\theta_i} \langle Z_a\rangle_{q_\theta}$ |
| `grad_mmd2_analytic(theta, G, data, param_idx, kernel, ...)` | scalar | $\partial_{\theta_i} \mathrm{MMD}^2$ |
| `grad_mmd2_finite_diff(theta, G, data, param_idx, eps, ...)` | scalar | Finite-difference validation |
| `estimate_gradient_variance(G, data, param_idx, theta_seeds, ...)` | dict | Variance over a $\theta$ ensemble — the BP metric |

## `grad_expectation_analytic` — The Inner Kernel

```python
a_dot_gi = int((a @ g_i) % 2)
if a_dot_gi == 0:
    return 0.0  # generator g_i doesn't contribute to Z_a

z = rng.integers(0, 2, size=(num_z_samples, n), dtype=np.uint8)
phases = iqp_phase(theta, G, z, a)
z_dot_gi = (z @ g_i) % 2
sign_i = 1 - 2 * z_dot_gi.astype(float)
return float(-2.0 * a_dot_gi * np.mean(np.sin(phases) * sign_i))
```

This implements the derivative formula from [[Gradient Derivation]]:

$$
\partial_{\theta_i}\langle Z_a\rangle = -2 (a \cdot g_i \bmod 2) \cdot \mathbb{E}_z[\sin(\Phi) \cdot (-1)^{z \cdot g_i}]
$$

with the parity-gate early exit.

## `grad_mmd2_analytic` — Wraps The Inner Kernel

```python
a_samples = sample_a(kernel=kernel, n=n, num_a_samples=num_a_samples, rng=rng, **kernel_params)
exp_p = dataset_expectations_batch(data, a_samples)
contributions = []
for a, ep in zip(a_samples, exp_p):
    eq, _ = iqp_expectation(theta, G, a, num_z_samples, rng)
    dq = grad_expectation_analytic(theta, G, a, param_idx, num_z_samples, rng)
    contributions.append((ep - eq) * dq)
return float(-2.0 * np.mean(contributions))
```

Average over Z-word samples of `(residual) × (∂⟨Z_a⟩/∂θ_i)`, then multiply by $-2$.

## `grad_mmd2_finite_diff` — Validation Only

Re-seeds **both** `mmd2` calls with the **same** RNG so Monte Carlo noise is correlated out between $\theta \pm \epsilon$:

```python
seed = int(rng.integers(0, 2**31))
f_plus = mmd2(theta_plus, G, data, ..., rng=np.random.default_rng(seed), ...)
f_minus = mmd2(theta_minus, G, data, ..., rng=np.random.default_rng(seed), ...)
return (f_plus - f_minus) / (2 * eps)
```

This correlation trick is essential: without it, the difference is buried in sampling noise.

## `estimate_gradient_variance` — The Primary BP Metric

```python
grads = [
    grad_mmd2_analytic(theta=th, G=G, data=data, param_idx=param_idx, ...)
    for th in theta_seeds
]
grads = np.array(grads)
return {
    "mean": float(grads.mean()),
    "var": float(grads.var()),
    "std": float(grads.std()),
    "median": float(np.median(grads)),
    "n_seeds": len(theta_seeds),
}
```

Called from [[Scaling Runner]] once per `(setting, param_idx)`.

## Open Work

- **D2.1** — JAX autodiff estimator to cross-check analytic path on small $n$
- **D4.2 / D4.3** — heavy-tail detection, median-of-means, gradient-norm proxies

## Related

- [[Gradient Derivation]]
- [[Gradient Variance]]
- [[IQP Expectation]]
- [[Kernel Module]]
- [[Scaling Runner]]
