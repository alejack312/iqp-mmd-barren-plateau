---
title: MMD Loss
aliases:
  - MMD²
  - Maximum Mean Discrepancy
tags:
  - theory
  - loss
  - mmd
---

# MMD² Loss

The **Maximum Mean Discrepancy** loss is the objective the project optimizes. In the IQP setting it has a remarkable property: it can be decomposed as a sum of squared differences of Pauli-Z expectations, and each term is classically computable.

## Standard Form

For samples from data distribution $p$ and model distribution $q_\theta$:

$$
\mathrm{MMD}^2(p, q_\theta) = \mathbb{E}_{x,x' \sim p}[k(x,x')] - 2\mathbb{E}_{x \sim p, y \sim q_\theta}[k(x,y)] + \mathbb{E}_{y,y' \sim q_\theta}[k(y,y')]
$$

This is the "mean embedding" distance in the reproducing kernel Hilbert space induced by $k$.

## Mixture-of-Z-Words Form (the one the code uses)

For kernels with a Walsh/Fourier decomposition over $\{0,1\}^n$, the MMD² can be rewritten as:

$$
\mathrm{MMD}^2(p, q_\theta) = \sum_{a \in \{0,1\}^n} w_k(a) \left(\langle Z_a \rangle_p - \langle Z_a \rangle_{q_\theta}\right)^2
= \mathbb{E}_{a \sim P_k}\!\left[(\langle Z_a \rangle_p - \langle Z_a \rangle_{q_\theta})^2\right]
$$

where:

- $w_k(a)$ is the [[Spectral Weight|spectral weight]] of mode $a$
- $P_k(a) \propto w_k(a)$ is the induced sampling distribution over [[Z-Word|Z-words]]
- $\langle Z_a \rangle_p$ is a parity average over the dataset (see [[Data-Side Parity]])
- $\langle Z_a \rangle_{q_\theta}$ is the [[IQP Expectation]]

This is the **locked MMD² derivation** of the repo. See [[Locked MMD² Derivation]].

## Why This Decomposition Is the Point

> [!tip] The entire project hinges on this identity
> It turns a distribution-matching loss over $\{0,1\}^n$ into a **weighted sum of classically-computable expectation differences**. No state vector, no shot sampling, no exponential memory — just Monte Carlo over Z-words $a$ and uniform bitstrings $z$.

## Monte Carlo Estimator

From [`src/iqp_bp/mmd/loss.py`](../src/iqp_bp/mmd/loss.py):

```python
def mmd2(theta, G, data, kernel, num_a_samples, num_z_samples, ...):
    a_samples = sample_a(kernel, n, num_a_samples, ...)     # shape (B, n)
    exp_p = dataset_expectations_batch(data, a_samples)     # shape (B,)
    exp_q = [iqp_expectation(theta, G, a, num_z_samples)[0]
             for a in a_samples]                            # length B
    return float(np.mean((exp_p - exp_q) ** 2))
```

Two nested Monte Carlo loops:

- **Outer:** $a \sim P_k$ (Z-word mixture) — driven by [[Kernel Spectral Decomposition|kernel spectral weights]]
- **Inner:** $z \sim U(\{0,1\}^n)$ (IQP expectation) — see [[IQP Expectation]]

Defaults: `num_a_samples=512`, `num_z_samples=1024`.

## Open Diagnostics Work

From the [[TODO Roadmap|TODO P5]]: the estimator currently returns only a scalar. Planned extensions:

- Per-observable contribution breakdown
- Confidence intervals via bootstrap
- Small-$n$ exact MMD² cross-check (T2/D2.1)

## Related

- [[Gradient Derivation]] — $\partial_{\theta_i} \mathrm{MMD}^2$
- [[Gradient Variance]] — $\mathrm{Var}_\theta[\partial \mathrm{MMD}^2]$ the main BP metric
- [[Kernels MOC]] — spectral weights for each kernel
- [[Z-Word]], [[Spectral Weight]]
- [[Locked MMD² Derivation]]
