---
title: Gradient Variance
tags:
  - theory
  - barren-plateau
  - metric
---

# Gradient Variance — The Primary Metric

This is the quantity the whole project is organized to measure.

## Definition

For parameter index $i$, distribution over initializations $\theta_{\text{dist}}$, family $F$, kernel $K$, and qubit count $n$:

$$
V(i; \theta_{\text{dist}}, F, K, n) = \mathrm{Var}_{\theta \sim \theta_{\text{dist}}}\!\left[\partial_{\theta_i} \mathrm{MMD}^2_K(p, q^F_\theta)\right]
$$

The aggregate reported per cell:

$$
V_{\text{agg}}(F, K, n) = \frac{1}{m}\sum_i V(i; \ldots) \quad \text{or} \quad \mathrm{median}_i V(i; \ldots)
$$

The median is preferred if heavy tails are detected across parameter indices.

## The Scaling Hypothesis

$$
\log V_{\text{agg}}(F, K, n) \sim -\alpha(F, K) \cdot n + \beta(F, K) \log n + c(F, K)
$$

Three regimes:

| Regime | Shape | Trainability |
|---|---|---|
| $\alpha > 0$, $\beta \approx 0$ | **Exponential decay** | Barren plateau — infeasible at scale |
| $\alpha = 0$, $\beta < 0$ | **Polynomial decay** | Mild plateau — shot count scales poly |
| $\alpha = 0$, $\beta = 0$ | **Constant** | No plateau — trainability preserved |

See [[Barren Plateaus]] for the theoretical backdrop.

## Sources of Randomness

The variance is over $\theta$, but each gradient evaluation further contains:

- $z \sim U(\{0,1\}^n)$ — the [[IQP Expectation]] Monte Carlo
- $a \sim P_k$ — the [[MMD Loss|Z-word mixture]] Monte Carlo
- (optional) a new random circuit per seed — currently the scaling runner fixes the circuit per seed, see [[How a Scaling Run Works]]

## Measurement Protocol

From [`estimate_gradient_variance`](../src/iqp_bp/mmd/gradients.py):

1. Fix $G$ for one (family, $n$) setting.
2. Draw a list of `theta_seeds` — one $\theta$ per ensemble sample.
3. For each `param_idx in range(min(5, m))`:
    1. Call `grad_mmd2_analytic(theta, G, data, param_idx, ...)` for every $\theta$.
    2. Report `mean`, `var`, `std`, `median`, `n_seeds`.
4. Write one JSONL record per `(family, kernel, init, n, param_idx)` — this gives you the $n$-dependence curve when combined across settings.

The cap at `min(5, m)` is a cost-control heuristic; see [[TODO Roadmap|D4.2/D4.3]] for planned heavy-tail and gradient-norm extensions.

## Open Planned Work

- Aggregate gradient-norm proxies
- Heavy-tail detection across $i$
- Median-of-means statistics
- Curve fitting using `scipy.optimize.curve_fit` to extract $(\alpha, \beta, c)$

## Related

- [[Gradient Derivation]]
- [[Barren Plateaus]]
- [[Scaling Runner]]
- [[Research Questions]] — Q1/Q2 directly reference this metric
- [[Initialization Schemes]]
