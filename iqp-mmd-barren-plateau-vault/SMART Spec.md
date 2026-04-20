---
title: SMART Spec
tags:
  - planning
  - spec
---

# SMART Execution Specification

Condensed from [`docs/SMART-spec.md`](../docs/SMART-spec.md).

## Time Window

- **Start:** Mar 18, 2026
- **End:** May 15, 2026
- **Duration:** ~8 weeks

## SMART Breakdown

### Specific

Over ~8 weeks:

1. Lock the precise problem definition (circuit, kernel, parameter distribution, gradient target) — see [[Scope Lock]]
2. Implement a reproducible computational pipeline:
    - Hypothesis-driven circuit generator → [[Hypergraph Families]]
    - Classical IQP expectation engine → [[IQP Expectation]]
    - Gradient/variance estimator → [[Gradients Module]]
    - Full classical training loop for MMD
3. Execute scaling experiments across **4 circuit families × 3 kernel types × ≥3 initialization schemes**
4. Validate selected regimes in Qiskit (statevector / shots / noise)
5. Use Forge to search for structural plateau-inducing patterns
6. Write a thesis-quality report

### Measurable

- **1** complete derivation note for $\partial_\theta \mathrm{MMD}^2$
- **1** reproducible repo with config-driven runs
- **Experiments:** 4 × 3 × ≥3 × ≥6 × ≥100 seeds
    - Phase 1: Gaussian × 4 families (complete)
    - Phase 2: Laplacian + multi-scale Gaussian
- **≥12** publication-quality figures
- **≥3** Qiskit comparisons (exact vs SV vs shots; with/without noise)
- **≥2** Forge structural findings
- **Final report** target 40–70 pages + 10–15 slide deck

### Achievable

Core expectation values for parameterized IQP circuits in MMD² training are classically estimable efficiently using the den Nest-style formulation (see [[IQP Expectation]]). Qiskit validation is limited to small/moderate $n$.

### Relevant

Directly answers whether IQP+MMD generative learning avoids barren plateaus, and how/when shot noise or hardware noise reinstates trainability barriers.

### Time-Bound

Weekly/biweekly deliverables. Final submission package by **May 15, 2026**.

## Study Order

| Phase | Scope |
|---|---|
| Phase 1 | Gaussian kernel exhaustively across all four families |
| Phase 2 | Laplacian + multi-scale Gaussian added after Gaussian lock |

## Operational Definitions

### IQP Model

Gates $\exp(i\theta_j X^{g_j})$ with generator bitmask $g_j \in \{0,1\}^n$, measurement in computational basis. Umbrella: use IQP circuits as generative models, train on classical with MMD loss, sample on quantum at inference.

### Loss

Squared MMD in Z-word mixture form:

$$
\mathrm{MMD}^2(p, q_\theta) = \mathbb{E}_{a \sim P_k}\!\left[(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta})^2\right]
$$

See [[MMD Loss]] and [[Locked MMD² Derivation]].

### Gradient Target

- Per-parameter: $\partial_{\theta_i}\mathrm{MMD}^2$
- Aggregate: $\mathbb{E}_i[\mathrm{Var}(\partial_{\theta_i}\mathcal{L})]$ or median across $i$

### Parameter and Circuit Distributions

See [[Initialization Schemes]] (three schemes are primary axes) and [[Families MOC]] (four families).

### Datasets

D1 Product Bernoulli (primary), D2 Ising, D3 Binary mixture. See [[Datasets]].

### Anti-Concentration Target

In addition to gradient scaling, evaluate whether the trained distribution is anti-concentrated:

$$
\Pr_{x \sim U(\{0,1\}^n)}\!\left[p_\theta(x) \ge \alpha 2^{-n}\right] \ge \beta
$$

for constants $\alpha, \beta > 0$ independent of $n$. See [[Anti-Concentration]].

## Related

- [[Proposal]]
- [[Scope Lock]]
- [[TODO Roadmap]]
- [[Research Questions]]
