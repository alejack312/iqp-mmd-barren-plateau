---
title: Gradient Concentration
tags:
  - theory
  - barren-plateau
---

# Gradient Concentration

**Gradient concentration** means gradients from different random parameter draws become tightly clustered near zero, so the variance shrinks and optimization loses useful directional signal.

In practice:

- Most random initializations give almost the same tiny gradient
- Changing parameters slightly does not produce a reliably distinguishable update direction
- The optimizer becomes dominated by estimator noise, shot noise, or finite-precision effects

## Observable Symptom of Barren Plateaus

Concentration is the observable symptom of a [[Barren Plateaus|barren plateau]]. If concentration strengthens rapidly with system size — and crosses the scale of Monte Carlo or shot noise — training becomes impossible to scale.

## How The Project Measures It

The primary metric is [[Gradient Variance]]:

$$
V(i; \theta_{\text{dist}}, F, K, n) = \mathrm{Var}_{\theta \sim \theta_{\text{dist}}}[\partial_{\theta_i}\mathcal{L}]
$$

If $V(i; \ldots) \to 0$ rapidly as $n$ grows — faster than the Monte Carlo budget can shrink its own noise floor — concentration has collapsed the trainable signal.

The scaling law

$$
\log V_{\text{agg}} \sim -\alpha n + \beta \log n + c
$$

separates exponential ($\alpha > 0$, [[Barren Plateaus|BP regime]]), polynomial ($\alpha = 0$, $\beta < 0$), and constant ($\alpha = \beta = 0$) concentration rates.

## Glossary Note

From [[Glossary#Gradient concentration]]:

> "In this project, gradient concentration is the observable symptom we measure when diagnosing a barren plateau. If concentration strengthens rapidly with system size, training becomes less scalable."

## Related

- [[Barren Plateaus]]
- [[Gradient Variance]]
- [[Research Questions]]
