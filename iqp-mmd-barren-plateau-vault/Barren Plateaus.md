---
title: Barren Plateaus
tags:
  - theory
  - barren-plateau
  - background
---

# Barren Plateaus

The phenomenon the entire project is trying to test for in the IQP+MMD setting.

## Definition

A **barren plateau** is the regime where the variance of the loss gradient decays **exponentially** with system size $n$:

$$
\mathrm{Var}_{\theta \sim \mathcal{D}}[\partial_{\theta_i} \mathcal{L}] = O(2^{-\alpha n}) \quad \text{for some } \alpha > 0
$$

When this happens:

- Most random initializations give almost the same tiny gradient
- Distinguishing a useful descent direction requires an **exponential number of shots**
- Optimizers become dominated by estimator noise, shot noise, or finite precision
- Training does not scale

See [[Gradient Concentration]] for the observable symptom.

## Why This Paper Exists

From the [[Project Overview]]:

> "Even if gradients can be computed classically, does their variance decay exponentially with system size, or does the special commuting structure of IQP circuits prevent this?"

Standard random-VQA results (McClean et al. 2018, Larocca et al. 2024) say that for generic unitary-2-design circuits, barren plateaus are the rule. IQP is **not** a unitary 2-design — it is a much more structured, commuting gate set. That structure might protect trainability; it also might not. This project measures which.

## Average-Case Framing

Recent [Larocca et al. 2024](References#Larocca%202024) framing: barren plateaus are an **average-case statement over the landscape**, not pointwise. That leaves open the existence of **trainable valleys** — regions of the parameter space where gradients are well-behaved even if the ensemble average is not.

This matters for this project because:

- [[Initialization Schemes|Small-angle initialization]] may target a trainable valley near $\theta = 0$
- Warm-start results (Mhiri et al. 2025) show perturbations around favorable starting points can avoid exponential gradient suppression

## Five Possible Conclusions

From [[Research Questions]], the expected outcomes are:

- **A.** Generic exponential decay ($V \sim 2^{-\alpha n}$)
- **B.** Structured avoidance (bounded-degree or lattice families preserve trainability)
- **C.** Loss-induced regularization (kernel choice determines the outcome)
- **D.** Hardware-induced plateaus (finite shots or noise re-introduce exponential suppression)
- **E.** Init-dependent trainability (small-angle qualitatively changes scaling)

## Measurement

The project operationalizes "barren plateau" as a **variance scaling law** on the metric $V_{\text{agg}}(F, K, n)$ — see [[Gradient Variance]]. Different regimes are distinguished by fitting:

$$
\log V_{\text{agg}} \sim -\alpha n + \beta \log n + c
$$

with explicit cells for each (family $F$, kernel $K$, init $I$, dataset $D$) combination.

## Related

- [[Gradient Variance]]
- [[Gradient Concentration]]
- [[Research Questions]]
- [[Initialization Schemes]]
- [[Scaling Runner]]
- [[References]] — Rudolph, Larocca, Mhiri
