---
title: Learning Task
tags:
  - theory
  - task
---

# The Learning Task

What exactly is the model being trained to do? This page summarizes the learning setup to anchor the rest of the study.

## Setup

- **Inputs**: $n$-bit binary strings $x \in \{0,1\}^n$
- **Target**: a distribution $p(x)$ we want to approximate
- **Model**: an IQP circuit with parameters $\theta$ producing distribution $q_\theta(x) = |\langle x|\psi(\theta)\rangle|^2$
- **Loss**: $\mathrm{MMD}^2(p, q_\theta)$ with a chosen kernel $k$
- **Optimizer**: classical gradient descent on $\theta$, using the classical estimator for $\langle Z_a\rangle_{q_\theta}$

## The Training Loop (Conceptually)

```python
θ ← init_scheme()
while not converged:
    θ ← θ - η · ∇_θ MMD²(p, q_θ)
return θ
```

with $\nabla_\theta \mathrm{MMD}^2$ computed via [[Gradient Derivation|the analytic formula]] and $\langle Z_a\rangle_{q_\theta}$ via [[IQP Expectation|Monte Carlo]]. The exact training loop lives in [[iqp_mmd Package|`iqp_mmd`]]; `iqp_bp` itself does not run gradient descent — it only measures gradient variance.

## Train on Classical, Deploy on Quantum

The motivation is the paradigm from [[References#Scaling GQML|Rudolph et al. 2023]]:

1. **Training** happens entirely on classical hardware because expectation values are efficient
2. **Inference** happens on a quantum device by literally running the trained IQP circuit and sampling

This lets the trained model retain the sampling hardness of IQP circuits (if anti-concentration is preserved) while making training feasible.

## What `iqp_bp` Tests About This

`iqp_bp` is a **diagnostic** on the training pipeline, not a training pipeline itself:

- **Gradient variance** — will training even work at scale? ([[Barren Plateaus]])
- **Anti-concentration** — after training, does $q_\theta$ stay spread out? ([[Anti-Concentration]])

If gradient variance decays exponentially, training can't scale.
If anti-concentration fails, the trained model can be classically sampled and the quantum-deploy advantage evaporates.

Both diagnostics must pass.

## Three Targets

See [[Datasets]] for the three target distributions:

- **D1 Product Bernoulli** — primary, isolates circuit/kernel from data structure
- **D2 Ising** — pairwise correlations aligned with lattice topology
- **D3 Binary mixture** — multi-modal, closest to real data

## Related

- [[Project Overview]]
- [[MMD Loss]]
- [[Research Questions]]
- [[iqp_mmd Package]]
- [[IQP Classical Sampling]]
