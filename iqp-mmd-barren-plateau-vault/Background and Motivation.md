---
title: Background and Motivation
tags:
  - planning
  - background
---

# Background and Motivation

Condensed from `docs/Background and Project Summary.md`.

## Context

Variational Quantum Algorithms (VQAs) optimize parameterized quantum circuits against a loss function — typically the expectation value of an observable. They underpin most near-term quantum machine learning proposals. A compelling direction is **quantum generative modeling**: learn an unknown distribution from data and generate new samples from it.

Quantum circuits might offer an advantage here because certain architectures — notably IQP circuits — are believed to be **classically hard to sample from**.

## The IQP Sampling-Hardness Backdrop

Complexity results (Bremner–Jozsa–Shepherd) show that efficiently sampling from IQP circuit outputs would collapse the polynomial hierarchy under plausible assumptions. This suggests IQP circuits can represent distributions that are classically difficult to reproduce — a potential quantum advantage **at the sampling level**.

## The Barren Plateau Obstacle

For many random VQA architectures, gradient variance decays exponentially in $n$. In practice this means **shots to distinguish a gradient from zero** scale exponentially with system size, making training infeasible at scale. This has significantly limited the scalability of quantum generative models.

## The "Train Classically" Recent Development

Recent work [[References#Scaling GQML|(Rudolph et al. 2023; arXiv:2503.02934)]] shows that while **sampling** from IQP circuits is believed to be classically hard, the **expectation values** needed for MMD training are classically computable. This suggests a new paradigm:

> **Train on classical, deploy on quantum.**
> Optimize parameters entirely on classical hardware, then only after training, deploy the circuit on a quantum device for sampling.

## The Central Tension

This creates a three-way tension:

1. IQP sampling is hard
2. IQP MMD training is classically feasible
3. Standard VQAs suffer barren plateaus

Do the MMD loss landscapes for IQP circuits still exhibit exponential gradient concentration even though the gradients are classically computable? This is the question the project tries to answer.

## Scope

Across the four axes — circuit family, kernel, initialization, system size — the central analytical question is whether the induced MMD loss landscapes exhibit barren plateaus, and in particular whether **small-angle initialization** can improve trainability.

## Related

- [[Project Overview]]
- [[Research Questions]]
- [[IQP Classical Sampling]]
- [[Barren Plateaus]]
- [[Proposal]]
- [[References]]
