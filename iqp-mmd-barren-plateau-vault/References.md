---
title: References
tags:
  - references
  - papers
---

# References

Cited papers and their role in the project. The PDFs live under [`docs/papers/`](../docs/papers/).

## Core References

### Recio-Armengol et al. 2025 — Scaling GQML

**Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits**
Erik Recio-Armengol, Shahnawaz Ahmed, and Joseph Bowles (Xanadu + ICFO + Eurecat)
arXiv:[2503.02934](https://arxiv.org/abs/2503.02934) (v1 Mar 2025, v2 Feb 2026)

The paper that motivates the whole classical-training paradigm. Shows that MMD loss expectations for IQP circuits can be computed classically and that training is feasible up to ~1000 qubits. The [[Gaussian Convention]] follows this paper's convention. The [[iqp_mmd Package]] is a modular reimplementation of its pipeline. The [[Locked MMD² Derivation]] walks through Secs. 3.3–4 of the paper in detail.

The data-dependent init recipe used as "Recipe A" in [[Data-Dependent Initialization Across Architectures|Data-Dependent Initialization Across Families]] comes from Sec. 8.1.2 of this paper.

> [!warning] Attribution fix
> Earlier versions of this note incorrectly credited 2503.02934 to "Rudolph et al." The actual authors are Recio-Armengol, Ahmed, and Bowles. Rudolph et al. *is* cited by this paper (for the MMD² mixture form, Rudolph et al. 2024, which is the precursor to Prop. 2), but they are not authors of 2503.02934. The separate "Rudolph 2023 — Trainability Barriers" entry below is a different paper (2305.02881) that is actually authored by Manuel S. Rudolph and collaborators.

**PDF:** `docs/papers/2503.02934v2 (3).pdf`

### Rudolph 2023 — Trainability Barriers

**Trainability barriers and opportunities in quantum generative modeling**
arXiv:[2305.02881](https://arxiv.org/abs/2305.02881)

Shows that the MMD loss has favorable trainability properties compared to explicit losses like KL divergence, which can introduce a new flavor of barren plateaus. One of the earliest indications that the **choice of loss** matters as much as the choice of circuit.

**PDF:** `docs/papers/2305.02881-implicit-explicit-losses.pdf`

### Larocca 2024 — BP Review

**Barren Plateaus in Variational Quantum Computing**
Nature Reviews Physics 7, 174–189 (2024). arXiv:[2405.00781](https://arxiv.org/abs/2405.00781)

The review that frames barren plateaus as an **average-case statement** over the landscape, leaving open the possibility of trainable valleys. Important theoretical backbone for Outcome E (init-dependent trainability).

**PDF:** `docs/papers/2405.00781-barren-plateau-review.pdf`

### Mhiri 2025 — Warm Start Guarantees

**A unifying account of warm start guarantees for patches of quantum landscapes**
arXiv:[2502.07889](https://arxiv.org/abs/2502.07889)

Shows that perturbations around favorable starting points can avoid exponential gradient suppression. Motivates the [[Initialization Schemes|small-angle init]] hypothesis as a principled trainable-valley strategy.

**PDF:** `docs/papers/2502.07889-warm-start-guarantees.pdf`

## Anti-Concentration Paper

**Paper 2512.24801 v1** (locked reference for [[Anti-Concentration]])

**PDF:** `docs/papers/2512.24801v1.pdf`

The paper that defines the anti-concentration criterion used by the deterministic validation runner:

$$
\Pr_x[p(x) \ge \alpha 2^{-n}] \ge \beta \quad \text{vs.} \quad 2^{2n}\mathbb{E}_x[p(x)^2] \ge \beta' > 1
$$

## Complexity-Theoretic Background

- **1504.07999** — IQP sampling hardness
- **1610.01808** — follow-up complexity results
- **2012.09265** — IQP + anticoncentration
- **2512.24801** — recent anti-concentration result (used by this project)

These PDFs are all under `docs/papers/`. They motivate the [[IQP Classical Sampling]] backdrop.

## Summary Table

| Reference | Role in project |
|---|---|
| Rudolph et al. 2023 (2305.02881) | "Loss matters" — kernel-induced plateau avoidance |
| Recio-Armengol et al. 2025 (2503.02934) | Classical training at 1000 qubits; Gaussian convention; MMD-as-observable derivation |
| Larocca 2024 (2405.00781) | BP theory; average-case landscape framing |
| Mhiri 2025 (2502.07889) | Warm-start trainable-valley guarantees |
| Paper 2512.24801 | Anti-concentration criterion |

## Related

- [[Background and Motivation]]
- [[Proposal]]
- [[IQP Classical Sampling]]
- [[Barren Plateaus]]
- [[Anti-Concentration]]
