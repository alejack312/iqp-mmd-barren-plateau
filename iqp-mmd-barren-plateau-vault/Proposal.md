---
title: Proposal
tags:
  - planning
  - proposal
---

# Thesis Proposal

Condensed summary of [`docs/Proposal.md`](../docs/Proposal.md).

## Title

**Gradient Concentration and Barren Plateau Phenomena in IQP-Based Quantum Generative Models: A Theoretical, Computational, and Circuit-Level Study**

## Four Integrated Components

The project structures the investigation around four components, all exercising the same formal definition of IQP+MMD:

### Part I — Analytical Derivation

Using the classical IQP representation:

$$
\langle Z_a\rangle_{q_\theta} = \mathbb{E}_{z \sim U}[\cos\Phi(\theta, z, a)]
$$

derive:

1. $\partial_{\theta_i}\langle Z_a\rangle_{q_\theta}$ — see [[Gradient Derivation]]
2. The MMD² gradient as a weighted mixture over Pauli-Z derivatives
3. Explicit MMD² forms per (kernel family, circuit family)
4. A closed-form structural expression for gradient variance
5. Dependence on hyperedge overlaps, degree statistics, commuting structure, kernel spectrum, init scheme

### Part II — Structured Computational Exploration (Hypothesis-Based)

Using Python's [Hypothesis](https://hypothesis.readthedocs.io/) for property-based generation, study four IQP families × three kernels × three inits at exact numerical gradient level:

- Compute exact expectation values
- Compute exact gradients (noise-free)
- Estimate gradient variance
- Fit scaling laws $\log V \sim -\alpha n + \beta \log n + c$

### Part III — Qiskit Circuit-Level Validation

Bridge the analytic path to executable circuits:

1. Automatically construct Qiskit IQP circuits from `G` (see [[Qiskit Circuit Builder]])
2. Evaluate under statevector, shot-based, and Aer noise simulation
3. Compare analytic vs statevector vs shot-based vs noisy
4. Determine whether finite-shot/noise effects reinstate plateaus

See [[Qiskit Runner]].

### Part IV — Structural Modeling in Forge

Use Forge (bounded-model-finding) to:

- Model hypergraph overlap patterns
- Identify minimal plateau-inducing configurations
- Explore symmetry-induced constraints
- Analyze structural invariants of commuting generators

See [[Forge Runner]] and [[Forge Export]].

## Expected Outcomes

| Label | Shape | Implication |
|---|---|---|
| A | Generic exponential decay | Trainability depends on structure/init |
| B | Structured avoidance | Commuting structure fundamentally alters BP |
| C | Loss-induced regularization | Kernel choice central to scalability |
| D | Hardware-induced plateaus | Theory doesn't translate to hardware |
| E | Init-dependent trainability | Small-angle init is necessary, not just a heuristic |

See [[Research Questions]] for the full list.

## Timeline

From the proposal (also restated in [[SMART Spec]]):

- **Month 1** — derive analytic gradients; implement exact classical pipeline
- **Month 2** — structured scaling experiments via Hypothesis; identify candidate regimes
- **Month 3** — implement Qiskit generator; statevector + shot-based comparisons
- **Month 4** — noise experiments; Forge structural modeling; thesis writing

## Deliverables

1. Formal derivation of gradient variance expressions
2. Scaling study across structured hypergraph families
3. Qiskit pipeline for IQP circuit generation
4. Comparative gradient variance analysis (exact vs shot vs noisy)
5. Structural classification of plateau-inducing regimes
6. Forge-based structural modeling results
7. Final thesis report synthesizing theory, computation, and experiment

## Related

- [[SMART Spec]]
- [[Scope Lock]]
- [[Background and Motivation]]
- [[Research Questions]]
- [[References]]
