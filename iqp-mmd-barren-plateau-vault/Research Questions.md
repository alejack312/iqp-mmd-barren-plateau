---
title: Research Questions
tags:
  - research
  - questions
  - overview
---

# Research Questions

The project is organized around four central questions, each tied to specific experimental axes and analytic derivations.

## Q1. Asymptotic Gradient Scaling

Let $\mathcal{L}(\theta) = \mathrm{MMD}^2(p, q_\theta)$. Does

$$
\mathrm{Var}_{\theta \sim \mathcal{D}}[\partial_{\theta_i} \mathcal{L}]
$$

decay:

- **Exponentially** in $n$? (→ barren plateau, Outcome A)
- **Polynomially**? (→ mild concentration, Outcome B)
- **Remain constant** under structured regimes? (→ favourable, Outcome B/C)

The scaling hypothesis to fit:

$$
\log \mathrm{Var}(\partial_{\theta_i} \mathcal{L}) \sim -\alpha(F,K)\, n + \beta(F,K) \log n + c(F,K)
$$

- $\alpha > 0$, $\beta \approx 0$ → exponential decay
- $\alpha = 0$, $\beta < 0$ → polynomial decay
- $\alpha = 0$, $\beta = 0$ → constant

See [[Gradient Variance]] and [[Scaling Runner]] for the measurement path.

## Q2. Structural and Kernel Dependence

How does gradient variance depend on:

- Hypergraph sparsity and connectivity family? — see [[Families MOC]]
- Gate locality (k-local structure)?
- Overlap patterns of generators?
- Kernel type and bandwidth $\sigma$? — see [[Kernels MOC]]
- Initialization scheme? — see [[Initialization Schemes]]

> [!question] Specific sub-question
> For a fixed IQP circuit family, does the **choice of kernel** determine whether the loss exhibits a barren plateau — independent of circuit structure?

## Q3. Hardware and Sampling Effects

Even if analytic gradients do not vanish exponentially:

- Does **finite-shot estimation** induce effective plateaus?
- Does **hardware noise** suppress gradients?
- Is classical trainability preserved under realistic execution constraints?

Measured in the [[Qiskit Runner]] cross-check pipeline.

## Q4. Theoretical Compatibility

How can anticoncentration and sampling hardness results coexist with classical trainability of IQP generative models? See [[Anti-Concentration]] and [[IQP Classical Sampling]].

---

## Expected Outcomes

Each outcome corresponds to a possible conclusion shape:

| Outcome | Statement | Implication |
|---|---|---|
| **A** | Gradient variance scales $O(2^{-n})$ under generic conditions | Trainability depends on structure/init |
| **B** | Bounded-degree or structured graphs avoid exponential suppression | Commuting structure matters |
| **C** | MMD mixture suppresses concentration relative to standard VQA losses | Loss choice is central |
| **D** | Finite shots or noise reintroduce exponential suppression | Theoretical results don't translate to hardware |
| **E** | Small-angle initialization qualitatively changes scaling | Init is a necessary condition for trainability |

## Barren Plateau Question Matrix

For each cell in the 4×3 grid (four SMART families × three primary kernels), answer:

```
BP(F_i, K_j)?  →  Var_{θ~I}[∂_θ MMD²_{K_j}(p, q^{F_i}_θ)] = Θ(exp(−α·n))?
```

By May 15, 2026 (per [[SMART Spec]]): for each of the 12 cells, produce a stated conclusion of the form:

> "Under connectivity family $F_i$ with kernel $K_j$ and initialization $I$, $\mathrm{Var}(\partial_\theta \mathcal{L})$ scales [exponentially / polynomially / constant] with $n$; small-angle init [does / does not] change this; shot/noise effects [do / do not] reinstate exponential suppression."
