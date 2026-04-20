---
title: IQP Classical Sampling
tags:
  - theory
  - complexity
  - background
---

# IQP Classical Sampling

The complexity-theoretic backdrop for the project: sampling from the output distribution of IQP circuits is believed to be **classically hard**, yet expectation values used by MMD² are classically **efficient**. Understanding this tension is essential to the research motivation.

## The Hardness Result

From Bremner–Jozsa–Shepherd (and follow-ups):

> Efficient classical sampling from arbitrary IQP circuits would collapse the polynomial hierarchy to the third level.

This is a conditional hardness result, but the assumptions (stability of PH, average-case hardness of certain partition functions) are widely believed.

So sampling is hard, at least in the worst case and under plausible assumptions.

## The Classical Efficiency Result

At the same time, the expectation values needed for MMD² training:

$$
\langle Z_a \rangle_{q_\theta} = \mathbb{E}_{z \sim U}[\cos \Phi(\theta, z, a)]
$$

are **classically estimable with bounded-variance Monte Carlo** in time $O(B \cdot m \cdot n)$. The cosine is bounded, so $B = O(1/\epsilon^2)$ samples give an $\epsilon$-accurate estimate independent of $n$.

This is the den Nest-style classical estimator, used throughout [[IQP Expectation]].

## The Tension

The two results live side by side:

1. You **cannot** efficiently draw typical samples $x \sim q_\theta$ (sampling hardness).
2. You **can** efficiently compute $\langle Z_a \rangle_{q_\theta}$ for any observable (classical training feasibility).

This is the basis of the "train on classical, deploy on quantum" paradigm:

- Train the parameters $\theta$ entirely on classical hardware using the MMD² loss and the classical expectation estimator
- Only at inference time, run the trained circuit on a quantum device and sample from it

## Why This Is Subtle

> [!question] The open question
> If the learned distribution $q_\theta$ still has the sampling-hardness property of generic IQP circuits, then the trained model inherits quantum sampling advantage. But if training pushes $\theta$ into a region where the distribution becomes classically easy to sample, we lose the advantage.
>
> The [[Anti-Concentration]] study is one angle on this: if the learned distribution is **not** anti-concentrated, it may have collapsed to a small support that can be sampled classically.

## Four Central Questions

From [[Research Questions]]:

- **Q1** — do gradients vanish exponentially under this scheme? (trainability)
- **Q2** — does kernel choice or circuit structure control this?
- **Q3** — do shots/noise reintroduce barriers?
- **Q4** — **How can anticoncentration and sampling hardness results coexist with classical trainability of IQP generative models?**

Q4 is the philosophical question that motivates the whole project.

## Related

- [[Anti-Concentration]]
- [[IQP Circuits]]
- [[IQP Expectation]]
- [[Research Questions]]
- [[References#Rudolph 2023]]
