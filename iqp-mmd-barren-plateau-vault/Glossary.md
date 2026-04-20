---
title: Glossary
tags:
  - glossary
  - reference
---

# Glossary

Project-specific vocabulary and notation. Cross-linked into the theory and code notes. See also [`docs/technical/glossary.md`](../docs/technical/glossary.md) for the source document.

## Locked

**Locked** means the version currently fixed for the agreed study scope. Not mathematically proven forever — a shared contract the team has chosen to implement and compare against.

Examples: [[Locked MMD² Derivation|locked MMD² derivation]], [[Scope Lock|locked SMART scope]], locked experiment axes.

## Theory / Implementation Parity

The formula in the docs and the formula in the code are the same object, not merely qualitatively similar. See [[Theory-Implementation Parity]].

## IQP Circuit Family

A **family** is a rule for constructing the [[Generator Matrix|generator matrix G]] as the system size $n$ changes. Primary four: product state, ZZ lattice, sparse Erdős–Rényi, complete graph. See [[Families MOC]].

## Generator Matrix `G`

Binary matrix of shape `(m, n)` that specifies which qubits each IQP generator acts on. Row $j$ is the bitmask $g_j$ for generator $\exp(i\theta_j X^{g_j})$. See [[Generator Matrix]].

## Generator / Hyperedge

One row of `G`. Three roles at once: defines one interaction pattern, identifies which parameter $\theta_j$ controls it, determines which Z-word modes can see it through `(a · g_j mod 2)`.

## Product-State Family

No-entanglement baseline. $G = I_n$. See [[Product State Family]].

## ZZ Lattice Family

Local-interaction baseline. Open-boundary nearest-neighbor pairs on an $L \times L$ square grid. $n$ must be a perfect square, `range_=1`, $m = 2L(L-1)$. See [[ZZ Lattice Family]].

## Sparse Erdős–Rényi Family

Random-connectivity baseline, pairwise. `p_edge` interpreted as target average degree $c$; edge probability $\min(1, c/n)$. Row count intrinsic. See [[Erdos-Renyi Family]].

## Complete-Graph Family

Densest pairwise baseline. Every qubit pair interacts. $m = n(n-1)/2$. See [[Complete Graph Family]].

## Pairwise Baseline

A reference circuit family built only from two-qubit interactions. All four SMART families are pairwise.

## Hamming Weight

The number of 1s in a binary mask. For a generator, the number of qubits in the interaction. For a Z-word, the number of qubits in the observable.

## Mask / Binary Mask / Binary Selector

A binary selector — $0/1$ vector that marks which positions are included. In this repo, masks come in two flavors: generator masks $g_j$ and Z-word masks $a$.

## Mode

One Fourier/Walsh component indexed by a subset of qubits $S$ or equivalently by a binary mask $a$. Each mode corresponds to one Z-word observable $Z_S$. See [[Spectral Weight]].

## Z-Word / Z-String

Subset $S$ of qubits, encoded as $a \in \{0,1\}^n$, associated with the Pauli observable $Z_S$. Its expectation is a parity average. See [[Z-Word]].

## Z-Word Weight

The Hamming weight of the Z-word mask. Low weight → single-qubit or pairwise structure; high weight → global structure.

## Pairwise Parity

The parity of two selected bits — whether $x_i + x_j \bmod 2$ is even or odd. A weight-2 Z-word observable.

## Parity

Whether the number of selected 1-bits is even or odd. For bitstring $x$ and mask $a$, $(a \cdot x \bmod 2)$ is 0 for even parity and 1 for odd. The sign $(-1)^{a \cdot x}$ is $+1/-1$. See [[Parity Algebra]].

## Spectral Distribution $P_\sigma$

The sampling distribution over Z-word modes induced by the kernel spectrum. For Gaussian, $P_\sigma(a) \propto \tau^{|a|}$ with $\tau = \tanh(1/(4\sigma^2))$. See [[Kernel Spectral Decomposition]].

## Spectral Weight

Coefficient assigned to one Fourier/Z-word mode in the kernel expansion. Tells you how much that mode contributes to the loss. See [[Spectral Weight]].

## Spectral Weights Decaying

Spectral weights shrink as the mode becomes more complex, usually as Z-word weight $|a|$ increases. For Gaussian: $\tau^{|a|}$ with $0 < \tau < 1$. Higher-weight modes are progressively suppressed.

## Interaction Pattern

The subset of qubits coupled together by one generator. Weight-1 = single-site; weight-2 = pairwise; higher = many-body.

## Gaussian Spectral Normalization

The exact constant-and-parameter convention used when writing the Gaussian kernel in Walsh form. Includes global prefactor, decay parameter $\tau$, and the $\sigma$ reparameterization. Locked to $k(x,y) = \exp(-H/(2\sigma^2))$, $\tau = \tanh(1/(4\sigma^2))$. See [[Gaussian Convention]].

## Bandwidth $\sigma$

The Gaussian kernel scale parameter. Small $\sigma$ → higher-order structure matters more; large $\sigma$ → emphasis on low-order structure.

## Barren Plateau

The regime where gradient variance decays exponentially with system size $n$, making optimization infeasible at scale. See [[Barren Plateaus]].

## Gradient Concentration

The observable symptom of a barren plateau: gradients from different random parameter draws cluster tightly near zero. See [[Gradient Concentration]].

## Scaling Law

The fitted dependence of a quantity (usually gradient variance) on system size. Exponential / polynomial / constant regimes are the three possible outcomes.

## Related

- [[Theory MOC]]
- [[Code MOC]]
- [[Planning MOC]]
