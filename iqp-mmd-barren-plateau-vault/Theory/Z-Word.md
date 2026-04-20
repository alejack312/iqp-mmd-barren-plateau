---
title: Z-Word
aliases:
  - Z-string
  - "Z_a"
tags:
  - theory
  - glossary
---

# Z-Word

A **Z-word** (also *Z-string*, or just *mask*) is a subset $S \subseteq [n]$ of qubits, encoded as a binary vector $a \in \{0,1\}^n$, associated with the Pauli observable $Z_S = \bigotimes_{i \in S} Z_i$.

## Expectation

The expectation of a Z-word under any binary distribution on $\{0,1\}^n$ is a **parity average**:

$$
\langle Z_S \rangle = \mathbb{E}_x\!\left[(-1)^{S \cdot x}\right] = \mathbb{E}_x[\chi_a(x)]
$$

where $\chi_a(x) = (-1)^{a \cdot x}$ is the Walsh character.

So Z-words are exactly the [[Parity Algebra|parity observables]] of the bitstring distribution, and the whole MMD² decomposition is a comparison of parity averages between $p$ and $q_\theta$.

## Z-Word Weight

The **Z-word weight** $|a|$ is the Hamming weight of the mask — the number of qubits $Z_S$ touches.

| Weight | Meaning |
|---|---|
| 0 | Trivial observable, $\langle Z_\emptyset\rangle = 1$ |
| 1 | Single-qubit marginal |
| 2 | Pairwise parity / correlation |
| 3 | Triple parity |
| … | … |
| $n$ | Global parity |

## Role in MMD²

The [[MMD Loss]] mixture form samples Z-words from a distribution $P_k(a)$ induced by the kernel:

$$
\mathrm{MMD}^2(p, q_\theta) = \mathbb{E}_{a \sim P_k}\!\left[(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta})^2\right]
$$

When the docs say "the Gaussian kernel emphasizes low-weight Z-words," they mean $P_G$ concentrates on masks with small $|a|$ — simple, low-order correlations are what the loss pays attention to.

## Role in Gradients

The parity-gate factor in [[Gradient Derivation]]:

$$
\partial_{\theta_i}\langle Z_a\rangle = -2 (a \cdot g_i \bmod 2) \cdot \mathbb{E}_z[\sin\Phi \cdot (-1)^{z \cdot g_i}]
$$

vanishes unless **Z-word $a$ has odd overlap with generator $g_i$**. This connects Z-words directly to the [[Generator Matrix]] — different families see different subsets of Z-words through different parameters.

## In the Code

Z-words are `np.ndarray` of shape `(n,)` with `dtype=uint8`, values in `{0, 1}`. In batches, they're shape `(B, n)`.

Built by [[Kernel Module|`sample_a(kernel, n, num_a_samples, rng, **params)`]] and consumed by [[Mixture Module|`dataset_expectations_batch`]] and [[IQP Expectation|`iqp_expectation`]].

## Related

- [[Parity Algebra]]
- [[Spectral Weight]]
- [[Generator Matrix]]
- [[MMD Loss]]
- [[Kernel Spectral Decomposition]]
