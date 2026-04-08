---
title: Kernel Spectral Decomposition
tags:
  - theory
  - kernel
  - spectral
---

# Kernel Spectral Decomposition

The identity that turns the MMD² over $\{0,1\}^n$ into a sum over Pauli-Z observables. It's the reason the whole project's classical estimator works.

## The Identity

For any kernel $k : \{+/-1\}^n \times \{+/-1\}^n \to \mathbb{R}$ that depends only on the difference $x \oplus y$ (or equivalently on Hamming distance in the $\{0,1\}$ encoding), the Walsh-Fourier decomposition gives:

$$
k(x, y) = \sum_{a \in \{0,1\}^n} w_k(a) \cdot \chi_a(x) \chi_a(y)
$$

where $\chi_a(x) = (-1)^{a \cdot x}$ is the Walsh character (equivalently $Z_a$ expectation in the $\pm 1$ encoding). Plugging into the MMD² formula gives:

$$
\mathrm{MMD}^2(p, q) = \sum_a w_k(a) \left(\mathbb{E}_p[\chi_a] - \mathbb{E}_q[\chi_a]\right)^2
= \sum_a w_k(a)\left(\langle Z_a\rangle_p - \langle Z_a\rangle_q\right)^2
$$

This is the form used throughout the code: see [[MMD Loss]].

## Spectral Weights for Each Kernel

| Kernel | Formula | Spectral weight $w_k(a)$ |
|---|---|---|
| Gaussian | $e^{-H/(2\sigma^2)}$ | $\propto \tau^{\|a\|}$, $\tau = \tanh(1/(4\sigma^2))$ |
| Laplacian | $e^{-\sqrt{H}/\sigma}$ | Krawtchouk sum over $h$, see [[Laplacian Kernel]] |
| Multi-scale Gaussian | $\sum_i w_i e^{-H/(2\sigma_i^2)}$ | $\sum_i w_i \tau_i^{\|a\|}$ |
| Polynomial $d$ | $(x \cdot y/n + c)^d$ | supported only on $\|a\| \le d$ |
| Linear | $x \cdot y / n$ | $1/n$ if $\|a\|=1$, else 0 |

## Why the Gaussian Is "Nice"

The Gaussian decomposes completely into a product over qubits: each bit independently contributes a factor that is either $\tau$ (if $a_i = 1$) or $1$ (if $a_i = 0$). This means:

$$
w_G(a) \propto \prod_{i=1}^n \tau^{a_i} = \tau^{|a|}
$$

which has two big practical payoffs:

1. **Weight-stratified sampling** — to sample $a \sim P_G$, first sample the Hamming weight $w \sim \text{Binomial-like}(n, \tau)$, then pick a uniform subset of that size. This is what `gaussian_sample_a` does.
2. **Exponential suppression of high-weight modes** — higher-order correlations don't matter much for small $\sigma$, which is what makes the Gaussian the "smooth" kernel baseline.

## Why the Laplacian Is Awkward

The Laplacian kernel has the form $e^{-\sqrt{H}/\sigma}$, and $\sqrt{H}$ does **not** decompose qubit-by-qubit. Its spectral weights require a Krawtchouk polynomial sum:

$$
w_L(a; \sigma) = \frac{1}{2^n}\sum_h K_{|a|}(h; n) e^{-\sqrt{h}/\sigma}
$$

These weights can be **negative** or **non-monotone** in $|a|$, which is why the current sampler is an explicit stub and the MMD² decomposition is not yet locked. See [[Laplacian Kernel]] and [[TODO Roadmap|T2]].

## Related

- [[MMD Loss]]
- [[Spectral Weight]]
- [[Z-Word]]
- [[Gaussian Kernel]]
- [[Locked MMD² Derivation]]
