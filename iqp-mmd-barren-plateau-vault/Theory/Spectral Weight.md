---
title: Spectral Weight
tags:
  - theory
  - kernel
  - glossary
---

# Spectral Weight

A **spectral weight** is the coefficient assigned to one Fourier/Walsh mode in a kernel's expansion. It tells you how much that mode contributes to the loss.

## Formal Definition

For a kernel $k$ with Walsh-Fourier decomposition

$$
k(x, y) = \sum_{a \in \{0,1\}^n} w_k(a) \chi_a(x) \chi_a(y)
$$

the **spectral weight** of mode $a$ is $w_k(a)$.

## In the MMD² Mixture

Spectral weights induce a sampling distribution:

$$
P_k(a) = \frac{w_k(a)}{\sum_{a'} w_k(a')}
$$

and the MMD² estimator samples $a \sim P_k$, then averages squared residuals $(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta})^2$.

Large $w_k(a)$ → mismatch on mode $a$ matters a lot.
Small $w_k(a)$ → the loss mostly ignores mismatch on mode $a$.

## Spectral Weights Decaying

When we say spectral weights **decay**, we mean they shrink as the mode becomes more complex, usually as the Hamming weight $|a|$ of the Z-word increases.

Example: the Gaussian kernel has $w_G(a) \propto \tau^{|a|}$ with $0 < \tau < 1$. Higher-weight modes are progressively suppressed:

- weight-1 modes multiplied by $\tau$
- weight-2 modes multiplied by $\tau^2$
- weight-10 modes multiplied by $\tau^{10}$

Intuitively, the kernel focuses on low-order structure first and treats global many-body structure as less important.

## For Each Kernel

See the full table in [[Kernel Spectral Decomposition]].

| Kernel | $w_k(a)$ |
|---|---|
| Gaussian | $\tau^{\|a\|}$, $\tau = \tanh(1/(4\sigma^2))$ |
| Laplacian | Krawtchouk sum (awkward) |
| Multi-scale Gaussian | $\sum_i w_i \tau_i^{\|a\|}$ |
| Polynomial $d$ | supported on $\|a\| \le d$ |
| Linear | $1/n$ on weight-1, else 0 |

## Related

- [[Z-Word]]
- [[Kernel Spectral Decomposition]]
- [[MMD Loss]]
- [[Gaussian Kernel]]
- [[Glossary#Spectral weight]]
