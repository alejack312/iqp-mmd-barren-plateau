---
title: Kernels MOC
tags:
  - moc
  - kernels
---

# Kernels — Map of Content

The MMD loss depends on a kernel $k(x, y)$. Different kernels induce **different spectral weights** $w_k(a)$ over Z-word modes, which in turn change the loss landscape.

## Primary Study Order

From the [[SMART Spec]]:

1. **Phase 1** — [[Gaussian Kernel]] exhaustively across all four [[Families MOC|connectivity families]]
2. **Phase 2** — [[Laplacian Kernel]] + [[Multi-Scale Gaussian Kernel]]

## The Kernels

| Kernel | $k(x, y)$ | Spectral weight | Phase |
|---|---|---|---|
| **Gaussian** | $e^{-H(x,y)/(2\sigma^2)}$ | $w_G(a) \propto \tanh(1/(4\sigma^2))^{|a|}$ | 1 |
| **Laplacian** | $e^{-\sqrt{H(x,y)}/\sigma}$ | heavier tails; exact via Krawtchouk sum | 2 |
| **Multi-Scale Gaussian** | $\sum_i w_i\, e^{-H/(2\sigma_i^2)}$ | mixture of Gaussians | 2 |
| **Polynomial** | $(x \cdot y / n + c)^d$ | support only on $|a| \le d$ | Legacy |
| **Linear** | $x \cdot y / n$ | only weight-1 Z-words | Legacy |

See [[Kernel Module]] for the implementation and [[Kernel Spectral Decomposition]] for the theory.

## Kernel Pages

- [[Gaussian Kernel]]
- [[Laplacian Kernel]]
- [[Multi-Scale Gaussian Kernel]]
- [[Polynomial Kernel]]
- [[Linear Kernel]]

## Convention Locks

- [[Gaussian Convention]] — $k(x,y) = e^{-H/(2\sigma^2)}$, Walsh decay $\tau = \tanh(1/(4\sigma^2))$
- [[Locked MMD² Derivation]] — the exact formula the code implements
- [[Theory-Implementation Parity]] — why these locks matter

## The Q2 Question

> *For a fixed IQP circuit family, does the choice of kernel determine whether the loss exhibits a barren plateau?*

The Phase 1 + Phase 2 sweep is organized to answer this cell-by-cell in the 4×3 grid. See [[Research Questions]].

## Related

- [[Kernel Spectral Decomposition]]
- [[Kernel Module]]
- [[MMD Loss]]
