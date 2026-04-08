---
title: Theory MOC
tags:
  - moc
  - theory
---

# Theory Map of Content

All mathematical machinery the project relies on. Each page links back into the [[Code MOC]] where the formula is implemented.

## Foundations

- [[IQP Circuits]] — definition, structure, $H^{\otimes n} D_\theta H^{\otimes n}$
- [[IQP Expectation]] — classical estimator $\langle Z_a \rangle_{q_\theta} = \mathbb{E}_z[\cos(\Phi(\theta,z,a))]$
- [[Generator Matrix]] — the `(m,n)` binary matrix `G` that drives everything
- [[Parity Algebra]] — why $(G a) \bmod 2$ and $(G z) \bmod 2$ are the only operations that matter

## Loss Function

- [[MMD Loss]] — the mixture-of-Z-words decomposition
- [[Kernel Spectral Decomposition]] — expressing $k(x,y)$ as a Walsh/Fourier sum
- [[Z-Word]] — $a \in \{0,1\}^n$ and the Pauli observable $Z_a$
- [[Spectral Weight]] — the $w_k(a)$ factor

## Gradients

- [[Gradient Derivation]] — analytic formula for $\partial_{\theta_i} \mathrm{MMD}^2$
- [[Gradient Variance]] — the primary scalar `V(i; θ_dist, F, K, n)`
- [[Barren Plateaus]] — the phenomenon, the scaling law, and the three regimes

## Distribution Diagnostics

- [[Anti-Concentration]] — second-moment and threshold forms
- [[Walsh-Hadamard Transform]] — how we go from diagonal phases to exact amplitudes
- [[IQP Classical Sampling]] — the complexity-theoretic backdrop

## Experimental Design

- [[Initialization Schemes]] — uniform, small-angle, data-dependent
- [[Families MOC]] — IQP connectivity families
- [[Kernels MOC]] — MMD kernels
- [[Datasets]] — product Bernoulli, Ising, binary mixture
- [[Learning Task]] — what the model is trying to do

## Convention Locks

- [[Locked MMD² Derivation]] — the exact formula the code implements
- [[Gaussian Convention]] — $k(x,y) = \exp(-H/(2\sigma^2))$, $\tau = \tanh(1/(4\sigma^2))$
- [[Theory-Implementation Parity]] — why convention locks matter
