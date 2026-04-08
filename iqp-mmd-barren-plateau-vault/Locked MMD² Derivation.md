---
title: Locked MMD² Derivation
tags:
  - theory
  - convention
  - lock
---

# Locked MMD² Derivation

The specific analytic form of the MMD² loss that the code is guaranteed to implement. "Locked" means the team has fixed a concrete formulation for the current study phase — not a mathematical theorem, a shared contract.

## Why a Lock?

> [!important] One formula, everywhere
> - Theory needs one unambiguous formula for proofs and scaling arguments.
> - Implementation needs one unambiguous formula for `sample_a()`, kernel weights, and estimators.
> - If these drift, the Monte Carlo estimator is no longer sampling from the kernel it claims to use.

The project went through a period where different files quietly used different meanings of `sigma` and different Walsh decay constants. The "lock" is the discipline that closed that drift.

## The Locked Gaussian Formula

For the Gaussian kernel case:

$$
\mathrm{MMD}^2_\sigma(p, q) = C \sum_{S \subseteq [n]} \tau^{|S|}\left(\langle Z_S\rangle_p - \langle Z_S\rangle_q\right)^2
$$

with:

- $k(x, y) = \exp(-H(x, y)/(2\sigma^2))$ — the kernel
- $\tau = \tanh(1/(4\sigma^2))$ — the Walsh decay
- Sampling distribution over Z-word masks $S$: $P(S) \propto \tau^{|S|}$, with weight-stratified sampling

All three — `gaussian_kernel`, `gaussian_spectral_weights`, `gaussian_sample_a` — now agree on this exact convention. See [[Gaussian Convention]].

## What "Locked" Does NOT Mean

- **Not mathematically proven forever.** The locked formula is the one we're comparing against *during this study phase*. If the derivation is later tightened, the lock moves.
- **Not all kernels.** Gaussian is locked. Laplacian is explicitly stubbed until its decomposition is derived. Multi-scale Gaussian is validated but not fully exact-mixture-checked.
- **Not frozen as a test.** Tests check invariants like "sum of weights equals something sensible" and "MC path agrees with exact on small $n$" — not frozen floats.

## Remaining Lock Work

From [[TODO Roadmap|T2]]:

- Laplacian MMD² decomposition derivation + lock
- Multi-scale Gaussian exact mixture validation

## Related

- [[MMD Loss]]
- [[Kernel Spectral Decomposition]]
- [[Gaussian Convention]]
- [[Theory-Implementation Parity]]
- [[Glossary#Locked]]
