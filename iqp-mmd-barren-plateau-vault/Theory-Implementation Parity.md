---
title: Theory-Implementation Parity
tags:
  - methodology
  - convention
---

# Theory-Implementation Parity

**Theory/implementation parity** means the formula in the docs and the formula in the code are **the same object**, not merely qualitatively similar.

## Example

If the theory says the Gaussian spectrum uses a particular normalization or bandwidth parameterization, the implementation should expose the same choice explicitly:

- Same $\sigma$ convention across kernel, spectral weights, and sampler
- Same normalization constant where it appears in reported MMD² values
- Same factor conventions on gradients (e.g., the $-2$ in $\partial_{\theta_i}\mathrm{MMD}^2$)

## Why This Matters

Two formulas can look **qualitatively correct** — decay in $|a|$, bounded spectrum, right asymptotic shape — and still be numerically different enough that:

- Monte Carlo estimators don't sample from what they claim to
- Gradient variances get multiplied by wrong constants
- Cross-validation against exact paths silently disagrees on order-of-unit factors
- Regression fits get the wrong scaling exponent $\alpha$ in $\log V \sim -\alpha n$

In a barren plateau study, a factor-of-2 mismatch in the kernel decay constant could turn a "borderline polynomial" result into an "exponential" result or vice versa. That's not a safety margin you can blow.

## The Discipline

1. **Write the formula in the docs first** — see [[Locked MMD² Derivation]]
2. **Cite the docs from the code** — module docstrings reference the glossary entry (see e.g. `kernel.py` pointing to [[Gaussian Convention]])
3. **Test invariants, not snapshots** — see [[Tests]]
4. **Change all related code together** — if you touch the kernel, touch the spectral weights and sampler in the same commit

## Where Parity Is Currently Locked

- **Gaussian** — locked, see [[Gaussian Convention]]
- **Laplacian** — explicit stub, parity not yet achieved
- **Multi-scale Gaussian** — implemented, exact validation pending
- **IQP expectation** — locked, the phase formula is the same in docs and code
- **MMD² estimator** — locked for Gaussian only

## Where Parity Is Open

- **Laplacian MMD² decomposition** — [[TODO Roadmap|T2]]
- **Small-n exact MMD² cross-check** for each supported kernel — D2.1

## Related

- [[Locked MMD² Derivation]]
- [[Gaussian Convention]]
- [[Implementation Choices]]
- [[Glossary]]
