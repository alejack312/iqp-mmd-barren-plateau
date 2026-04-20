---
title: Gaussian Convention
tags:
  - theory
  - lock
  - gaussian
---

# Gaussian Convention

The exact constant and parameter convention the code uses for the Gaussian kernel and its spectral weights. Locked in code and docs.

## The Lock

```text
k(x, y) = exp(-H(x, y) / (2 sigma^2))
```

where $H(x, y)$ is Hamming distance.

Once this is fixed, the per-bit mismatch factor is:

$$
q = \exp\!\left(-\frac{1}{2\sigma^2}\right)
$$

and the Walsh decay constant is:

$$
\tau = \frac{1 - q}{1 + q} = \tanh\!\left(\frac{1}{4\sigma^2}\right)
$$

Spectral weights:

$$
w_G(a) \propto \tau^{|a|}
$$

## The Drift That Prompted the Lock

Before the lock, different files used different conventions:

- `gaussian_kernel` — sometimes $\exp(-H/\sigma^2)$, sometimes $\exp(-H/(2\sigma^2))$
- `gaussian_spectral_weights` — sometimes $\tanh(1/\sigma^2)$, sometimes $\tanh(1/(2\sigma^2))$
- `gaussian_sample_a` — sometimes sampling from a different $\tau$ than the spectral weights

Any combination of these was "qualitatively correct" because the shape of the decay is similar, but the Monte Carlo MMD² estimator was not sampling from the kernel it claimed to use. The symptom only became visible once the Gaussian tests were tight enough to probe the path directly — at which point the [[Mixture Module#The Parity Sign Bug|parity sign bug]] also surfaced.

See [[Implementation Choices#The Gaussian convention]] for the full story.

## Where the Lock Lives

Three functions in [`src/iqp_bp/mmd/kernel.py`](../src/iqp_bp/mmd/kernel.py):

- `gaussian_kernel(x, y, sigma)` — direct kernel
- `gaussian_spectral_weights(n, sigma)` — spectral weights array
- `gaussian_sample_a(n, num_a_samples, sigma, rng)` — sampler

All three use the same `_gaussian_tau(sigma)` helper.

## Discipline Rule

> [!warning] Change all three together
> If you change any Gaussian formula, update the kernel, the spectral weights, and the sampler in the same commit. Convention drift is silent and catastrophic.

From [[Implementation Choices#If you are changing this code later]]:

> "If you change the Gaussian formula, change the Walsh decay and the sampler in the same commit."

## Source

The convention follows the paper in [`docs/papers/2503.02934v2 (3).pdf`](../docs/papers/2503.02934v2%20(3).pdf) — the [[References#Scaling GQML|Scaling GQML]] paper.

## Related

- [[Gaussian Kernel]]
- [[Kernel Module]]
- [[Locked MMD² Derivation]]
- [[Implementation Choices]]
- [[Theory-Implementation Parity]]
