---
title: Scope Lock
tags:
  - planning
  - scope
---

# Scope Lock — Study Definition (Mar 18, 2026)

Condensed from [`ScopeLock.md`](../ScopeLock.md). The source of truth for what the project is measuring and how.

## 1. Model Definition

### IQP Circuit Family

$$
|\psi(\theta)\rangle = H^{\otimes n} \cdot \exp\!\left(i \sum_j \theta_j X^{g_j}\right) \cdot H^{\otimes n} |0\rangle^{\otimes n}
$$

### Classical Expectation Estimator

$$
\langle Z_a\rangle_{q_\theta} = \mathbb{E}_{z \sim U}[\cos\Phi(\theta, z, a)]
$$

$$
\Phi(\theta, z, a) = 2 \sum_j \theta_j (a \cdot g_j \bmod 2)(-1)^{z \cdot g_j}
$$

See [[IQP Expectation]].

## 2. Loss Function

Mixture-of-Z-words form:

$$
\mathrm{MMD}^2(p, q_\theta) = \mathbb{E}_{a \sim P_k}[(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta})^2]
$$

### Kernels (Primary)

| Kernel | Formula | Status |
|---|---|---|
| Gaussian | $e^{-H/\sigma^2}$ (pm-encoding) / $e^{-H/(2\sigma^2)}$ (binary) | **Primary** |
| Laplacian | $e^{-\sqrt{H}/\sigma}$ | Phase 2, stub |
| Polynomial $d$ | $(x \cdot y/n + c)^d$ | Legacy |
| Linear | $x \cdot y / n$ | Legacy baseline |

See [[Kernels MOC]] for the full details.

## 3. Gradient Target

Primary metric:

$$
V(i; \theta_{\text{dist}}, F, K, n) = \mathrm{Var}_{\theta \sim \theta_{\text{dist}}}[\partial_{\theta_i}\mathrm{MMD}^2_K(p, q^F_\theta)]
$$

Scaling hypothesis:

$$
\log V_{\text{agg}}(F, K, n) \sim -\alpha(F, K) n + \beta(F, K) \log n + c(F, K)
$$

See [[Gradient Variance]].

## 4. Circuit Families

Six hypergraph families with controlled connectivity statistics:

| F | Name | Structure | BP expectation |
|---|---|---|---|
| F1 | k-Local Bounded-Degree | $|g_j| \le k$ | No BP for $k \le 3$ |
| F2 | Erdős–Rényi | $p \in \{2/n, \log n/n, 0.5\}$ | No BP at sparse; BP at dense |
| F3 | Lattice (1D/2D) | NN pairs/triples | No BP expected |
| F4 | Dense | $\mathbb{E}[\|g_j\|] = n/2$ | BP expected |
| F5 | Community | 4–8 blocks | Partial BP |
| F6 | Symmetric | Global bitflip symmetric | Unknown |

See [[Families MOC]]. Current primary sweep is narrowed to the SMART four: product state, 2D lattice, sparse ER, complete graph.

## 5. Dataset Plan

- **D1 Product Bernoulli** — primary, isolates structure axes
- **D2 Ising-like synthetic** — secondary, pairwise correlations
- **D3 Binary mixture** — tertiary, multi-modal

See [[Datasets]].

## 6. Initialization Schemes

- **I1 Uniform** — stress test
- **I2 Small-angle** — $\sigma_\theta \in \{0.01, 0.1, 0.3\}$, main trainability hypothesis
- **I3 Data-dependent** — covariance-based

See [[Initialization Schemes]].

## 7. Experiment Grid

| Axis | Values |
|---|---|
| Circuit family | F1–F6 |
| Kernel | Gaussian, Laplacian, Polynomial(d=2), Linear |
| Bandwidth $\sigma$ | 3 values per kernel |
| Init | I1, I2 (×3 σ_θ), I3 |
| $n$ | 16, 24, 32, 48, 64, 96 (+128) |
| Seeds | ≥100 per setting |
| Dataset | D1 primary, D2/D3 comparison |

**Per-setting compute budget:**

- `num_a_samples`: 512 default, up to 2048 for variance stability
- `num_z_samples`: 1024 default, up to 4096 for large $n$

## 8. BP Question Per Cell

For each cell in the 6×4 grid, answer:

```
BP(F_i, K_j)?  →  Var_{θ~I1}[∂_θ MMD²_{K_j}] = Θ(exp(−α·n))?
```

With secondary questions for I2, I3, bandwidth, and polynomial degree thresholds.

## 9. Success Criterion

By **May 15, 2026**, for each (F, K) pair, produce a stated conclusion of the form:

> "Under connectivity family $F_i$ with kernel $K_j$ and initialization $I$, $\mathrm{Var}(\partial \mathcal{L})$ scales [exponentially / polynomially / constant] with $n$; small-angle init [does / does not] change this; shot/noise effects [do / do not] reinstate exponential suppression."

## Related

- [[Proposal]]
- [[SMART Spec]]
- [[Research Questions]]
- [[Families MOC]]
- [[Kernels MOC]]
