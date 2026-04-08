---
title: Mixture Module
tags:
  - code
  - iqp_bp
  - mmd
---

# `iqp_bp.mmd.mixture`

The **data-side** parity expectation path. Small but important: this is where $\langle Z_a \rangle_p$ is computed from dataset samples.

**File:** [`src/iqp_bp/mmd/mixture.py`](../src/iqp_bp/mmd/mixture.py)

## Functions

### `dataset_expectation(data, a)`

Single-observable version:

$$
\langle Z_a \rangle_p \approx \frac{1}{N} \sum_{x \in \text{data}} (-1)^{a \cdot x}
$$

```python
def dataset_expectation(data, a):
    parities = (data @ a) % 2
    signs = 1.0 - 2.0 * parities.astype(np.float64)
    return float(signs.mean())
```

### `dataset_expectations_batch(data, a_batch)`

Vectorized over a batch of observables — the hot path for the scaling runner:

```python
def dataset_expectations_batch(data, a_batch):
    parities = (data @ a_batch.T) % 2  # (N, B)
    signs = 1.0 - 2.0 * parities.astype(np.float64)
    return signs.mean(axis=0)  # (B,)
```

## The Parity Sign Bug

> [!warning] Cast parities to float64
> A subtle bug existed here before the Gaussian normalization work: `1 - 2 * parities` on `uint8` input **underflows** because `uint8(0) - uint8(2)` wraps around to 254 instead of yielding -1. The fix is to cast to `float64` first.
>
> This is why both functions explicitly do `parities.astype(np.float64)` before the sign computation. See [[Implementation Choices#The parity bug that surfaced while doing this]].

The bug became visible only when the Gaussian kernel tests started probing the pipeline directly. Before that, the incorrect signs were masked by other sources of variance.

## Cost

$O(N B n)$ for a batched call — a single large matmul plus a cheap mod and sign.

## Open Work

- **D4.1** — add cached parity statistics and structured target-data helpers for the Ising and binary-mixture datasets, so expensive parity computations are reused across the grid.

## Callers

- [[MMD Loss Module]] — `mmd2()`
- [[Gradients Module]] — `grad_mmd2_analytic()`
- [[Scaling Runner]] — used indirectly via `_make_theta` for the `data_dependent` init

## Related

- [[MMD Loss]]
- [[Parity Algebra]]
- [[Datasets]]
