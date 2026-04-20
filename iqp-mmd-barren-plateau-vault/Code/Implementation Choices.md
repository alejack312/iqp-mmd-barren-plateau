---
title: Implementation Choices
tags:
  - planning
  - design
---

# Implementation Choices

Why the code looks the way it does. Condensed from [`docs/technical/implementation-choices.md`](../docs/technical/implementation-choices.md).

## Two Threads of Work

The recent round of changes had two intertwined goals:

1. **Make the SMART family layer real** — replace the old 2D patch sampler with the exact nearest-neighbor grid family; replace the generic sparse row sampler with a proper sparse Erdős–Rényi graph family.
2. **Lock the Gaussian kernel convention** — the repo had drifted into using different meanings of $\sigma$ in different files.

## The Gaussian Convention

Locked to the paper in [`docs/papers/2503.02934v2 (3).pdf`](../docs/papers/2503.02934v2%20(3).pdf):

$$
k(x, y) = \exp\!\left(-\frac{H(x, y)}{2\sigma^2}\right)
$$

with per-bit factor $q = \exp(-1/(2\sigma^2))$ and Walsh decay $\tau = (1 - q)/(1 + q) = \tanh(1/(4\sigma^2))$.

Applies to:

- `gaussian_kernel()`
- `gaussian_spectral_weights()`
- `gaussian_sample_a()`
- `multi_scale_gaussian_kernel()`

See [[Gaussian Convention]].

## Why the Lattice Is Square-Only

The SMART lattice family is the **exact** 2D nearest-neighbor ZZ baseline:

- Qubits on an $L \times L$ square grid
- Open-boundary horizontal + vertical NN edges only
- Every row has weight 2
- `range_` must be 1
- Deterministic for fixed $n$
- $m = 2L(L-1)$

Once this is the contract, $n$ must be a perfect square. There is no honest way to represent $n = 24$ or $n = 48$ as an exact square grid. The code **fails fast** instead of pretending.

See [[ZZ Lattice Family]].

## Why Erdős–Rényi Was Changed This Way

The old family was a generic sparse row sampler, not an actual graph family. The new one:

- Samples an undirected ER graph on $n$ qubits
- Uses one weight-2 generator row per edge
- Interprets `p_edge` as target average degree $c$
- Uses $p = \min(1, c/n)$ so degree is bounded as $n$ grows
- Has intrinsic row count (`m` is ignored)

This keeps all four SMART families pairwise, and the comparison becomes about **connectivity** rather than interaction order.

See [[Erdos-Renyi Family]].

## Why the Runners Now Trust `G.shape[0]`

Since some families have intrinsic row counts, runners can't trust the requested `m`:

```python
G = make_hypergraph(family, n, m, rng, **kwargs)
actual_m = G.shape[0]  # <-- authoritative
```

Affected: `run_scaling.py`, `run_forge.py`.

## Why the Hypothesis Layer Was Narrowed to SMART Families

The earlier Hypothesis layer exercised arbitrary binary matrices and a few generic family placeholders. That's fine for shape regression testing, but not for verifying that the experiments actually run on the SMART families.

The new layer defines `smart_family_config()`, `smart_family_instance()`, and `mmd_instance()` — all scoped to the primary four families.

See [[Hypothesis Strategies]].

## The Invariants Each Family Tests

Each family-specific test catches a particular regression:

- **product_state** — identity; if this drifts, everything built on top suffers
- **lattice** — exact edge set on square grid; distinguishes real 2D NN from patch samplers
- **erdos_renyi** — weight-2 pairwise with bounded expected degree; catches non-graph regressions
- **complete_graph** — every pair exactly once

## Why the Init Layer Changed Too

The init refactor shares the same surface:

- `uniform`
- `small_angle` with exact sweep `{0.01, 0.1, 0.3}`
- `data_dependent` — lightweight warm start based on empirical parity expectations

The SMART plan called for an exact small-angle sweep, not "some small float." The `data_dependent` path exists in both the Hypothesis layer and the scaling runner but is still a lightweight warm start rather than a covariance-driven design.

## The Parity Sign Bug

A bug surfaced during the Gaussian cleanup:

```python
# Broken on uint8 — underflows
signs = 1 - 2 * parities

# Fixed — cast first
signs = 1.0 - 2.0 * parities.astype(np.float64)
```

The bug was silent until the Gaussian tests probed the path directly. See [[Mixture Module#The Parity Sign Bug]].

## Rules for Future Changes

From the technical note:

- If you change the Gaussian formula, change the Walsh decay and the sampler in the same commit.
- If you change a family with intrinsic structure, update the runner logic that derives `m`.
- If you relax a family invariant, say why in the docs and adjust the Hypothesis tests on purpose. Don't just weaken a failing test and move on.
- If a config example stops matching the real family contract, fix the example. Examples are part of the interface now.

## Related

- [[Gaussian Convention]]
- [[ZZ Lattice Family]]
- [[Erdos-Renyi Family]]
- [[Hypothesis Strategies]]
- [[Theory-Implementation Parity]]
