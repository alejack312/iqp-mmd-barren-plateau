---
title: Bandwidth vs Marginal Matching (AC12)
aliases:
  - Bandwidth Marginals
  - Sigma Sweep
tags:
  - anti-concentration
  - bandwidth
  - marginals
  - rudolph
date: 2026-04-19
---

# Bandwidth vs Marginal Matching

> [!abstract] What this note is
> The artifact contract and interpretation rules for the `AC12` σ-sweep. Mirrors [`docs/technical/bandwidth-marginals.md`](../docs/technical/bandwidth-marginals.md) and points at the Obsidian-side context. The sweep is **configured but not yet run**; this page deliberately claims no empirical results until it is.

## Why It Exists

MMD² with a Gaussian kernel weights Walsh modes of order $k$ by $\tau^k$ where $\tau(\sigma) = \tanh(1/(4\sigma^2))$. From [[References#Paper 2305.02881|Rudolph 2305.02881]]:

- **Small $\sigma$** ⇒ $\tau \to 1$ ⇒ all orders visible ⇒ high-order marginals in scope ⇒ but barren plateaus threaten trainability
- **Large $\sigma$** ⇒ $\tau \to 0$ ⇒ only low orders weighted ⇒ loss blind to high-order correlations but trainable

`AC12` tests this empirically: train at several σ values, then compare per-order marginal mismatch to the theoretical $\tau^k$ decay.

## Sweep Contract

Config: [`configs/experiments/bandwidth_marginal_sweep.yaml`](../configs/experiments/bandwidth_marginal_sweep.yaml). Four bandwidths `{1, 3, 9, 27}`, 20 Adam steps, $n=9$ ZZ lattice, binary-mixture target, exact-small-$n$ loss and diagnostics.

Output directory: `results/bandwidth_marginal_sweep/`. Layout matches [[AC7 to AC12 Implementation#3. How to Run It|the standard training layout]]:

```
results/bandwidth_marginal_sweep/
├── config.json
├── manifest.json
├── results.jsonl
└── runs/
    └── lattice__n9__gaussian__uniform__binary_mixture__sigma<σ>/
        ├── trajectory.jsonl
        ├── checkpoints/step_XXXX.npz
        └── marginals/step_XXXX.json
```

Each `marginals/step_XXXX.json` carries per-order `tau_power`, `order_weight`, `mean_fourier_squared_error`, `mean_tv`, `weighted_mmd2_contribution`, etc. See [[AC7 to AC12 Implementation#7. Field Reference (Trajectory Row Schema)|the field reference]].

## Interpretation Rules

These are instructions for reading the artifacts when they land — not claims about what they'll show.

- For each σ, plot `mean_fourier_squared_error` or `mean_tv` at step $N$ as a function of order $k$, and overlay `tau_power(σ)` on the same axis.
- If the curves flatten at small σ (all orders roughly equal mismatch), that's the barren-plateau regime: the kernel sees everything but trains nothing.
- If the curves decay steeply at large σ (low orders small, high orders large), that's the low-pass regime: the model matches low marginals but the high-order structure of the target is invisible.
- The sweet spot should sit near $\sigma = \Theta(\sqrt n)$ per [[References#Paper 2305.02881|Rudolph]]; at $n=9$ that's $\sigma \approx 3$.

## Related

- [[AC7 to AC12 Implementation]] — implementation review and FAQ
- [[Design Decisions - AC7 to AC12]] — why `AC12` is scoped this way
- [[Anti-Concentration]] — the AC story `AC11` answers
- [[Gaussian Convention]] — the locked τ definition
- [[Kernel Spectral Decomposition]] — the Walsh basis identity
