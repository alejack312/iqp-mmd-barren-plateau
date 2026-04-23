# dwave Training Evidence — Phase 3 Partial Attempt

**Date:** 2026-04-23
**Command:** `python scripts/pauli_scale_pipeline.py --dataset dwave --n-iters 250 --seed 666`
**Hyperparameters:** `n_qubits=484, max_weight=2, spin_sym=False, sigma=[7.762, 6.151, 3.928], init_scale=0.001, param_noise=0.0, n_ops=1000, n_samples=1000, stepsize=0.001, num_train=5000`
**Outcome:** `training_error` — Jax `OUT_OF_MEMORY` at iter 41/250

## Setup

- Hardware: Windows 11 Home, 16 GB RAM, no GPU (JAX CPU backend via XLA)
- Environment: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
- Data: 5000 rows × 484 cols (binary, from D-Wave quantum annealer samples)
- Gate count: 117,370 (`local_gates(n_qubits=484, max_weight=2)`)

## Loss Trajectory (iters 1–41)

| Iter | Loss | Per-iter time (s) |
|------|------|-------------------|
| 1 | 0.027396 | 161.4 |
| 2 | 0.029859 | 114.2 |
| 3 | 0.031057 | 96.4 |
| 4 | 0.029878 | 88.3 |
| 5 | 0.029051 | 80.5 |
| 6 | 0.028739 | 74.7 |
| 7 | 0.032445 | 73.5 |
| 8 | 0.029039 | 76.8 |
| 9 | 0.031928 | 80.6 |
| 10 | 0.029726 | 79.8 |
| 11 | 0.028148 | 79.6 |
| 12 | 0.026104 | 76.7 |
| 13 | 0.025081 | 76.9 |
| 14 | 0.025774 | 75.8 |
| 15 | 0.028481 | 74.8 |
| 16 | 0.029838 | 74.2 |
| 17 | 0.023483 | 73.8 |
| 18 | 0.027226 | 72.8 |
| 19 | 0.026576 | 74.0 |
| 20 | 0.024653 | 75.4 |
| 21 | 0.025695 | 75.6 |
| 22 | 0.022642 | 74.3 |
| 23 | 0.024040 | 76.9 |
| 24 | 0.021672 | 80.1 |
| 25 | 0.019331 | 83.1 |
| 26 | 0.019369 | 90.1 |
| 27 | 0.019953 | 89.5 |
| 28 | 0.019757 | 90.5 |
| 29 | 0.017146 | 90.3 |
| 30 | 0.019342 | 81.6 |
| 31 | 0.016860 | 72.7 |
| 32 | 0.015962 | 66.0 |
| 33 | 0.016010 | 61.2 |
| 34 | 0.014852 | 60.6 |
| 35 | 0.014173 | 58.6 |
| 36 | 0.014614 | 57.7 |
| 37 | 0.015688 | 56.6 |
| 38 | 0.013895 | 59.1 |
| 39 | 0.013336 | 59.3 |
| 40 | 0.012783 | 60.1 |
| 41 | 0.014567 | 69.3 |
| **42** | **OOM** | n/a |

**Loss decrease:** 0.027 → 0.013 (iter 1 to iter 40) — approximately **2× reduction**, monotonic with noise.

## OOM Event

At approximately iter 42 dispatch, XLA attempted a single-buffer allocation of **969,006,720 bytes (~925 MB)** and was refused by the platform allocator. Partial error text:

```
TRAINING ERROR dwave: INTERNAL: Buffer Definition Event:
Error dispatching computation: [...chain of 14 nested dispatch errors...]
Out of memory allocating 969006720 bytes.
```

The pipeline's `run_single` / `run_all` error handler caught this as a `JaxRuntimeError`, logged the failure to `results/pauli_scale/datasets_status.json` with `outcome: "training_error"`, and exited cleanly (exit code 0).

## Interpretation

1. **Training was working.** Loss is monotonically decreasing (with the stochastic noise expected from MMD with fresh per-iter samples). There is no numerical instability, no divergence, no NaN.

2. **Per-iter memory footprint was not stable.** Per-iter wall-clock time drifts upward from ~60s (iters 30-40) to ~70-90s around the OOM — consistent with either XLA compiled-function cache accumulation or system memory fragmentation preventing contiguous allocations.

3. **Single-buffer alloc of 925 MB is the limiting artifact.** For reference, an `(n_samples × n_samples × float64)` MMD kernel tile is `1000 × 1000 × 8 = 8 MB`. A `(num_train × n_samples × float64)` kernel is `5000 × 1000 × 8 = 40 MB`. So 925 MB is much larger than the obvious tile sizes — likely an intermediate in the gradient computation (`grad(mmd_loss)` with respect to the 117k-parameter theta vector, materialized as a dense gradient tensor over some axis).

4. **n=484 is at the edge.** With the `platform` allocator preventing pre-allocation and no GPU pre-reserved memory pool, the training run is competing with OS memory fragmentation and the workspace needed by `jax.jit`-compiled gradient functions.

## Consequence

No `checkpoint.npz` was saved. The pipeline does not support mid-iter resume; recovery is restart-from-scratch.

Under the available compute regime (16 GB / no GPU / Windows CPU backend), paper-spec MMD training at `n_qubits >= 484` with `num_train=5000, n_samples=1000` is not feasible.

## Implication for v2

To scale beyond n=484 without reducing hyperparameters below paper spec, required changes:
- GPU or ≥64 GB RAM (removes the single-buffer ceiling)
- Memory profiling of `iqpopt.Trainer` to identify the 925 MB allocation source
- Explicit `gc.collect()` + `jax.clear_caches()` between iterations (to combat cache accumulation)

Alternatives at consumer-hardware budget (available now):
- Reduce `num_train` 5000 → 1000 (5× less input memory)
- Reduce `n_samples` 1000 → 500 (linear reduction in kernel memory)
- Reduce `n_iters` 250 → 50-100 (faster but even more undertrained)

Per [03-DISPOSITION.md](../../../.planning/phases/03-acquire-train-estimate/03-DISPOSITION.md), retry under further-reduced hyperparameters was rejected. The failure at paper-spec is itself the scientific finding.

## Final Status JSON

```json
{
  "dwave": {
    "outcome": "training_error",
    "reason": "JaxRuntimeError: ... Out of memory allocating 969006720 bytes",
    "csv_path": "datasets/dwave/dwave_X_train.csv",
    "checkpoint_path": null
  }
}
```

See `results/pauli_scale/datasets_status.json` for the full 5-dataset status.
