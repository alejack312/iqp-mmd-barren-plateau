---
title: Checkpoint Bridge
tags:
  - architecture
  - integration
---

# Checkpoint Bridge

The `.npz` checkpoint format that connects the [[iqp_mmd Package|`iqp_mmd`]] training pipeline to the [[Validation Runner|`iqp_bp` validation runner]].

## The Contract

A checkpoint is a NumPy `.npz` file with:

```python
{
    "G": np.ndarray,      # (m, n) uint8
    "theta": np.ndarray,  # (m,) float64
    "metadata": str,      # JSON-encoded provenance dict
}
```

Both loaders (`iqp_bp.experiments.run_validation.load_iqp_checkpoint`) and savers (`save_iqp_checkpoint` in the same module; [[Checkpoint Export|`iqp_mmd.checkpoint_export`]]) agree on this exact shape.

## Why This Bridge Exists

> [!info] Keeping concerns separate
> - **Training** stays in `iqp_mmd` — complex optimizer loops, sampling, evaluation metrics, lots of upstream Xanadu code
> - **Anti-concentration evaluation** stays in `iqp_bp` — deterministic, exact, small-$n$, no training concerns
>
> Without a well-defined bridge, both packages would grow copies of each other's code.

## Two Producers

### 1. `iqp_mmd` training pipeline

When a training run completes, [[Checkpoint Export]] writes a checkpoint next to the parameter pickle — if gate reconstruction for the model type is supported.

### 2. `iqp_bp.run_scaling`

During a scaling sweep, if `anti_concentration.export_checkpoint: true` is set, each small-$n$ setting exports one checkpoint using the first theta seed. See [[Scaling Runner#Anti-Concentration Block]].

## Two Consumers

### 1. `iqp_bp.run_validation`

The primary consumer. Loads a checkpoint, wraps it in [[IQP Model|`IQPModel`]], computes the exact probability vector, and emits JSON/CSV anti-concentration artifacts.

### 2. Manual re-evaluation

You can load a saved checkpoint in a notebook or script and redo any downstream analysis without re-running the expensive training or scaling sweep.

## Practical Workflow

```
┌─────────────────┐    training     ┌────────────┐
│ iqp_mmd CLI     │ ──────────────> │ params.pkl │
└─────────────────┘                 │  + G.npz   │
                                    └─────┬──────┘
                                          │ checkpoint
                                          ▼
                              ┌─────────────────────┐
                              │ iqp_bp.run_validation│
                              └─────────┬───────────┘
                                        │
                                        ▼
                       ┌────────────────────────────────┐
                       │ summary.json + thresholds.csv  │
                       │ + threshold.png + diagnostics.png│
                       └────────────────────────────────┘
```

## Related

- [[iqp_mmd Package]]
- [[Validation Runner]]
- [[Anti-Concentration]]
- [[Scaling Runner]]
- [[Checkpoint Export]]
