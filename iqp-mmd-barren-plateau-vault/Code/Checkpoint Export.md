---
title: Checkpoint Export
tags:
  - code
  - iqp_mmd
  - bridge
---

# `iqp_mmd.checkpoint_export`

The bridge module that emits `.npz` checkpoints from the `iqp_mmd` training pipeline in the exact format the `iqp_bp` [[Validation Runner]] expects.

**File:** [`src/iqp_mmd/checkpoint_export.py`](../src/iqp_mmd/checkpoint_export.py)

## Purpose

After a successful `iqp_mmd` training run, the pipeline has:

- A set of learned parameters (`params.pkl`) in the upstream format
- A structured description of the IQP gate set

But [[Anti-Concentration]] validation in `iqp_bp` needs a canonical `{G, theta, metadata}` layout. This module handles the translation.

## Output Format

`.npz` file with:

- `G` — shape `(m, n)` uint8, the generator matrix
- `theta` — shape `(m,)` float64, the parameter vector
- `metadata` — JSON string with provenance (dataset, training config, model type, source file paths, etc.)

## When It Fires

During `iqp_mmd` training, when gate reconstruction succeeds. If the upstream model type doesn't expose a clean `G` matrix (e.g., RBM or DeepEBM), no checkpoint is written.

## How It's Consumed

Two consumers in `iqp_bp`:

1. **`run_validation.load_iqp_checkpoint(path)`** — loads the `.npz` for deterministic validation
2. **The scaling runner's anti-concentration block** — can optionally export its own checkpoints using `save_iqp_checkpoint(model, path, metadata)` for replay

## Related

- [[iqp_mmd Package]]
- [[Checkpoint Bridge]]
- [[Validation Runner]]
- [[Anti-Concentration]]
