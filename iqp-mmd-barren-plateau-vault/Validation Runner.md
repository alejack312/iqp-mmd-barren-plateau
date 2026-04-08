---
title: Validation Runner
tags:
  - code
  - runner
  - validation
  - anti-concentration
---

# `iqp_bp.experiments.run_validation`

Deterministic validation runner for the [[Anti-Concentration]] checker. Takes a model artifact (inline probability vector, bitstring samples, or an `.npz` checkpoint) and emits JSON/CSV anti-concentration evidence.

**File:** [`src/iqp_bp/experiments/run_validation.py`](../src/iqp_bp/experiments/run_validation.py)

## Public API

### Checkers

- `check_anti_concentration(probabilities, alphas, primary_alpha, beta_min, second_moment_threshold, atol)` — the core deterministic check on an exact probability vector
- `evaluate_anti_concentration_from_model(model, provenance, max_qubits, alphas, ...)` — convenience wrapper that calls `model.probability_vector_exact()` first

### Artifact I/O

- `write_anti_concentration_artifacts(result, output_dir, stem)` — writes summary JSON, thresholds CSV, and plots
- `save_iqp_checkpoint(model, path, metadata)` — saves an `.npz` file with `G`, `theta`, and metadata
- `load_iqp_checkpoint(path)` — loads the inverse
- `run(cfg)` — entry point called by the CLI

## Default Constants

```python
DEFAULT_ALPHA_GRID = (0.5, 1.0, 2.0)
DEFAULT_PRIMARY_ALPHA = 1.0
DEFAULT_BETA_MIN = 0.25
DEFAULT_SECOND_MOMENT_THRESHOLD = 1.0
```

## Computed Fields

From [[Anti-Concentration]], the check produces:

```json
{
  "mode": "exact",
  "scaled_second_moment": 2.13,          // 2^n · Σ p²
  "primary_alpha": 1.0,
  "primary_beta_hat": 0.42,              // 2^-n · |{x : p(x) ≥ α/2^n}|
  "passes_primary_threshold": true,
  "passes_second_moment_threshold": true,
  "max_probability_scaled": 3.88,
  "collision_probability": 0.0083,
  "effective_support": 120.5,
  "threshold_checks": [
    {"alpha": 0.5, "beta_hat": 0.74},
    {"alpha": 1.0, "beta_hat": 0.42},
    {"alpha": 2.0, "beta_hat": 0.11}
  ],
  "provenance": {...}
}
```

## Input Contract

The runner accepts three input modes (configured via YAML):

1. **Inline exact probabilities** — a list of floats summing to 1
2. **Bitstring samples** — used to build an empirical histogram (secondary evidence only)
3. **`.npz` checkpoint** — with keys `G`, `theta`, and optional `metadata` — most common path

See [[Checkpoint Bridge]] for how checkpoints are produced.

## Output Contract

For each validation run:

- `{stem}.summary.json` — machine-readable record with all diagnostics + provenance
- `{stem}.thresholds.csv` — one row per alpha for easy plotting
- `{stem}.threshold.png` — threshold curve plot
- `{stem}.diagnostics.png` — scalar diagnostics plot

The JSON/CSV split is deliberate: JSON is the experiment record, CSV is the plotting input.

## Runtime Cap

Exact validation requires enumerating $\{0,1\}^n$, so it's capped by `max_qubits` (default 20). The checker raises `ValueError` if exceeded — callers can catch this and fall back to sample-based diagnostics.

## Related

- [[Anti-Concentration]]
- [[IQP Model]] — `probability_vector_exact`
- [[Walsh-Hadamard Transform]]
- [[Checkpoint Bridge]]
- [[Scaling Runner#Anti-Concentration Block]]
