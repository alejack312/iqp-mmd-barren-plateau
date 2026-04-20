# Bandwidth vs Marginal Matching

This note defines the artifact contract for the AC12 bandwidth sweep.

## Goal

For Gaussian MMD, the relevant analytical weight is

`tau(sigma) = tanh(1 / (4 sigma^2))`

and the order-`k` Walsh modes are weighted by `tau(sigma)^k`.

The empirical question is:

- which marginal orders actually get matched during training at each `sigma`
- whether that empirical pattern tracks the theoretical `tau(sigma)^k` decay

## Implemented Pipeline

`run-training` now emits, at every persisted checkpoint:

- a loss value
- anti-concentration diagnostics
- a per-order marginal summary returned by `summarize_by_order(...)`

For Gaussian runs, each order entry includes:

- `mean_tv`
- `mean_chi_square`
- `mean_fourier_squared_error`
- `tau_power`
- `order_weight`
- `weighted_mmd2_contribution`

That last quantity is the important AC12 bridge. It is the contribution that
the current Gaussian kernel would assign to all order-`k` Walsh modes.

## Expected Sweep Artifact Layout

The checked-in sweep config is:

- `configs/experiments/bandwidth_marginal_sweep.yaml`

The runner writes one run directory per resolved `sigma` setting under:

- `results/bandwidth_marginal_sweep/runs/`

Each run contains:

- `trajectory.jsonl`
- `checkpoints/step_*.npz`
- `marginals/step_*.json`

## Interpretation Rules

- small `sigma`: higher orders are visible because `tau` is closer to `1`, but
  optimization may become unstable or plateau-like
- large `sigma`: `tau^k` decays quickly, so the loss is dominated by low-order
  marginals and high-order mismatch can persist

This note intentionally does not claim results yet. It defines how to read the
artifacts once the AC12 sweep is run.
