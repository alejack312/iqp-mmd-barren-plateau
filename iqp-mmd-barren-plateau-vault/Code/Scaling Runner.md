---
title: Scaling Runner
tags:
  - code
  - runner
  - scaling
---

# `iqp_bp.experiments.run_scaling`

The main experiment runner. Sweeps the full (family × kernel × init × n) grid, computes gradient variance for each cell, and writes JSONL output. Also emits anti-concentration summaries for small $n$.

**File:** [`src/iqp_bp/experiments/run_scaling.py`](../src/iqp_bp/experiments/run_scaling.py)
**Walkthrough:** [[How a Scaling Run Works]]

## Entry Point

```python
def run(cfg: dict[str, Any]) -> None
```

Called by [[CLI|`iqp_bp.cli`]] when the subcommand is `run-scaling`.

## Loop Structure

```
for setting in resolve_scaling_settings(cfg):
    G     = make_hypergraph(family, n, m, rng_circuit, **kwargs)
    data  = make_dataset(dataset_cfg, n, seed_data)
    θ_seeds = [_make_theta(...) for _ in range(num_seeds)]
    ac_summary = _summarize_anti_concentration(..., θ_seeds[0])
    for param_idx in range(min(5, actual_m)):
        stats = estimate_gradient_variance(G, data, param_idx, θ_seeds, ...)
        write JSONL: {setting, m, param_idx, **stats, **ac_summary}
```

## Grid Resolution

`resolve_scaling_settings(cfg)` returns the explicit flattened grid. See [[How a Scaling Run Works#The Explicit Grid]] for the expansion rules.

## Helper Functions

| Helper | Role |
|---|---|
| `_as_list(val)` | Wrap scalars as single-element lists |
| `_compute_m(n, formula)` | Evaluate `n_generators` formula (int or string like `"n"`) |
| `_make_G(family, n, m, circuit_cfg, rng, er_p_edge)` | Build G with per-family kwargs |
| `_make_theta(init_scheme, G, data, init_cfg, seed, small_angle_std)` | Build one theta vector |
| `_get_kernel_params(kernel, kernel_cfg, bandwidth)` | Select kernel kwargs |
| `_summarize_anti_concentration(...)` | Compute AC fields for one setting |
| `_bandwidth_values_for_kernel(kernel, kernel_cfg)` | Per-kernel bandwidth sweep |
| `_erdos_renyi_values_for_family(family, circuit_cfg)` | Per-family ER degree sweep |
| `_small_angle_values_for_init(init_scheme, init_cfg)` | Per-init small-angle std sweep |
| `_record_setting_fields(setting)` | Flatten setting dict for JSONL |
| `_setting_identity(setting)` | Setting key for seed derivation |
| `_setting_stem(setting)` | Filename stem for artifacts |
| `_format_scalar(value)` | Make scalar values safe for filenames |
| `_checkpoint_name(family, init_scheme, kernel, n)` | Standard checkpoint filename |

## Anti-Concentration Block

Optional, controlled by `anti_concentration:` in the config. See [[Anti-Concentration]] for the math.

```yaml
anti_concentration:
  enabled: true
  max_n: 16                 # skip above this
  alphas: [0.5, 1.0, 2.0]
  primary_alpha: 1.0
  beta_min: 0.25
  second_moment_threshold: 1.0
  export_checkpoint: true   # optionally save .npz
  checkpoint_dir: checkpoints
  artifact_dir: anti_concentration
```

When a setting has $n \le $ `max_n`, the block:

1. Runs `evaluate_anti_concentration_from_model(model, provenance, ...)` on `IQPModel(G, theta_list[0])`
2. Writes JSON summary + CSV + plots to `{output_dir}/anti_concentration/{stem}.*`
3. Optionally saves a `.npz` checkpoint
4. Copies a compact field set onto each JSONL row for the setting

## Output

- `results.jsonl` — one record per `(setting, param_idx)`
- `anti_concentration/` — per-setting AC artifacts (if enabled)
- `checkpoints/` — per-setting `.npz` checkpoints (if `export_checkpoint: true`)

## Related

- [[How a Scaling Run Works]]
- [[Hypergraph Families]]
- [[Gradients Module]]
- [[Validation Runner]]
- [[Configs]]
- [[RNG]]
