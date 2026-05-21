---
title: AC11 Ghosh-Kim Learned AC Results 2026-05-08
tags:
  - anti-concentration
  - marginals
  - training
  - ac11
---

# AC11 Ghosh-Kim Learned AC Results 2026-05-08

> [!summary]
> `AC11` is complete. Both the exact small-n cell and sampled larger-n cell were run through `run-training`, and each run now writes trajectory rows, checkpoints, marginal sidecars, target AC summaries, and a target `power_spectrum.json` sidecar.

## Commands

```bash
python -m iqp_bp.cli run-training configs/experiments/ghosh_kim_small_n.yaml
python -m iqp_bp.cli run-training configs/experiments/ghosh_kim_large_n_sampled.yaml
```

## Artifacts

| Cell | Config | Output |
|---|---|---|
| Exact small-n | `configs/experiments/ghosh_kim_small_n.yaml` | `results/ac_ghosh_kim/small_n_exact/` |
| Sampled larger-n | `configs/experiments/ghosh_kim_large_n_sampled.yaml` | `results/ac_ghosh_kim/large_n_sampled/` |

Each output directory has `config.json`, `manifest.json`, `results.jsonl`, and one `runs/<setting>/` folder per bandwidth. Each run folder has:

- `trajectory.jsonl` with learned AC and per-order marginal metrics at persisted steps.
- `checkpoints/step_*.npz` with IQP checkpoints.
- `marginals/step_*.json` with full per-order marginal summaries.
- `power_spectrum.json` with target Fourier power by order.

## Exact Small-n Cell

Setup: `n = 9`, lattice family, binary-mixture target, exact diagnostics, 20 Adam steps, sigma in `{1, 3, 9}`.

| sigma | final MMD2 | target S_p | learned S_q final | S_p - S_q |
|---:|---:|---:|---:|---:|
| 1 | 0.1019228667 | 70.984375 | 20.327775 | 50.656600 |
| 3 | 0.0255036653 | 100.660156 | 4.827589 | 95.832567 |
| 9 | 0.0010016029 | 85.140625 | 12.125220 | 73.015405 |

Reading: the learned model remains much smoother than the empirical target in every exact cell. The MMD loss can become small, especially at sigma 9, while the target's collision structure is still not reproduced.

## Sampled Larger-n Cell

Setup: `n = 20`, sparse Erdos-Renyi family, Ising target, sample diagnostics with 8192 pseudo-shots, 10 Adam steps, sigma in `{1, 3, 9}`.

| sigma | final MMD2 | target S_p | learned S_q final | S_p - S_q |
|---:|---:|---:|---:|---:|
| 1 | 0.0339782085 | 1030.0 | 1287.8125 | -257.8125 |
| 3 | 0.0789940973 | 1030.0 | 408.8750 | 621.1250 |
| 9 | 0.0402775949 | 1024.0 | 12039.34375 | -11015.34375 |

Reading: the sampled large-n cell does not have one uniform direction. At sigma 3, the learned distribution is smoother than the target. At sigma 1 and especially sigma 9, the learned histogram is more concentrated than the target under the sampled diagnostic. These are sample-mode rows, so the trajectory fields should be read with their `distribution_sample_count = 8192` provenance.

## Status Against The Original AC11 Ask

- Exact small-n learned-distribution AC: done.
- Sampled larger-n learned-distribution AC: done.
- Per-step marginal evolution sidecars: done.
- Target-vs-learned AC gap in `results.jsonl`: done.
- Target power spectrum sidecar tying the mismatch back to the data itself: done.

## Relation To The Earlier AC Track

This closes the handoff from [[AC7 to AC12 Implementation]]. The exact small-n AC11 result agrees with [[AC12 Bandwidth Sweep Results 2026-04-21]]: small MMD does not imply learned-vs-target agreement at the distribution-shape level. The sampled larger-n cell adds the caveat that concentration can move in either direction depending on target, bandwidth, and sampling mode, which is why we keep strict AC, target AC, and per-order marginal agreement as separate diagnostics.

## Related

- [[AC7 to AC12 Implementation]]
- [[AC12 Bandwidth Sweep Results 2026-04-21]]
- [[Anti-Concentration vs Marginal Agreement]]
- [[Grid5000 Native iqp_mmd AC and Marginals 2026-05-03]]
