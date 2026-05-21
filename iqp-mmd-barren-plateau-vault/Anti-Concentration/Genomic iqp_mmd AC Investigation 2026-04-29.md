---
title: Genomic iqp_mmd AC Investigation
date: 2026-04-29
tags:
  - anti-concentration
  - marginals
  - genomic
  - iqp-mmd
status: draft
---

# Genomic iqp_mmd AC Investigation - 2026-04-29

This note records the exact small-`n` genomic control for the April 23 learned-distribution AC investigation. The native genomic dataset has 805 SNPs, so exact enumeration over `2^805` outcomes is impossible. This run projects the genomic train CSV to 16 deterministic SNP columns and then applies the same exact `q_theta` AC and marginal-TV workflow used in the April 23 `n=16` investigation.

## Setup

- Source CSV: `C:\Users\cuqui\iqp-mmd-barren-plateau\datasets\genomic\805_SNP_1000G_real_train.csv`
- Original shape: `[5008, 805]`
- Projection rule: `top_n_by_bernoulli_variance_then_sorted_by_original_column`
- Selected source columns: `[137, 258, 311, 363, 379, 406, 453, 492, 502, 524, 543, 578, 592, 715, 741, 753]`
- Projected shape: `[5000, 16]`
- IQP settings: `max_weight=2`, `spin_sym=false`, `sigma=[0.6, 1.3]`, `n_iters=1000`
- Checkpoint: `C:\Users\cuqui\iqp-mmd-barren-plateau\results\genomic_iqp_mmd_ac_investigation\checkpoints\genomic_n16_iters1000_seed666.npz`

## Results

![Genomic summary](C:/Users/cuqui/iqp-mmd-barren-plateau/results/genomic_iqp_mmd_ac_investigation/genomic_n16_iters1000_seed666_summary.png)

| Distribution | scaled second moment | beta_hat(1.0) | passes beta >= 0.25 |
|---|---:|---:|:---:|
| learned `q_theta` | 13.8407 | 0.1300 | False |
| empirical target | 43.2642 | 0.0554 | False |

| k | learned mean TV | uniform mean TV | subsets |
|---:|---:|---:|---:|
| 1 | 0.0068 | 0.0023 | 16 |
| 2 | 0.0131 | 0.0788 | 120 |
| 4 | 0.0453 | 0.1849 | 128 |
| 8 | 0.1561 | 0.3362 | 128 |
| 12 | 0.3983 | 0.5347 | 128 |
| 16 | 0.7263 | 0.9446 | 1 |

## Interpretation Boundary

This is an exact-control result for a 16-SNP projection, not a paper-faithful native `genomic-805` training result. A native run should use the Pauli estimator path rather than exact probability vectors, but local Phase 3 evidence already showed `n=484` paper-spec training OOM'd before checkpointing on this 16 GB/no-GPU machine. Therefore any `genomic-805` claim needs either larger hardware or an explicitly reduced-hyperparameter caveat.
