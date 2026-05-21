---
title: Figure Index
tags:
  - report
  - figures
  - artifacts
status: draft
---

# Figure Index

This is the figure checklist for [[Final Findings - IQP MMD Barren Plateaus]]. It points to existing figures rather than duplicating them.

## Forge Figures

| Figure | Path | Use |
|---|---|---|
| F4 extended-n confusion | `results/forge_f4_holdout_n/figures/f4_holdout_n_confusion.png` | Shows 80/80 extended-n holdout. |
| F4 extended-n ablation confusion | `results/forge_f4_holdout_n_ablation/figures/f4_holdout_n_ablation_confusion.png` | Shows complete-graph uniform false positives without escape hatch. |
| F4 unseen-family confusion | `results/forge_f4_holdout_families/figures/f4_holdout_families_confusion.png` | Shows 88/92 unseen-family holdout. |
| F4 unseen-family ablation confusion | `results/forge_f4_holdout_families_ablation/figures/f4_holdout_families_ablation_confusion.png` | Shows escape hatch is silent on unseen families. |

## Anti-Concentration and Marginal Figures

| Figure | Path | Use |
|---|---|---|
| AC/marginals summary | `iqp-mmd-barren-plateau-vault/Anti-Concentration/ac_and_marginals_summary.png` | General learned AC and marginal summary. |
| Grid5000 AC/marginals summary | `iqp-mmd-barren-plateau-vault/Anti-Concentration/grid5000_ac_marginals_summary_2026-05-03.png` | Presentation-ready paper-fidelity summary. |
| Pauli scale headline | `results/pauli_scale/headline_scaled_second_moment.png` | Estimator validation and scale-up headline. |

## Generated Tables

| Table | Path | Use |
|---|---|---|
| Evidence ledger | `results/final_report/evidence_ledger.csv` | Provenance table. |
| Forge summary | `results/final_report/forge_summary.csv` | Confusion metrics. |
| Candidate rules | `results/final_report/candidate_rule_scores.csv` | Pattern-mining comparison. |
| Uniform n=4 thresholds | `results/final_report/uniform_n4_feature_thresholds.csv` | F4.1 negative result. |
| AC/marginal summary | `results/final_report/ac_marginal_summary.csv` | Report AC table. |
| Qiskit summary | `results/final_report/qiskit_summary.json` | Qiskit report-smoke headline. |

## Missing or Optional

- A final thesis figure combining F4 rule scores and AC target-vs-learned gaps would be useful, but the current implementation stops at reproducible tables.
- A full Qiskit Q3 figure should wait for the large `configs/experiments/qiskit_validation.yaml` sweep, not the report smoke run.
