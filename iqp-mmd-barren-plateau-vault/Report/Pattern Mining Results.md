---
title: Pattern Mining Results
tags:
  - report
  - pattern-mining
  - forge
  - anti-concentration
status: generated-summary
---

# Pattern Mining Results

This note summarizes the extra deep-mining pass requested for the closing report. The pass reads existing scaling rows and checkpoints, extracts structural features from `G`, and scores candidate plateau rules against the empirical AC labels. It does not edit the frozen Forge predicate.

Generated artifacts:

- `results/final_report/pattern_mining_summary.json`
- `results/final_report/scaling_feature_rows.csv`
- `results/final_report/candidate_rule_scores.csv`
- `results/final_report/uniform_n4_feature_thresholds.csv`

## Dataset

Rows mined: `283`

| Split | Rows | Source |
|---|---:|---|
| training | 111 | `results/scaling_ac_diverse/results.jsonl` |
| holdout_n | 80 | `results/scaling_ac_holdout_n/results.jsonl` |
| holdout_families | 92 | `results/scaling_ac_holdout_families/results.jsonl` |

Labels:

| Label | Count |
|---|---:|
| PlateauObserved | 238 |
| PlateauAbsent | 45 |

## Candidate Rules

| Rule | Rows | Accuracy | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| F4 current | 283 | 97.9% | 100.0% | 97.5% | 0 | 6 |
| Init+n only, no escape | 283 | 90.8% | 92.1% | 97.5% | 20 | 6 |
| Threshold `n >= 4`, no escape | 283 | 84.1% | 84.1% | 100.0% | 45 | 0 |
| Threshold `n >= 4`, complete-graph escape | 283 | 92.9% | 92.2% | 100.0% | 20 | 0 |
| Structure-only dense-or-overlap | 283 | 47.3% | 86.2% | 44.5% | 17 | 132 |

Reading:

- F4 current is still the best balanced rule.
- The complete-graph escape hatch is responsible for the difference between `0` false positives and `20` false positives.
- Lowering the uniform branch to `n >= 4` catches every observed plateau but loses too much precision.
- The simple structure-only rule is much worse than the joint init/n/structure rule.

## Known F4 Misses

All misses are false negatives. They are concentrated in `uniform x n=4`:

- Training split: `erdos_renyi x uniform x n=4`, 2 rows.
- Unseen-family holdout: `symmetric x uniform x n=4`, 4 rows.
- Extended-n holdout: no misses.

This confirms the existing Forge note: F4.1 should investigate the small-uniform corner rather than reworking the complete-graph escape hatch.

## Uniform n=4 Threshold Probe

The mining pass tested one-feature thresholds on the `uniform x n=4` subset. The best threshold by accuracy was:

| Feature | Operator | Threshold | Rows | Accuracy | Precision | Recall | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mean_overlap | `>=` | 0.8167 | 31 | 74.2% | 42.9% | 100.0% | 8 | 0 |

This is not good enough to become F4.1. It has full recall but too many false positives. The useful conclusion is negative: the remaining `n=4` failure mode is not closed by a simple scalar structural threshold.

## Structural Feature Set

Each checkpoint-backed row now has:

- `edge_density`
- `mean_row_weight`
- `max_row_weight`
- `mean_qubit_degree`
- `max_qubit_degree`
- `mean_overlap`
- `max_overlap`
- `complete_graph_like`
- `pairwise_disjoint`

These features are written to `results/final_report/scaling_feature_rows.csv` for future analysis.

## Next F4.1 Direction

If F4.1 is pursued, use a new predicate version and keep the current frozen F4 as baseline. The next plausible probes are:

- Small-n uniform corner only, not a global threshold change.
- Family-geometry-specific clause for symmetric/Erdos-Renyi-like overlap patterns.
- A held-out scoring protocol before updating any headline claim.

Do not edit `forge/models/hypergraph.frg` in-place for a final report claim without re-freezing and re-running the holdouts.
