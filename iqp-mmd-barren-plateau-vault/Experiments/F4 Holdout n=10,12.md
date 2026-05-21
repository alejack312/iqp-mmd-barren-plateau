---
title: F4 Holdout n=10,12
tags:
  - forge
  - experiments
  - anti-concentration
  - holdout
---
> [!info]
> This is the first post-freeze holdout for F4. The predicate text stayed frozen in [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md), the label source was regenerated from a new extended-`n` sweep, and none of these `n in {10, 12}` rows were part of the training-set fit documented in [[F4 Joint Predicate First Attempt]].

# F4 Holdout n=10,12

> [!tip] In plain English
> First real test of the F4 rule. We locked the rule down and ran it on circuits we hadn't looked at when designing it: 10 and 12 qubits, bigger than anything the rule was built from. Every single one classified correctly.
>
> That's the first real evidence the rule isn't just memorizing the data it came from. There was one hiccup where the logic checker timed out on certain circuits, but that turned out to be a pipeline issue, not a rule issue. See the unblock history below.

## Data Provenance

All inputs are local project artifacts.

- Label source: `results/scaling_ac_holdout_n/results.jsonl`
- Regeneration command: `python -m iqp_bp.cli run-scaling configs/experiments/scaling_ac_holdout_n.yaml`
- Holdout config: [`../../configs/experiments/scaling_ac_holdout_n.yaml`](../../configs/experiments/scaling_ac_holdout_n.yaml)
- Local file hash: `SHA256 bde0d83cfe03ff30ee12c7f2fb76815d25454f78a7c0ef6e87015bae87049a6f`
- Row count: `80`
- Label distribution: `10 PlateauAbsent / 70 PlateauObserved`
- Families represented: `bounded_degree`, `complete_graph`, `erdos_renyi`, `product_state`
- Initializations represented: `uniform`, `small_angle`
- Frozen predicate snapshot: [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md)

This holdout extends the same family/init grid used during training to `n = 10` and `n = 12`, with 5 seeds per bucket.

## Run Configuration

- Main config: [`../../configs/experiments/forge_f4_holdout_n.yaml`](../../configs/experiments/forge_f4_holdout_n.yaml)
- Ablation config: [`../../configs/experiments/forge_f4_holdout_n_ablation.yaml`](../../configs/experiments/forge_f4_holdout_n_ablation.yaml)
- Forge mode: `plateau_agreement`
- Query templates: `plateau_manifests_predicate` and `plateau_manifests_predicate_no_escape`
- Output dirs: `results/forge_f4_holdout_n/` and `results/forge_f4_holdout_n_ablation/`
- Holdout timeout: `600s`

Guardrails were rechecked before and after the run:

- `python scripts/verify_frozen_predicate.py`
- `python -m pytest tests/test_run_forge.py -q`

## Unblock History

- On `2026-04-21` the first attempt at this holdout timed out on all 20 `complete_graph` rows.
- Root cause: `overlaps_consistent` inside the label-free predicate templates was encoding `O(m^2)` pairwise clauses on a field the inst block had already pinned, which blew up SAT translation at `complete_graph x n in {10, 12}`.
- Fix: removed `overlaps_consistent` from `plateau_manifests_predicate` and `plateau_manifests_predicate_no_escape` only. Training-set fit was re-verified unchanged first: `109/111` agree for the main run, `99/111` agree for the ablation.
- Timeout bumped from `60s` to `600s` as insurance.

## Results

Main holdout run:

- Classified rows: `80 / 80`
- Unknown rows: `0 / 80`
- Confusion counts: `TP = 70, TN = 10, FP = 0, FN = 0`
- Accuracy: `80 / 80 = 100.0%`
- Recall: `70 / 70 = 100.0%`
- Precision: `70 / 70 = 100.0%`
- Accuracy by `n`: `100.0%` at `n=10`, `100.0%` at `n=12`

Ablation holdout run:

- Classified rows: `80 / 80`
- Unknown rows: `0 / 80`
- Confusion counts: `TP = 70, TN = 0, FP = 10, FN = 0`
- Accuracy: `70 / 80 = 87.5%`
- Recall: `70 / 70 = 100.0%`
- Precision: `70 / 80 = 87.5%`
- Accuracy by `n`: `87.5%` at `n=10`, `87.5%` at `n=12`

The main holdout now validates cleanly on the full 80-row sweep. The ablation cleanly isolates the structure clause again: all `10` errors are false positives on `complete_graph x uniform`, exactly where the `not complete_graph_like` escape hatch should matter.

Generated artifacts:

- `results/forge_f4_holdout_n/figures/f4_holdout_n_confusion.png`
- `results/forge_f4_holdout_n/breakdown_by_init_n_family.csv`
- `results/forge_f4_holdout_n_ablation/figures/f4_holdout_n_ablation_confusion.png`
- `results/forge_f4_holdout_n_ablation/breakdown_by_init_n_family.csv`

## Breakdown

Main run:

| Family | Init | n | Total | Agree | Disagree | Timeout | Agreement rate |
|---|---|---:|---:|---:|---:|---:|---:|
| bounded_degree | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| bounded_degree | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| bounded_degree | uniform | 10 | 5 | 5 | 0 | 0 | 1.000 |
| bounded_degree | uniform | 12 | 5 | 5 | 0 | 0 | 1.000 |
| complete_graph | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| complete_graph | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| complete_graph | uniform | 10 | 5 | 5 | 0 | 0 | 1.000 |
| complete_graph | uniform | 12 | 5 | 5 | 0 | 0 | 1.000 |
| erdos_renyi | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| erdos_renyi | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| erdos_renyi | uniform | 10 | 5 | 5 | 0 | 0 | 1.000 |
| erdos_renyi | uniform | 12 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | uniform | 10 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | uniform | 12 | 5 | 5 | 0 | 0 | 1.000 |

Ablation run:

| Family | Init | n | Total | Agree | Disagree | Timeout | Agreement rate |
|---|---|---:|---:|---:|---:|---:|---:|
| bounded_degree | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| bounded_degree | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| bounded_degree | uniform | 10 | 5 | 5 | 0 | 0 | 1.000 |
| bounded_degree | uniform | 12 | 5 | 5 | 0 | 0 | 1.000 |
| complete_graph | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| complete_graph | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| complete_graph | uniform | 10 | 5 | 0 | 5 | 0 | 0.000 |
| complete_graph | uniform | 12 | 5 | 0 | 5 | 0 | 0.000 |
| erdos_renyi | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| erdos_renyi | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| erdos_renyi | uniform | 10 | 5 | 5 | 0 | 0 | 1.000 |
| erdos_renyi | uniform | 12 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | small_angle | 10 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | small_angle | 12 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | uniform | 10 | 5 | 5 | 0 | 0 | 1.000 |
| product_state | uniform | 12 | 5 | 5 | 0 | 0 | 1.000 |

The load-bearing structure story is now visible instead of hidden behind timeouts:

- Main run: `complete_graph x uniform x n in {10, 12}` is `10 / 10` agree.
- Ablation run: `complete_graph x uniform x n in {10, 12}` is `0 / 10` agree and `10 / 10` disagree.

## Pass/Fail Against Plan Gate

This holdout passes the plan gate.

1. Zero `timeout` / `unknown` rows across both holdout runs. Passed: both runs classified all `80` rows.
2. Main accuracy `>= 95%` at `n=10` and `>= 90%` at `n=12`. Passed: main accuracy is `100.0%` at both `n=10` and `n=12`.
3. Main precision `>= 95%` overall. Passed: main precision is `100.0%`.
4. `complete_graph x uniform x n in {10, 12}` at `100%` agree in the main run. Passed: the main run is `10 / 10` agree on those rows.

The extended-`n` holdout is now unblocked and validated. The next scientific step is the unseen-family holdout.

## Interpretation

The extended-`n` holdout now supports a stronger statement than the training fit alone:

- The frozen F4 predicate generalizes cleanly to `n=10` and `n=12` on the same family/init grid used during design.
- The structure clause remains load-bearing on holdout, not just on the training sweep. Removing the `not complete_graph_like` escape hatch produces exactly the expected `10` false positives on `complete_graph x uniform`.
- The timeout issue was an artifact of the scoring query, not a failure of the predicate itself.

## Reproducing

From the repo root:

```bash
python scripts/verify_frozen_predicate.py
python -m pytest tests/test_run_forge.py -q
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_manifests.yaml
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_manifests_ablation.yaml
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_holdout_n.yaml
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_holdout_n_ablation.yaml
python scripts/f4_confusion_figure.py --results results/forge_f4_holdout_n/results.jsonl --out results/forge_f4_holdout_n/figures/f4_holdout_n_confusion.png --breakdown-csv results/forge_f4_holdout_n/breakdown_by_init_n_family.csv
python scripts/f4_confusion_figure.py --results results/forge_f4_holdout_n_ablation/results.jsonl --out results/forge_f4_holdout_n_ablation/figures/f4_holdout_n_ablation_confusion.png --breakdown-csv results/forge_f4_holdout_n_ablation/breakdown_by_init_n_family.csv
python scripts/verify_frozen_predicate.py
```

## Slide Copy

After removing a redundant `overlaps_consistent` conjunct from the label-free scoring query, the first post-freeze F4 holdout on extended `n in {10, 12}` now classifies all `80` rows with zero timeouts. The main predicate achieves `80/80` agreement, including `10/10` agreement on the guardrail buckets `complete_graph x uniform x n in {10,12}`, while the no-escape ablation drops to `70/80` agreement by producing exactly `10` false positives on those same rows. That means the frozen F4 predicate generalizes to extended `n`, and the structure clause is still pulling real weight on holdout.

## Related

- [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md)
- [[F4 Joint Predicate First Attempt]]
- [`../../configs/experiments/scaling_ac_holdout_n.yaml`](../../configs/experiments/scaling_ac_holdout_n.yaml)
- [`../../configs/experiments/forge_f4_holdout_n.yaml`](../../configs/experiments/forge_f4_holdout_n.yaml)
- [`../../configs/experiments/forge_f4_holdout_n_ablation.yaml`](../../configs/experiments/forge_f4_holdout_n_ablation.yaml)
- [plan 2026-04-21](../../.claude/plans/twinkly-bouncing-beaver.md)
