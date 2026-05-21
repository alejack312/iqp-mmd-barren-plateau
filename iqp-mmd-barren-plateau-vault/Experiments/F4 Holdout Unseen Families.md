---
title: F4 Holdout Unseen Families
tags:
  - forge
  - experiments
  - anti-concentration
  - holdout
---
> [!info]
> This is the second post-freeze holdout for F4. The predicate text is unchanged from [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md), the label source was newly generated from [`../../configs/experiments/scaling_ac_holdout_families.yaml`](../../configs/experiments/scaling_ac_holdout_families.yaml), and none of the evaluated families in this note (`lattice`, `dense`, `community`, `symmetric`) were part of the original training sweep.

# F4 Holdout Unseen Families

> [!tip] In plain English
> Second real test. Same locked-down rule, but the circuit shapes are new ones: lattices, dense meshes, community clusters, symmetric patterns. None were in the data the rule was built from.
>
> 88 out of 92 correct. The 4 misses are all one specific type (small circuits, uniform start) and they're the same boundary issue we already knew about from the training set. What's more interesting: the rule's "structural escape hatch" stayed silent on all these new shapes, which is what you'd want if that escape hatch is really specific to one shape rather than pattern-matching.

## Data Provenance

All inputs are local project artifacts.

- Label source: `results/scaling_ac_holdout_families/results.jsonl`
- Regeneration command: `python -m iqp_bp.cli run-scaling configs/experiments/scaling_ac_holdout_families.yaml`
- Holdout scaling config: [`../../configs/experiments/scaling_ac_holdout_families.yaml`](../../configs/experiments/scaling_ac_holdout_families.yaml)
- Local file hash: `SHA256 c518239b1a15e04ac7c1235c9b3750436c660ffcd7f51b9f42b2e667a2030f73`
- Row count: `92`
- Label distribution: `12 PlateauAbsent / 80 PlateauObserved`
- Families represented: `community`, `dense`, `lattice`, `symmetric`
- Initializations represented: `uniform`, `small_angle`
- Requested n list: `4`, `6`, `8`
- Realized n list: `4`, `6`, `8`
- Excluded cells: `lattice x {uniform, small_angle} x n in {6, 8}` because the registered 2D lattice family requires perfect-square `n`

The exclusion is a family-definition constraint, not a predicate failure. `src/iqp_bp/hypergraph/families.py` only supports 2D lattice on square grids, so `n=6` and `n=8` are not valid 2D lattice settings.

## Run Configuration

- Main config: [`../../configs/experiments/forge_f4_holdout_families.yaml`](../../configs/experiments/forge_f4_holdout_families.yaml)
- Ablation config: [`../../configs/experiments/forge_f4_holdout_families_ablation.yaml`](../../configs/experiments/forge_f4_holdout_families_ablation.yaml)
- Forge mode: `plateau_agreement`
- Query templates: `plateau_manifests_predicate` and `plateau_manifests_predicate_no_escape`
- Output dirs: `results/forge_f4_holdout_families/` and `results/forge_f4_holdout_families_ablation/`
- Holdout timeout: `600s`

Guardrails rechecked before and after the run:

- `python scripts/verify_frozen_predicate.py`
- `python -m pytest tests/test_run_forge.py -q`

## Results

Main run:

- Confusion counts: `TP = 76, TN = 12, FP = 0, FN = 4`
- Accuracy: `88 / 92 = 95.7%`
- Recall: `76 / 80 = 95.0%`
- Precision: `76 / 76 = 100.0%`
- Unresolved rows: `0`

Ablation run:

- Confusion counts: `TP = 76, TN = 12, FP = 0, FN = 4`
- Accuracy: `88 / 92 = 95.7%`
- Recall: `76 / 80 = 95.0%`
- Precision: `76 / 76 = 100.0%`
- Unresolved rows: `0`

The key result is not the absolute score alone. It is that main and ablation are identical on this holdout, which means the `not complete_graph_like` escape hatch never activated on any unseen-family row.

Generated artifacts:

- `results/forge_f4_holdout_families/figures/f4_holdout_families_confusion.png`
- `results/forge_f4_holdout_families/breakdown_by_init_n_family.csv`
- `results/forge_f4_holdout_families_ablation/figures/f4_holdout_families_ablation_confusion.png`
- `results/forge_f4_holdout_families_ablation/breakdown_by_init_n_family.csv`

## Per-Family Breakdown

Main run:

| Family | Init | n | Total | Agree | Disagree | Pred True | Pred False |
|---|---|---:|---:|---:|---:|---:|---:|
| community | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| community | small_angle | 6 | 5 | 5 | 0 | 5 | 0 |
| community | small_angle | 8 | 5 | 5 | 0 | 5 | 0 |
| community | uniform | 4 | 4 | 4 | 0 | 0 | 4 |
| community | uniform | 6 | 5 | 5 | 0 | 5 | 0 |
| community | uniform | 8 | 5 | 5 | 0 | 5 | 0 |
| dense | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| dense | small_angle | 6 | 5 | 5 | 0 | 5 | 0 |
| dense | small_angle | 8 | 5 | 5 | 0 | 5 | 0 |
| dense | uniform | 4 | 4 | 4 | 0 | 0 | 4 |
| dense | uniform | 6 | 5 | 5 | 0 | 5 | 0 |
| dense | uniform | 8 | 5 | 5 | 0 | 5 | 0 |
| lattice | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| lattice | uniform | 4 | 4 | 4 | 0 | 0 | 4 |
| symmetric | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| symmetric | small_angle | 6 | 5 | 5 | 0 | 5 | 0 |
| symmetric | small_angle | 8 | 5 | 5 | 0 | 5 | 0 |
| symmetric | uniform | 4 | 4 | 0 | 4 | 0 | 4 |
| symmetric | uniform | 6 | 5 | 5 | 0 | 5 | 0 |
| symmetric | uniform | 8 | 5 | 5 | 0 | 5 | 0 |

Ablation run:

| Family | Init | n | Total | Agree | Disagree | Pred True | Pred False |
|---|---|---:|---:|---:|---:|---:|---:|
| community | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| community | small_angle | 6 | 5 | 5 | 0 | 5 | 0 |
| community | small_angle | 8 | 5 | 5 | 0 | 5 | 0 |
| community | uniform | 4 | 4 | 4 | 0 | 0 | 4 |
| community | uniform | 6 | 5 | 5 | 0 | 5 | 0 |
| community | uniform | 8 | 5 | 5 | 0 | 5 | 0 |
| dense | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| dense | small_angle | 6 | 5 | 5 | 0 | 5 | 0 |
| dense | small_angle | 8 | 5 | 5 | 0 | 5 | 0 |
| dense | uniform | 4 | 4 | 4 | 0 | 0 | 4 |
| dense | uniform | 6 | 5 | 5 | 0 | 5 | 0 |
| dense | uniform | 8 | 5 | 5 | 0 | 5 | 0 |
| lattice | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| lattice | uniform | 4 | 4 | 4 | 0 | 0 | 4 |
| symmetric | small_angle | 4 | 4 | 4 | 0 | 4 | 0 |
| symmetric | small_angle | 6 | 5 | 5 | 0 | 5 | 0 |
| symmetric | small_angle | 8 | 5 | 5 | 0 | 5 | 0 |
| symmetric | uniform | 4 | 4 | 0 | 4 | 0 | 4 |
| symmetric | uniform | 6 | 5 | 5 | 0 | 5 | 0 |
| symmetric | uniform | 8 | 5 | 5 | 0 | 5 | 0 |

Family-level main accuracies:

- `community`: `28 / 28 = 100.0%`
- `dense`: `28 / 28 = 100.0%`
- `lattice`: `8 / 8 = 100.0%`
- `symmetric`: `24 / 28 = 85.7%`

The only failing cell is `symmetric x uniform x n=4`, where all four rows are false negatives. That miss is not an escape-hatch story: at `n=4`, the main predicate is false because the `#Qubit >= 6` branch is not available.

## Escape-Hatch Activity

This section uses the main run as the operative signal. On this holdout, an escape-hatch firing would show up as a `predicted = False` row under `uniform` init with `n >= 6`.

- `community`: `0 / 10` eligible rows fired the escape hatch.
- `dense`: `0 / 10` eligible rows fired the escape hatch.
- `lattice`: `0 / 0` eligible rows fired the escape hatch because the only realized lattice rows are at `n=4`.
- `symmetric`: `0 / 10` eligible rows fired the escape hatch.

Observed-label cross-check:

- No unseen-family row had `predicted = False` due to `complete_graph_like`.
- Therefore there is no unseen-family counterexample where the escape hatch catches more than intended.
- The ablation is identical for the same reason: the structure clause is silent on this holdout.

## Pass/Fail

This holdout passes the planned unseen-family bar.

1. Overall main accuracy `>= 85%`. Passed: `95.7%` (`88 / 92`).
2. The escape hatch does not misfire silently. Passed: `complete_graph_like` never fired on any unseen-family row, so there is no false negative attributable to the escape hatch.
3. At least two of the four unseen families individually reach `>= 80%` accuracy. Passed: all four families do (`community 100.0%`, `dense 100.0%`, `lattice 100.0%`, `symmetric 85.7%`).

## Interpretation

The unseen-family holdout gives a narrower but still meaningful generalization result than holdout #1:

- The core `(init, n)` logic generalizes well beyond the four training families.
- The structure clause does not overfire on the unseen families. In fact, it is completely silent here.
- The remaining error mode is specific and interpretable: `symmetric x uniform x n=4` produces observed plateaus that the current `#Qubit >= 6` threshold necessarily predicts as absent.

That means F4 survives this holdout as written, but the enrichment direction for F4.1 is now clearer: if we want to catch the last misses, the next clause should target the `uniform, n<6` corner for certain family geometries, not rework the `complete_graph_like` escape hatch.

## Reproducing

From the repo root:

```bash
python scripts/verify_frozen_predicate.py
python -m pytest tests/test_run_scaling.py -q
python -m pytest tests/test_run_forge.py -q
python -m iqp_bp.cli run-scaling configs/experiments/scaling_ac_holdout_families.yaml
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_holdout_families.yaml
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_holdout_families_ablation.yaml
python scripts/f4_confusion_figure.py --results results/forge_f4_holdout_families/results.jsonl --out results/forge_f4_holdout_families/figures/f4_holdout_families_confusion.png --breakdown-csv results/forge_f4_holdout_families/breakdown_by_init_n_family.csv
python scripts/f4_confusion_figure.py --results results/forge_f4_holdout_families_ablation/results.jsonl --out results/forge_f4_holdout_families_ablation/figures/f4_holdout_families_ablation_confusion.png --breakdown-csv results/forge_f4_holdout_families_ablation/breakdown_by_init_n_family.csv
python scripts/verify_frozen_predicate.py
```

## Slide Copy

The second post-freeze F4 holdout tests unseen families rather than unseen `n`, and the frozen predicate still holds up: on `community`, `dense`, `lattice`, and `symmetric` rows it reaches `88/92` agreement (`95.7%`) with zero unresolved rows. The `not complete_graph_like` escape hatch never fires on any unseen-family row, so the main run and the no-escape ablation are identical here; the only misses are four `symmetric x uniform x n=4` false negatives, which point to the current `#Qubit >= 6` threshold rather than to a structure-clause overfit.

## Related

- [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md)
- [[F4 Joint Predicate First Attempt]]
- [[F4 Holdout n=10,12]]
- [`../../configs/experiments/scaling_ac_holdout_families.yaml`](../../configs/experiments/scaling_ac_holdout_families.yaml)
- [`../../configs/experiments/forge_f4_holdout_families.yaml`](../../configs/experiments/forge_f4_holdout_families.yaml)
- [`../../configs/experiments/forge_f4_holdout_families_ablation.yaml`](../../configs/experiments/forge_f4_holdout_families_ablation.yaml)
