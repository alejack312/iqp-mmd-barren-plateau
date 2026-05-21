---
title: F4 Joint Predicate First Attempt
tags:
  - forge
  - experiments
  - anti-concentration
  - training-fit
---
> [!warning]
> The current F4 numbers are hypothesis-generation only, not validation. The predicate in [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md) was designed from the same `results/scaling_ac_diverse/results.jsonl` sweep it is being scored on, and the audit that triggered this rewrite called that confirmation-bias risk out explicitly. Treat every metric below as training-set fit until the planned holdout runs land.

# F4 Joint Predicate First Attempt

> [!info] Purpose
> Freeze the exact F4 predicate text, record the current training-sweep scorecard, and document the epistemic status before any new runs change the rule or the labels underneath it.

> [!tip] In plain English
> Second attempt at the rule, and the one that actually works. Instead of "is the shape bad?", we ask three things together: did we start at small angles, is the circuit big enough, and is it the one specific shape we already know survives. It agrees with reality on 109 of 111 circuits.
>
> Here's the catch, and the reason this note is tagged "training-fit" not "validation": we designed the rule by looking at those same 111 circuits. So 109 out of 111 really just means the rule fits the data we used to write it. Whether it generalizes needs fresh data. That's what the two holdout notes cover.

## Data Provenance

All inputs are local project artifacts.

- Label source: `results/scaling_ac_diverse/results.jsonl`
- Regeneration command: `python -m iqp_bp.cli run-scaling configs/experiments/scaling_ac_diverse.yaml`
- Local file hash at freeze time: `SHA256 dc7ccff35422c1ab71c642f1968b74a02846854f21ec384a350751e0fb8444ad`
- Row count: `111`
- Label distribution: `23 PlateauAbsent / 88 PlateauObserved`
- Frozen predicate snapshot: [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md)

The hypergraph-side rule itself still lives in [`forge/models/hypergraph.frg`](../../forge/models/hypergraph.frg), but the frozen snapshot above is the committed reference for "F4 as scored on this sweep."

## Run Configuration

- Main config: [`configs/experiments/forge_f4_manifests.yaml`](../../configs/experiments/forge_f4_manifests.yaml)
- Ablation config: [`configs/experiments/forge_f4_manifests_ablation.yaml`](../../configs/experiments/forge_f4_manifests_ablation.yaml)
- Forge mode: `plateau_agreement`
- Query templates now used for scoring: `plateau_manifests_predicate` and `plateau_manifests_predicate_no_escape`
- Output dirs: `results/forge_f4_manifests/` and `results/forge_f4_manifests_ablation/`

These runs now ask Forge the label-free sibling questions "predicate true?" and "predicate false?" and derive agreement in Python, so the `test expect` block no longer mentions `plateau_observed`.

## Results

Current training-set fit on the 111-row sweep:

- F4 main training-set fit accuracy: `109 / 111 = 98.2%`
- F4 main training-set fit recall: `86 / 88 = 97.7%`
- F4 main training-set fit precision: `86 / 86 = 100.0%`
- F4 main confusion counts: `TP = 86, TN = 23, FP = 0, FN = 2`

- F4 ablation training-set fit accuracy: `99 / 111 = 89.2%`
- F4 ablation training-set fit recall: `86 / 88 = 97.7%`
- F4 ablation training-set fit precision: `86 / 96 = 89.6%`
- F4 ablation confusion counts: `TP = 86, TN = 13, FP = 10, FN = 2`

What changed between the two runs:

- On this training sweep, the structure clause carries exactly 10 rows of weight; whether that survives holdout is an open question.
- All 10 changed rows are false positives introduced by the ablation.
- The two false negatives are shared by both runs, so the ablation delta is entirely a precision story on this sweep.

## Interpretation

The current sweep supports a narrow statement and no stronger one:

- On this training sweep, branching on `SmallAngleInit` and `UniformInit AND #Qubit >= 6` explains most of the observed labels.
- On this training sweep, the `not complete_graph_like` carve-out matters for exactly 10 rows.
- On this training sweep, the rule still misses the same two `erdos_renyi x uniform x n=4` rows that sit below the `#Qubit >= 6` threshold.

That is enough to justify freezing the predicate and planning holdouts. It is not enough to claim the rule has generalized.

Decision point:

F4 has now survived both planned validation passes: the extended-`n` holdout in [[F4 Holdout n=10,12]] reproduced the escape-hatch effect cleanly at `n in {10, 12}` with main `80/80` agree and ablation `70/80`, and the unseen-family holdout in [[F4 Holdout Unseen Families]] reached `88/92` agree (`95.7%`) with zero unresolved rows, all four unseen families above `80%` individual accuracy, and no escape-hatch misfires. The rule now has real generalization evidence across two axes it was not fit on. F4.1 should start from enrichment, not re-derivation: the next useful clause is something that explains the remaining `symmetric x uniform x n=4` misses or broadens the init axis, not a rewrite of the current `complete_graph_like` carve-out.

## Reproducing

From the repo root:

```bash
python -m iqp_bp.cli run-scaling configs/experiments/scaling_ac_diverse.yaml
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_manifests.yaml
python -m iqp_bp.cli run-forge configs/experiments/forge_f4_manifests_ablation.yaml
python scripts/f4_confusion_figure.py
python scripts/f4_confusion_figure.py \
  --results results/forge_f4_manifests_ablation/results.jsonl \
  --out results/forge_f4_manifests_ablation/figures/f4_ablation_confusion.png \
  --breakdown-csv results/forge_f4_manifests_ablation/breakdown_by_init_n_family.csv
```

If the predicate text in `forge/models/hypergraph.frg` changes, re-freeze it before trusting any comparison against the numbers in this note.

## Slide copy

On the current 111-row training sweep, the frozen F4 predicate achieves training-set fit accuracy `109/111 (98.2%)`, training-set fit recall `86/88 (97.7%)`, and training-set fit precision `86/86 (100.0%)`. The ablation that removes `not complete_graph_like` keeps the same training-set fit recall but drops training-set fit accuracy to `99/111 (89.2%)` and training-set fit precision to `86/96 (89.6%)`, which means the structure clause carries 10 rows of weight on this sweep. That is enough to motivate holdout work, but not enough to call the rule validated.

## Related

- [`../../results/forge_f4_manifests/PREDICATE_FROZEN.md`](../../results/forge_f4_manifests/PREDICATE_FROZEN.md)
- [plan 2026-04-20](../../.claude/plans/cuddly-swinging-badger-ultraplan.md)
- [[Forge Runner]]
- [[Scaling Runner]]
- [[Anti-Concentration]]
