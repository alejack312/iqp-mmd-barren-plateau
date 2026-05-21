---
title: F3 Plateau Agreement Validation
tags:
  - forge
  - experiments
  - anti-concentration
  - validation
---

# F3 Plateau Agreement — First Validation Run

The first validation of the F3 Forge pipeline against a labeled scaling dataset that actually exercises both anti-concentration (AC) outcomes — PlateauAbsent *and* PlateauObserved. The prior smoke run ([[Forge Runner]]) only had PlateauAbsent rows, so it could not test whether the structural predictor disagrees with observed plateau labels.

> [!info] Purpose
> Produce the first agree/disagree table from the F3 pipeline. Stress-test the structural theory (`bounded_degree ∧ high_overlap`) against real observed plateau labels across a diversified sweep of circuit families and initialization schemes.

> [!tip] In plain English
> First attempt at a plateau-predicting rule. The idea was that a circuit's shape alone, specifically how connected its generators are, might be enough to tell when training will fail.
>
> It wasn't. The rule caught 0 of 88 real failures, which is worse than a flat "nothing ever plateaus" guess. The real lesson: where training starts (the initial parameters) matters a lot more than the circuit's shape. That's what pushed us toward F4.

## Data Provenance — where every input came from

All data used in this run is **generated locally by the project's own runners**. Nothing is downloaded, nothing is hand-authored. Both the label source and the Forge library come from the committed repo.

### 1. Label source: `results/scaling_ac_diverse/results.jsonl`

Produced by running the [[Scaling Runner|`run_scaling`]] CLI against a newly authored config:

- **Config:** [`configs/experiments/scaling_ac_diverse.yaml`](../configs/experiments/scaling_ac_diverse.yaml)
- **Command:** `python -m iqp_bp.cli run-scaling configs/experiments/scaling_ac_diverse.yaml`
- **Sweep axes:**
  - `family ∈ {product_state, complete_graph, erdos_renyi, bounded_degree}` (4 [[Hypergraph Families|connectivity families]])
  - `n ∈ {4, 6, 8}` — bounded by `anti_concentration.max_n = 8` so every row uses the exact AC path
  - `kernel.type = gaussian`, `kernel.bandwidth = 1.0`
  - `init.scheme ∈ {uniform, small_angle}`, with `small_angle.std = 0.01`
  - `num_seeds = 2`, `param_idx` varies per setting
- **Label rule:** `ac_passes_primary_threshold = β̂(α=1.0) ≥ 0.25` (default thresholds from [[Anti-Concentration]]).
- **Raw output:** 111 rows in `results/scaling_ac_diverse/results.jsonl` + one `.npz` checkpoint per setting in `results/scaling_ac_diverse/checkpoints/` (via `anti_concentration.export_checkpoint = true`).

Label distribution over the 111 rows:

| Bucket | Rows |
|---|---|
| `ac_passes_primary_threshold = true` → PlateauAbsent | 23 |
| `ac_passes_primary_threshold = false` → PlateauObserved | 88 |

By `(init, label)`:
- `uniform, pass` = 23
- `uniform, fail` = 32 (larger n starts failing)
- `small_angle, fail` = 56 (every small_angle row fails — tiny parameters keep the distribution concentrated near |0…0⟩)

### 2. Hypergraph G matrices: derived from the same scaling run

Each labeled row in the jsonl carries an `ac_checkpoint_path` pointing to a `.npz` file that holds `G` and `theta` arrays. F3's `label_loader` reads each `.npz` via [`load_iqp_checkpoint`](../src/iqp_bp/experiments/run_validation.py) (see [[Checkpoint Bridge]]) and attaches the `G` matrix to the `LabeledRow`.

### 3. Forge library: `forge/models/hypergraph.frg`

Committed to the repo. No external fetch. F3 concatenates its text on top of every composed search file.

## F3 Run Configuration

- **Config:** [`configs/experiments/forge_plateau_diverse.yaml`](../configs/experiments/forge_plateau_diverse.yaml)
- **Command:** `python -m iqp_bp.cli run-forge configs/experiments/forge_plateau_diverse.yaml`
- **Mode:** `forge.mode: plateau_agreement`
- **Query template:** `plateau_agreement` (the iff-based pinned test from [[Forge Runner]])
- **Thresholds:** `max_weight = 3`, `overlap_threshold = 2` (so the structural predictor = `bounded_degree[3] ∧ high_overlap[2]`)
- **Label source:** `results/scaling_ac_diverse/results.jsonl`
- **Output:** `results/forge_plateau_diverse/results.jsonl` (one row per input row) + per-row `.frg` search files under `runs/` + per-row Forge stdout under `raw/`

Every Forge query pins both the hypergraph structure (F1 inst) and the Experiment atom (kernel, init, plateau_observed) before asking Forge to evaluate the iff.

## Results

111 rows processed end-to-end, 0 errors, total wall time ~147 min (mean 79 s per row).

### Agreement distribution

| bucket | count |
|---|---|
| agree | 19 |
| disagree | 92 |

### Crosstab — `plateau_observed × agreement`

| observed | agreement | count | interpretation |
|---|---|---|---|
| PlateauAbsent | agree | 19 | correct rejection — predictor and AC both say "no plateau" |
| PlateauAbsent | disagree | 4 | **false positive** — predictor fires but AC passes |
| PlateauObserved | disagree | 88 | **false negative** — AC fails but predictor is silent |
| PlateauObserved | agree | 0 | (never hits — predictor never fires on observed plateaus) |

### Structural predictor output at thresholds `(max_weight=3, overlap_threshold=2)`

| `structurally_predicted` | count | comment |
|---|---|---|
| False | 107 | predictor silent |
| True | 4 | fires, always on PlateauAbsent rows (see below) |

All 4 fires are `bounded_degree` family, `n=4`, `uniform` init — the random bounded-degree hypergraph at n=4 occasionally produces two weight-3 generators with overlap > 2 (i.e. same 3-qubit support), which is structurally degenerate but doesn't collapse the AC signal at this small n.

### Confusion matrix (treat `structurally_predicted` as the binary predictor of `plateau_observed == PlateauObserved`)

| | predicted plateau | predicted no plateau |
|---|---|---|
| **observed plateau** | TP = 0 | FN = 88 |
| **observed no plateau** | FP = 4 | TN = 19 |

- Accuracy = 19 / 111 = **17.1%**
- Recall (observed plateaus caught) = 0 / 88 = **0.0%**
- Precision (when predictor fires) = 0 / 4 = **0.0%**
- Naive "always predict no plateau" baseline = 23 / 111 = 20.7% — the current structural theory is **worse than a flat prediction**.

### Breakdown by `(family, init, n)`

Every row within each `(family, init, n)` bucket gets the same agreement verdict (verified: buckets are homogeneous across `param_idx`):

| family | init | n=4 | n=6 | n=8 |
|---|---|---|---|---|
| product_state | uniform | agree (4) | disagree (5) | disagree (5) |
| product_state | small_angle | disagree (4) | disagree (5) | disagree (5) |
| complete_graph | uniform | **agree (5)** | **agree (5)** | **agree (5)** |
| complete_graph | small_angle | disagree (5) | disagree (5) | disagree (5) |
| erdos_renyi | uniform | disagree (2) | disagree (5) | disagree (5) |
| erdos_renyi | small_angle | disagree (3) | disagree (5) | disagree (5) |
| bounded_degree | uniform | disagree (4) | disagree (5) | disagree (5) |
| bounded_degree | small_angle | disagree (4) | disagree (5) | disagree (5) |

Only `complete_graph × uniform` agrees at every `n` we tested — a tight, weight-2-regular structure whose AC signal stays strong under uniform init, and which the predictor (correctly silent) rejects. `product_state × uniform × n=4` also agrees, but AC fails by n=6.

## Interpretation

The headline is stark: **at the configured thresholds `(bounded_degree[3] ∧ high_overlap[2])`, the structural theory is not a plateau predictor** — it has zero recall against observed plateaus and all four of its positive predictions are false positives.

Three things this tells us:

1. **Observed plateaus in this sweep are dominated by the initialization axis, not the structural axis.** Every `small_angle` row (56/56) observes a plateau — the tiny-θ distribution concentrates near |0…0⟩, which is the barren-plateau signature the AC threshold is designed to catch. The structural thresholds we tested say nothing about this axis and miss all 56. Any future version of `plateau_structurally_predicted` that aims to explain plateaus has to reference `Init` (or some init-driven feature), not just `contains` / `overlaps`.
2. **Scaling `n` also matters more than structure at this scale.** Uniform-init rows at `n=6,8` begin observing plateaus (AC fails) for product_state, erdos_renyi, and bounded_degree — again at structure that the predictor considers sub-threshold. Only complete_graph keeps AC strong through `n=8` under uniform init.
3. **Thresholds `(3, 2)` are too loose to be a useful predictor, but lowering them is not obviously better.** The four false positives come from bounded_degree at `n=4` where two weight-3 generators happen to share all 3 qubits — structurally degenerate duplicates rather than barren-plateau-inducing overlap. Dropping `overlap_threshold` to 0 or 1 would fire everywhere, destroying precision in the opposite direction.

What this validates *about the pipeline*, independent of the research finding:

- F3 correctly pins structure, kernel, init, and observed-label atoms into a per-row Forge query, composes library + instance + experiment + bounds + test, and parses the agree/disagree signal from real Forge stdout.
- The `none->none` bug in `_relation_literal` (F1 → product_state rows errored in the first run) was caught end-to-end and fixed in this run; the tests now assert the omit-the-binding idiom directly so that class of bug can't recur silently.
- Per-row wall time is ~80 s at `n≤8`, dominated by Forge/Kodkod startup + Pardinus solve. A dedup pass that collapses rows sharing `(family, n, kernel, init)` before submitting queries would cut this by roughly 5x (the sweep's `param_idx` and seed axes don't change the Forge verdict for this query template, only the plateau label — which is already per-setting, not per-param).

### Open questions that this result forces

- Is the right next predicate `init_consistency[e: Experiment]`, firing when `e.init = SmallAngleInit`? That would capture 56/88 observed plateaus immediately, but it's a trivial restatement of the label axis — not a structural claim. A genuine structural predictor probably needs to condition on `(Init, n, structure)` jointly, which is squarely F4 territory.
- Does lowering thresholds to `(max_weight=2, overlap_threshold=1)` change anything? Quick mental check: `bounded_degree[2]` fails on most families (weights routinely hit 3), so the predictor would flip to "silent everywhere" rather than "silent + 4 FPs". Worth running once as a sensitivity pass before F4.
- Is `complete_graph × uniform` the only regime where the structural theory is *not* wrong, or is this an n=4–8 artifact that disappears at larger n? Extending the sweep to `n=16` (still within the exact-AC regime with `max_n=16`) would answer this directly.

### Known bug, now fixed

First pass of this run produced 28 `error` rows, all `product_state`. Root cause: `_relation_literal` in `src/iqp_bp/forge/export_instances.py` emitted `overlaps = none->none` (and similarly `contains = none->none`) when the pair list was empty. Forge 5.2 rejects both `none->none` and `none` in arity-2 `inst` field bindings (NONE-TOK parse error). The idiom is to omit the field entirely; Forge then defaults it to empty given the `overlaps_consistent` predicate in the query.

Fix: `_relation_literal` now returns `None` for empty input, and both callers in `export_to_forge` skip the `field = ...` line accordingly. The two regression tests in `tests/test_export_instances.py` (`test_emits_omitted_overlaps_binding_for_non_overlapping_case`, `test_empty_generators_omit_arity2_bindings`) assert the omission directly so this class of bug can't silently come back.

## Reproducing this run

From the repo root:
```bash
# 1. Generate labeled scaling dataset (about 2 min)
python -m iqp_bp.cli run-scaling configs/experiments/scaling_ac_diverse.yaml

# 2. Run F3 against it (about 25 min at n ≤ 8, 111 rows × ~12 s/row)
python -m iqp_bp.cli run-forge configs/experiments/forge_plateau_diverse.yaml

# 3. Inspect
cat results/forge_plateau_diverse/results.jsonl

# 4. Regenerate the slide figure
python scripts/f3_confusion_figure.py
```

## Slide copy

Drop-in artifact for the AC presentation (or any Q&A about measurement on real checkpoints). Figure rendered by [`scripts/f3_confusion_figure.py`](../scripts/f3_confusion_figure.py); regenerate by re-running the script.

![[../results/forge_plateau_diverse/figures/f3_confusion.png]]

> Across a 111-row sweep (4 families × 3 n ∈ {4, 6, 8} × 2 init schemes), a Forge-encoded structural predictor `bounded_degree[3] ∧ high_overlap[2]` achieves **0% recall** against the AC-observed plateau signal (`¬ac_passes_primary_threshold`): zero of 88 observed plateaus caught, and all four of its positive predictions are false. The **init axis dominates** — every one of the 56 `small_angle` rows (std = 0.01) fails anti-concentration, while only `complete_graph × uniform` keeps AC strong across all tested n. A structural-only theory of plateau manifestation is not tenable at these thresholds; the next predicate must condition on `Init` jointly with structure.

## Related

- [[Forge Runner]] — runner internals, mode dispatch, query composition
- [[Forge Export]] — F1 structural instance emission
- [[Anti-Concentration]] — the β̂(α) threshold machinery that produces the plateau labels
- [[Scaling Runner]] — the runner that produces the label source
- [[Checkpoint Bridge]] — `.npz` format used to ship `G` between runners
- [[Barren Plateaus]] — the underlying research question this validation supports
