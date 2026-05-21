---
title: Anti-Concentration
tags:
  - theory
  - anti-concentration
  - distribution
---

# Anti-Concentration

> [!summary] Closing report
> The final synthesis is [[Final Findings - IQP MMD Barren Plateaus]]. The closing position is: strict AC is necessary context but not enough for the learned-distribution claim; report target-vs-learned scaled second moment and per-order marginal mismatch alongside any AC pass/fail statement.

> [!tip] Presenting this next week?
> Start at [[Weekly Task - Anti-Concentration]] — a plain-language task brief with the two definitions, the function sketch, a ready-to-run example, and a Q&A section for likely supervisor questions. Then come back here for the deeper technical writeup.

> [!success] 2026-04-23 — the paper's actual learned distributions, tested
> [[iqp_mmd AC Investigation 2026-04-23]] finally answers the supervisor's question on `iqp_mmd`'s own training stack for two paper datasets at `n = 16`. Headline: learned `q_θ` is **not** anti-concentrated; Ising matches target on every marginal order, blobs only at low orders. Includes a [[Codex Audit - spin_sym Export Gap|Codex-discovered bug in the checkpoint bridge]] (silently drops `spin_sym=True`) that we had to work around.
> - Plain-language walkthrough: [[iqp_mmd AC Investigation - Plain English Walkthrough]]
> - ==Why the supervisor was asked about this in the first place==: [[Misattributed Plots Forensic 2026-04-23]] — a teammate attributed pre-existing AC plots to the paper; they were actually on synthetic `product_bernoulli` / `binary_mixture` data.

> [!success] 2026-04-24 — Parseval MC estimator validated at n=16; compute wall documented
> [[Pauli Estimator Scale Up 2026-04-24]] records the Phase 4 result: a Parseval Monte Carlo estimator for scaled_second_moment and per-order marginal mismatch, validated against exact ground truth on the same two checkpoints. 6/8 criteria pass or warn; 2 diagnosed failures (concentrated-distribution σ-underestimation, debias over-correction at k=4). Big-n training blocked at n=484 (dwave OOM); compute wall documented as a methodological finding. See the note for the full validation table, loss trajectory, and v2 hardware requirements.

> [!success] 2026-05-03 - Grid'5000 paper-fidelity run completed, including native genomic-805
> [[Grid5000 Native iqp_mmd AC and Marginals 2026-05-03]] is the presentation-ready writeup for the Nancy run. Headline: learned IQP-MMD distributions are **concentrated, not anti-concentrated**, because the targets are concentrated. Ising matches target concentration and marginals best; blobs is harder; native `genomic-805` completes with estimator-based AC and low-order marginal diagnostics through `k = 8`.

> [!success] 2026-05-08 - AC11 Ghosh-Kim exact and sampled cells completed
> [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]] closes the remaining learned-distribution AC TODO. The exact `n = 9` cell and sampled `n = 20` cell both ran through `run-training`, with per-step learned AC, marginal sidecars, target-vs-learned AC gaps, and target `power_spectrum.json` sidecars.

A distribution-shape property distinct from (and complementary to) [[Gradient Variance|gradient trainability]]. This is the "other" validation axis in the project.

> [!important] Gradients are not distributions
> A regime can look trainable from gradients and still produce a sparse or highly concentrated learned distribution. The anti-concentration check is a **separate** question: *after training, does the learned IQP output distribution stay spread out over $\{0,1\}^n$?*

## The Two Definitions (Equivalent)

From [`docs/technical/anti-concentration.md`](../docs/technical/anti-concentration.md), locked from paper `2512.24801v1`:

### Threshold form

$$
\Pr_x\!\left[p(x) \ge \frac{\alpha}{2^n}\right] \ge \beta
$$

for constants $\alpha, \beta > 0$ independent of $n$, with $x$ drawn uniformly from $\{0,1\}^n$. In words: a constant fraction of bitstrings have probability at least a constant multiple of the uniform baseline $2^{-n}$.

### Second-moment form

$$
2^{2n} \cdot \mathbb{E}_x[p(x)^2] \ge \beta' > 1
$$

For the exactly uniform distribution, this equals 1 exactly. Anti-concentration requires it to exceed 1 by a constant.

## Exact Finite-n Identities

$$
\mathbb{E}_x[p(x)^2] = 2^{-n}\sum_x p(x)^2
\quad \Longrightarrow \quad
2^{2n}\,\mathbb{E}_x[p(x)^2] = 2^n \sum_x p(x)^2
$$

This is the scalar the code computes as `scaled_second_moment`.

## Deterministic Diagnostics

Two primary fields, two different roles:

| Field | Formula | Role |
|---|---|---|
| `scaled_second_moment` | $2^n \sum_x p(x)^2$ | Primary scalar check; matches the paper's second-moment form exactly. ==Use the magnitude, not `passes_second_moment_threshold`== — that flag is [[Codex Audit - spin_sym Export Gap#3. Collateral findings (not spin_sym)\|vacuous]] (threshold hardcoded to 1.0, which every distribution passes). |
| `beta_hat(alpha)` | $2^{-n}\|\{x : p(x) \ge \alpha 2^{-n}\}\|$ | Primary interpretable diagnostic; answers "what fraction of the space has at least uniform-scale weight?" |

Supporting diagnostics:

- `max_probability_scaled = 2^n · max_x p(x)`
- `collision_probability = sum_x p(x)^2`
- `effective_support = 1 / sum_x p(x)^2`

## How The Exact Probability Vector Is Computed

For an IQP circuit $H^{\otimes n} D_\theta H^{\otimes n} |0^n\rangle$:

1. Enumerate all $z \in \{0,1\}^n$ (only feasible for small $n$)
2. Compute the diagonal phase vector $d(z) = e^{-i\phi(z)}$ where $\phi(z) = \sum_j \theta_j (-1)^{z\cdot g_j}$
3. Apply the [[Walsh-Hadamard Transform]]: $a(x) = 2^{-n}\sum_z (-1)^{x\cdot z} d(z)$
4. Square: $p(x) = |a(x)|^2$

This is implemented in `IQPModel.probability_vector_exact(max_qubits=20)` in [`src/iqp_bp/iqp/model.py`](../src/iqp_bp/iqp/model.py).

## Implementation Surface

From [`src/iqp_bp/experiments/run_validation.py`](../src/iqp_bp/experiments/run_validation.py):

- `check_anti_concentration(probabilities, alphas, primary_alpha, beta_min, second_moment_threshold, atol)`
- `evaluate_anti_concentration_from_model(model, provenance, max_qubits, ...)`
- `write_anti_concentration_artifacts(result, output_dir, stem)` — writes JSON summary + CSV + plots
- `save_iqp_checkpoint(model, path, metadata)` — writes `.npz` checkpoint for reuse

## Scaling Runner Integration

The scaling runner has an optional `anti_concentration` config block that appends compact anti-concentration fields to each JSONL record, for `n <= max_n` only (exponential cost).

Appended fields:

- `anti_concentration_available`, `anti_concentration_reason`
- `ac_scaled_second_moment`
- `ac_primary_beta_hat`
- `ac_passes_primary_threshold`, `ac_passes_second_moment_threshold`
- `ac_max_probability_scaled`
- `ac_beta_hat_by_alpha`

See [[Scaling Runner#Anti-Concentration Block]].

## Checkpoint Bridge

The deterministic validation runner accepts `.npz` checkpoints with `G` and `theta`. Two sources:

1. **`run_scaling.py`** can export one checkpoint per small-$n$ setting during a sweep (`anti_concentration.export_checkpoint: true`).
2. **The `iqp_mmd` training pipeline** now also emits `.npz` checkpoints next to its parameter pickles when gate reconstruction succeeds, so trained models can be fed into the `iqp_bp` anti-concentration validator.

See [[Checkpoint Bridge]].

## Why the Split Matters

- **Scaling runner** answers: "does this family/kernel/init combo maintain gradient variance?"
- **Validation runner** answers: "does a specific trained model produce an anti-concentrated output?"

Both are needed. A regime can pass trainability and fail anti-concentration (or vice versa).

## Learned-Distribution Trajectories

As of 2026-04-19 the AC check also runs at every persisted step of a training trajectory. `run-training` (see [[AC7 to AC12 Implementation]]) emits `ac_scaled_second_moment`, `ac_primary_beta_hat`, and the full `ac_beta_hat_by_alpha` dict on every trajectory row, in either `exact` mode (full probability vector) or `sample` mode (bitstring histogram).

> [!warning] Strict AC is the wrong question for this project
> The supervisor's 2026-04-19 ask was framed as "were the learned distributions anti-concentrated?" but what she actually wants is **agreement on high-order marginals**, which is a property *relating* two distributions. Learned distributions in [[References#Paper 2503.02934|Recio-Armengol et al.]] almost certainly pass strict AC — trivially, because training smoothed them *away* from the target's mode structure. See [[Anti-Concentration vs Marginal Agreement]] for the full unpacking.

Completed results: [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]] and [[AC12 Bandwidth Sweep Results 2026-04-21]].

## Related

- [[Grid5000 Native iqp_mmd AC and Marginals 2026-05-03]] - 2026-05-03 paper-fidelity Grid'5000 run, with native `genomic-805`
- [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]] - exact small-n and sampled larger-n learned-distribution AC cells
- [[iqp_mmd AC Investigation 2026-04-23]] — 2026-04-23 end-to-end test of the paper's stack
- [[Pauli Estimator Scale Up 2026-04-24]] — 2026-04-24 Phase 4 writeup: Parseval MC estimator validated at n=16 + dwave compute wall
- [[iqp_mmd AC Investigation - Plain English Walkthrough]] — non-expert explainer
- [[Codex Audit - spin_sym Export Gap]] — the bridge bug Codex caught
- [[Misattributed Plots Forensic 2026-04-23]] — audit trail for the "plots came from the paper" confusion
- [[Walsh-Hadamard Transform]]
- [[IQP Classical Sampling]]
- [[Validation Runner]]
- [[Checkpoint Bridge]]
- [[AC7 to AC12 Implementation]] — learned-distribution AC + marginal evolution pipeline
- [[Design Decisions - AC7 to AC12]] — scope decisions for the supervisor's 2026-04-19 extension
- [[Bandwidth Marginals]] — σ sweep (AC12)
- [[References#Paper 2512.24801]]
- [[Final Findings - IQP MMD Barren Plateaus]]
- [[Evidence Ledger]]
