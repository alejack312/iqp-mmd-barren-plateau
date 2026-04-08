---
title: Anti-Concentration
tags:
  - theory
  - anti-concentration
  - distribution
---

# Anti-Concentration

> [!tip] Presenting this next week?
> Start at [[Weekly Task - Anti-Concentration]] — a plain-language task brief with the two definitions, the function sketch, a ready-to-run example, and a Q&A section for likely supervisor questions. Then come back here for the deeper technical writeup.

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
| `scaled_second_moment` | $2^n \sum_x p(x)^2$ | Primary scalar check; matches the paper's second-moment form exactly |
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

## Related

- [[Walsh-Hadamard Transform]]
- [[IQP Classical Sampling]]
- [[Validation Runner]]
- [[Checkpoint Bridge]]
- [[References#Paper 2512.24801]]
