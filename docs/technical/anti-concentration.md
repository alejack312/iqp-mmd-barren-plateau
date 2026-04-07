# Anti-Concentration

This note locks the anti-concentration definitions and the deterministic
implementation targets used in this repo.

## Paper Definitions

Source: `docs/papers/2512.24801v1.pdf`

The paper states two equivalent definitions of anti-concentration.

### Threshold form

\[
\Pr_x \left[p(x) \ge \frac{y}{2^n}\right] \ge \beta
\]

Interpretation:

- `x` is drawn uniformly from `{0,1}^n`
- `2^{-n}` is the uniform baseline probability
- anti-concentration means a constant fraction of bitstrings have probability at
  least a constant multiple of the uniform scale

### Second-moment form

\[
2^{2n} \, \mathbb{E}_x \left[p(x)^2\right] \ge \beta' > 1
\]

Interpretation:

- the expectation is over uniformly random `x in {0,1}^n`
- for the exactly uniform distribution, `p(x) = 2^{-n}` for all `x`, so
  `2^{2n} E_x[p(x)^2] = 1`
- anti-concentration requires this scaled second moment to exceed the uniform
  value by a constant factor

## Exact Finite-n Identities

For a probability vector `p` over `{0,1}^n`,

\[
\mathbb{E}_x[p(x)^2] = 2^{-n} \sum_x p(x)^2
\]

Therefore,

\[
2^{2n} \, \mathbb{E}_x[p(x)^2] = 2^n \sum_x p(x)^2
\]

This gives the exact deterministic scalar used in the code.

## Repo Decision

For deterministic validation, the repo uses both equivalent forms, but with
different roles.

### Primary scalar check

Define

\[
\texttt{scaled\_second\_moment} = 2^n \sum_x p(x)^2
\]

This is exactly equal to the paper's scaled second-moment quantity

\[
2^{2n} \, \mathbb{E}_x[p(x)^2]
\]

and is the primary scalar check because it is exact, compact, and easy to
compute once the full probability vector is available.

### Primary interpretable diagnostic

Define, for `alpha > 0`,

\[
\hat{\beta}(\alpha) = 2^{-n} \left| \left\{ x : p(x) \ge \alpha 2^{-n} \right\} \right|
\]

This is the finite-`n` empirical threshold statistic corresponding to the
paper's threshold form. It answers:

"What fraction of bitstrings have probability at least `alpha` times the
uniform baseline?"

This is the primary interpretable diagnostic because it directly exposes how
much of the output space carries at least uniform-scale weight.

## Why We Are Implementing This

We are implementing this checker because the current barren-plateau pipeline
answers a gradient question:

- how does gradient variance scale with `n`?

but your new task asks a distribution-shape question:

- after training, does the learned IQP output distribution remain
  anti-concentrated?

Those are not the same question. A regime can look trainable from gradients and
still produce a sparse or highly concentrated learned distribution.

For this reason, the anti-concentration check belongs on the deterministic
validation side:

- small `n`: use the exact probability vector and compute the diagnostics
  without Monte Carlo ambiguity
- larger `n`: later use sample histograms only as secondary evidence

The split between the two reported quantities is deliberate:

- `scaled_second_moment` is the compact scalar that matches the paper's
  second-moment form exactly
- `beta_hat(alpha)` is the human-readable diagnostic that tells us how much of
  the space still carries at least uniform-scale weight

## Deterministic Implementation Target

Given an exact probability vector `p` for a trained IQP circuit at small `n`,
the deterministic checker should report at minimum:

- `scaled_second_moment = 2^n * sum_x p(x)^2`
- `beta_hat(alpha) = 2^{-n} * |{x : p(x) >= alpha 2^{-n}}|`

Recommended supporting diagnostics:

- `max_probability_scaled = 2^n * max_x p(x)`
- `collision_probability = sum_x p(x)^2`
- `effective_support = 1 / sum_x p(x)^2`

## How The Exact Probability Vector Is Computed

This repo already had an exact path for observables such as `<Z_a>`, but the
anti-concentration question needs the full output distribution itself. That is
why `AC2` adds exact small-`n` probability-vector extraction on the
deterministic side.

For an IQP circuit of the form

\[
H^{\otimes n} \, D_\theta \, H^{\otimes n} |0^n\rangle
\]

the diagonal block contributes a phase

\[
\phi(z) = \sum_j \theta_j (-1)^{z \cdot g_j}
\]

to each computational-basis string `z`, where `g_j` is generator row `j` of the
binary matrix `G`.

The exact extraction path does the following:

1. Enumerate all basis strings `z in {0,1}^n`.
2. Compute the diagonal phase vector

   \[
   d(z) = e^{-i \phi(z)}.
   \]

3. Apply the Walsh-Hadamard transform

   \[
   a(x) = 2^{-n} \sum_z (-1)^{x \cdot z} d(z)
   \]

   to obtain the exact output amplitudes.
4. Convert amplitudes into Born probabilities

   \[
   p(x) = |a(x)|^2.
   \]

This is exact for the current small-`n` model and is preferable to sample
histograms when the goal is to make a deterministic statement about
anti-concentration. Sample-based checks remain useful later for larger `n`, but
they are secondary evidence, not the primary validation object.

## Validation Runner Contract

The deterministic validation runner is responsible for turning a concrete model
artifact into serialized anti-concentration evidence.

Implemented inputs:

- exact probability vector on disk or inline in config
- generated bitstrings, which are converted into an empirical histogram
- deterministic `.npz` IQP checkpoints containing `G` and `theta`
- a small config-built IQP model for smoke tests and local validation

Implemented outputs:

- one summary JSON file with scalar diagnostics, threshold checks, and
  provenance
- one threshold CSV file with one row per `alpha`

This split is deliberate: the JSON is the machine-readable experiment record,
while the CSV is the easiest input for plotting threshold curves in notebooks or
 figure scripts.

## Scaling Runner Integration

The scaling runner now has an optional `anti_concentration` config block that
appends small-`n` exact anti-concentration summary fields directly into each
JSONL record produced by `run_scaling.py`.

Why this exists:

- the standalone validation runner answers the question for one model artifact
- the scaling runner is where families, inits, kernels, and sizes are compared
- writing compact anti-concentration summaries into the scaling JSONL lets us
  study trainability and output-distribution shape in one table

Recorded scaling fields:

- `anti_concentration_available`
- `anti_concentration_reason`
- `ac_scaled_second_moment`
- `ac_primary_beta_hat`
- `ac_passes_primary_threshold`
- `ac_passes_second_moment_threshold`
- `ac_max_probability_scaled`
- `ac_beta_hat_by_alpha`

Implementation choice:

- exact anti-concentration is exponential in `n`
- therefore `run_scaling.py` computes these fields only when `n <= max_n`
- for each `{family, kernel, init, n}` setting, the check is run once using the
  first theta seed, then that compact summary is copied onto each JSONL row for
  the setting
- rows above the cap are marked unavailable instead of attempting an exact
  computation

## Checkpoint Bridge

The deterministic anti-concentration runner accepts `.npz` checkpoints with
`G` and `theta`, but the repo previously had no built-in way to create those
files from an experiment path. That bridge now exists.

Implemented bridge:

- `save_iqp_checkpoint(model, path, metadata=...)` writes a deterministic
  checkpoint in the exact format that `load_iqp_checkpoint(...)` expects
- `run_scaling.py` can optionally export one checkpoint per small-`n` setting
  from the same first theta seed used for the anti-concentration summary
- `run_validation.py` can then validate that exported checkpoint directly

Practical workflow:

1. Run a small deterministic scaling sweep with
   `anti_concentration.export_checkpoint: true`.
2. Take the emitted `.npz` path from the scaling JSONL row.
3. Point `validation.checkpoint_path` at that `.npz`.
4. Run `run-validation` to produce the exact JSON and CSV anti-concentration
   artifacts for that saved model.

Copied-stack handoff:

- the older `iqp_mmd` training pipeline now also exports deterministic `.npz`
  checkpoints next to its parameter pickle files when gate reconstruction
  succeeds
- those checkpoints use the same `{G, theta, metadata}` structure expected by
  `iqp_bp.experiments.run_validation`
- this keeps training concerns on the `iqp_mmd` side and anti-concentration
  evaluation on the `iqp_bp` side

## Scope

This note locks the definitions only. Threshold choices such as a default
`alpha` grid or a pass/fail cutoff `beta_min` should be decided separately in
the validation implementation.
