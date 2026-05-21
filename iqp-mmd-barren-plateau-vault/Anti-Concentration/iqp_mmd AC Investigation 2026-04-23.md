---
title: iqp_mmd AC Investigation — Are Recio-Armengol's learned distributions anti-concentrated and do they match targets on high-order marginals?
date: 2026-04-23
tags:
  - anti-concentration
  - marginals
  - iqp-mmd
  - recio-armengol
  - investigation
  - results
  - codex-audit
aliases:
  - iqp_mmd Investigation
  - Does iqp_mmd Pass AC
  - Paper Learned Distributions Test
status: complete
related:
  - "[[Anti-Concentration vs Marginal Agreement]]"
  - "[[AC7 to AC12 Implementation]]"
  - "[[AC12 Bandwidth Sweep Results 2026-04-21]]"
  - "[[Codex Audit - spin_sym Export Gap]]"
  - "[[Anti-Concentration]]"
---

# iqp_mmd AC Investigation — 2026-04-23

> [!abstract] What this note is
> On **2026-04-22** a teammate claimed the Recio-Armengol paper `2503.02934` provides anti-concentration plots for its learned distributions. It does not — what he remembered were plots we generated in our [[AC12 Bandwidth Sweep Results 2026-04-21|`AC11`/`AC12` sandbox]] on a synthetic `binary_mixture` target. The supervisor's real question — *"are the iqp_mmd learned distributions anti-concentrated, and do they coincide with the target on high-order marginals?"* — had never been tested directly on the paper's actual training stack. This note documents the end-to-end investigation that finally answers that question at `n = 16`, including the Codex-discovered export bug that corrupted an intermediate result and how we fixed it.

> [!note] Plot attribution (2026-04-24 addendum)
> "The plots my friend showed came from the **AC6** deliverable (shipped 2026-04-07, the day before the Apr 8 deadline) — specifically the `_diagnostics.png` and `_threshold_curve.png` outputs that the scaling runner emits for every `(family, n, init)` cell. The data is synthetic `product_bernoulli` on untrained random-init IQP circuits (`source=scaling_seed`), used to tag Forge plateau-agreement rows with AC pass/fail — it's not trained-model output and has no relationship to the paper." See [[Misattributed Plots Forensic 2026-04-23]] for full provenance.
>
> *What is synthetic `product_bernoulli` data?* A `product_bernoulli` distribution is the simplest possible distribution over `{0,1}^n`: each of the `n` qubits is an **independent** coin flip with its own bias `p_i`. There are no correlations between qubits whatsoever — the joint probability of any bitstring is just the product of `n` independent Bernoulli probabilities. It has no multi-qubit structure, no clusters, no peaks — it is used as a cheap synthetic stand-in when the experiment only needs *some* target distribution, not a realistic one.
>
> *Feynman translation:* Imagine rolling 16 separate biased coins — each coin has its own probability of landing heads, but the coins don't interact at all. The distribution over all 65 536 possible outcomes of those 16 flips is a `product_bernoulli`. It's the "assume-independence" baseline: trivial to generate, trivial to analyse, and completely unlike any real physical dataset (which always has correlations). The AC6 runs used this as a dummy target purely to populate the Forge label database — not to say anything about what a real trained model does.
>
> *What is an untrained random-init IQP circuit?* The IQP circuit has a set of gates and a parameter vector `θ`. Normally `θ` is found by running an optimiser (Adam, many iterations) to minimise MMD² against the target. In the AC6 scaling runs, **no optimisation happened at all** — `θ` was drawn from a uniform distribution on `[-π, π]` (or a small-angle Gaussian with `std=0.01`) and used as-is. The resulting circuit is just a random IQP with no relationship to any target distribution.
>
> *Feynman translation:* Think of the circuit parameters as the knobs on an enormous mixing desk. Training means spending hours turning the knobs until the music coming out sounds like the target song. An untrained random-init circuit is what you get if you set all the knobs to a random position and immediately press record — the output is essentially noise, tuned to nothing. The AC6 plots describe the anti-concentration behaviour of that random noise across different circuit families and sizes, which is useful for understanding the circuit architecture itself, but says nothing about whether a *trained* model is anti-concentrated.

> [!success] Headline findings
> At `n = 16`, on the two smallest paper datasets, trained via `iqp_mmd`'s own stack:
> - **Anti-concentration:** Neither learned distribution is strictly anti-concentrated. Both fail the `β̂(α = 1.0) ≥ 0.25` threshold; both have scaled second moments ≫ 1 (uniform). `α = 1` uses the uniform probability `1/2ⁿ` as the baseline — so `β̂(1)` is literally "the fraction of bitstrings that carry at least as much mass as a fair coin flip", the canonical choice across the IQP hardness literature. `β ≥ 0.25` comes from Paley–Zygmund arguments on approximate 2-designs: Haar-random circuits satisfy this with `β ≈ 1/e ≈ 0.37`; 0.25 is the conservative floor consistent with the IQP-hardness lemmas in both Recio-Armengol (2503.02934) and Rudolph (2305.02881). Neither value is tuned to this experiment — they are the locked finite-`n` decision rule from `AC1`.
> - **High-order marginals:** For each order `k ∈ {1, …, 16}` we sampled up to 128 random `k`-subsets `S ⊂ [n]`. The **target marginal** `p_S` is obtained by histogramming the `S`-columns of the training samples (empirical distribution, ~3k–5k draws over `2^16` bins). The **learned marginal** `q_S` is obtained by summing the exact probability vector `q_θ` — computed via `IQPModel.probability_vector_exact`, which enumerates all `2^16 = 65 536` bitstring amplitudes exactly — over the `n − k` qubits outside `S`. TV distance `½ Σ_x |q_S(x) − p_S(x)|` is computed per subset and averaged to give `M_k`. On ==Ising== (spin-symmetric target) the learned `q_θ` matches the target closely at **every** order `k`. On ==spin_blobs== (multi-modal target) it matches low orders (`k ≤ 3`) well but drifts fast above.
>
> *Feynman explanation:* Think of each 16-qubit bitstring as a 16-character word, and the distribution as a language — it says how often each word appears. We have two versions of the language: the **real one** (estimated by tallying the training data, like a word-frequency table from a book) and the **model's version** (computed exactly from the IQP circuit, as if the circuit were a perfect grammar rulebook). To check whether they match, we don't compare all `2^16 = 65 536` words at once — that would need far more data than we have. Instead, we pick a random group of `k` character positions and ask: *"looking only at those `k` slots, does the model's language match the real one?"* At `k = 1` this is just checking whether each character appears at the right frequency on its own. At `k = 8` we're checking whether any 8-character combination appears at the right joint frequency — a much stricter test. Total variation distance measures how wrong the model is on that group: 0 means perfect, 1 means completely different. Averaging over 128 randomly chosen groups of size `k` gives `M_k` — a per-order report card. The Ising model gets full marks at every `k`; the Blobs model passes the easy tests (`k ≤ 3`) but fails the harder ones.
>
> **Visuals:** ![[ac_and_marginals_summary.png]]
> Left panel: `M_k` TV curves for Ising (blue) and Blobs (orange), solid = learned `q_θ`, dashed = uniform baseline. Right panel: scaled second moment for learned vs empirical target on log scale, with the AC threshold at `y = 1`.
> - The learned model is always dramatically closer to the target than the uniform distribution is, at every order tested.

---

## 1. Why this investigation exists

### 1.1 The confusion we had to clear up

The teammate's claim had to be disproven because:

- The paper itself never computes **any** anti-concentration diagnostic on `q_θ` — not collision probability, not threshold `β̂(α)`, not Pauli magnitude sums. A keyword sweep of the PDF for `anti-concentr`, `collision`, `second moment`, `heavy output`, `tv distance`, `total variation` returns **zero matches**. See [[Anti-Concentration vs Marginal Agreement#3. Reading Recio-Armengol, Ahmed & Bowles (2503.02934) in This Light]].
- The only anti-concentration plots in this repo (`results/bandwidth_marginal_sweep/figures/ac_trajectory.png`, `m_k_heatmap.png`) were produced by our `AC11`/`AC12` pipeline on a synthetic `binary_mixture` target at `n = 9`, using our own `iqp_bp` training stack — not `iqp_mmd`'s. See [[AC12 Bandwidth Sweep Results 2026-04-21]].
- Nobody in this project had ever fed a trained `iqp_mmd` checkpoint through the AC validator. The machinery existed; it had just never been used end-to-end.

> [!important] The two questions the supervisor is asking
> These are *different* — see [[Anti-Concentration vs Marginal Agreement#1. Two Properties, Not One]].
> 1. **AC (one distribution).** Does `q_θ` spread mass across enough of `{0,1}^n`?
> 2. **Marginal agreement (two distributions).** Does `q_θ`'s marginal on every subset `S ⊂ [n]` match the target's?
> A distribution can pass (1) and fail (2), or vice versa. Writing them up together without keeping them apart will read as confused.

### 1.2 What we already had vs. what we were missing

```mermaid
graph LR
    A[iqp_mmd training\nvia iqpopt] -->|save checkpoint| B[(G, theta).npz]
    B -->|load| C[iqp_bp IQPModel]
    C -->|probability_vector_exact| D[q_theta over 2^n]
    D --> E[check_anti_concentration]
    D --> F[per_order_marginal_mismatch ❗NEW]
    G[training target samples] --> E2[AC on target]
    G --> F
```

- ✅ `iqp_mmd/checkpoint_export.py` — bridges `iqpopt` gates → deterministic `(G, θ)`.
- ✅ `iqp_bp/experiments/run_validation.py` — `check_anti_concentration` on exact probability vectors.
- ✅ `iqp_bp/iqp/model.py` — `IQPModel.probability_vector_exact` for exact `q_θ` at small `n`.
- ❌ No trained `iqp_mmd` checkpoints on disk anywhere.
- ❌ No high-order marginal (`M_k`) diagnostic wired up for `iqp_mmd` outputs.
- ❌ `iqpopt` not installed in the local env.

Plan: install `iqpopt`, write a single script that trains → exports → evaluates, then interpret.

---

## 2. The investigation script

> [!note] File
> `scripts/investigate_iqp_mmd_ac.py` (finalised 2026-04-23). Companion `scripts/rerun_ac_via_iqpopt.py` re-runs the AC + M_k stage directly from `iqpopt.probs()` after the [[Codex Audit - spin_sym Export Gap|Codex-discovered spin_sym bug]].

### 2.1 What the pipeline does in plain English

1. **Make training data.** For `spin_blobs`: pick 20 random "peak" bitstrings, then sample by flipping each bit of a peak iid with prob 5%. For `ising`: run a simple 2D Gibbs sampler on a `4×4` periodic ferromagnetic Ising lattice at `T = 2.5` (above the `T_c ≈ 2.27` critical temperature — paramagnetic regime). Output `N` samples in `{0, 1}^16`.
2. **Train the IQP circuit.** Build `iqpopt.IqpSimulator` with `local_gates(n_qubits=16, max_weight=4)` → 2516 gates. Initialise `θ` from the data (`initialize_from_data`). Run `iqpopt.Trainer` with `mmd_loss_iqp`, two Gaussian bandwidths `σ ∈ {0.6, 1.3}`, `n_iters = 1000`. For Ising pass `spin_sym=True` (paper hyperparameter).
3. **Export a deterministic checkpoint.** Convert the `iqpopt` gate list to a binary generator matrix `G` (one row per gate), save `(G, θ)` plus metadata as a `.npz` via `iqp_mmd.checkpoint_export`.
4. **Load the checkpoint on the `iqp_bp` side.** `load_iqp_checkpoint(path)` → `IQPModel(G, θ)`.
5. **Compute the exact distribution.** `IQPModel.probability_vector_exact(max_qubits=16)` → vector `q_θ` of length `2^16 = 65 536`. Internally this evaluates the diagonal IQP phase `φ(z) = Σ_j θ_j (−1)^(z·g_j)`, then takes the Walsh-Hadamard transform and squares the amplitudes.
6. **Anti-concentration diagnostics on `q_θ`.** `check_anti_concentration(q_θ)` returns collision probability, scaled second moment `2^n Σ q(x)²`, `β̂(α)` at `α ∈ {0.5, 1.0, 2.0}`, plus pass/fail flags.
7. **Empirical target distribution.** `samples_to_probability_vector(X)` → histogram `p_emp` of length `2^16` from the training samples.
8. **Per-order marginal mismatch.** For each `k ∈ {1, …, 16}`, pick up to 128 random `k`-subsets `S ⊂ [n]`, compute:
   - `q_S`: marginal of `q_θ` over `S` (sum out the complement via a reshape-and-sum).
   - `p_S`: marginal of `p_emp` over `S` (histogram the `S`-columns of `X`).
   - TV distance `½ Σ |q_S − p_S|`.
   - Aggregate: mean, max, min across sampled subsets. This is `M_k`.
9. **Uniform baseline.** Repeat step 8 with `q_θ` replaced by the uniform distribution `2^(-n) · 1`. This tells us how much of the low-`k` agreement is trivial.
10. **Save a summary JSON + a headline plot.**

> [!tip] Bit-ordering convention (critical for correctness)
> `probability_vector_exact` uses `_basis_bits_exact(n)` where **qubit 0 is the MSB**: integer index `i` maps to bits `((i >> (n-1-q)) & 1)` for `q = 0, …, n-1`. Both `samples_to_probability_vector` and our `empirical_marginal_from_samples` use the same convention (`bit_weights = 2^(n-1), 2^(n-2), …, 2^0`). An off-by-one here would silently corrupt every `M_k`. ==We verified this matches== — this was one of Codex's Category-A attacks that held up.

### 2.2 Code implementation of the marginal comparison

The marginal pipeline lives in two places: the functions embedded in `scripts/investigate_iqp_mmd_ac.py` (used for this investigation) and the reusable modules in `src/iqp_bp/distributions/marginals.py` + `marginal_metrics.py` (the same logic, extracted into the library for AC7–AC12).

#### Learned-distribution side — `marginal_from_full`

```python
# investigate_iqp_mmd_ac.py:113–123
def marginal_from_full(p: np.ndarray, n: int, subset: tuple[int, ...]) -> np.ndarray:
    shape = (2,) * n
    p_nd = p.reshape(shape)                         # (2,2,...,2) — axis i = qubit i
    complement = tuple(q for q in range(n) if q not in subset)
    return np.sum(p_nd, axis=complement)            # sum out all non-S qubits
```

`p` is the exact probability vector `q_θ` returned by `IQPModel.probability_vector_exact` — a flat array of length `2^n`. Reshaping it to `(2,)*n` turns it into an `n`-dimensional binary tensor where axis `i` indexes qubit `i` (0 = MSB). Summing over the axes *not* in `S` marginalises out all `n − k` qubits outside the subset, leaving a `(2,)*k` tensor whose entries are the exact marginal probabilities `q_S(x)` for each of the `2^k` outcomes on `S`.

The library version (`src/iqp_bp/distributions/marginals.py:exact_marginal`) does the same operation differently — it iterates over all `2^n` basis bitstrings, reads off the `k` bits in `S`, and scatter-accumulates via `np.add.at` — but the result is identical.

> *Feynman explanation:* The circuit gives us a complete probability table — one number for every possible 16-bit string (all 65 536 of them). Think of it as a giant spreadsheet with one row per possible outcome and a column labelled "probability." To find the marginal for just `k` specific qubits, we want to collapse that spreadsheet down to `2^k` rows (one per outcome on those `k` qubits) by adding up the probabilities of all full outcomes that share the same `k`-bit pattern. The code does this by first folding the flat list into a 16-dimensional grid (one dimension per qubit, each of size 2), then calling `np.sum` over all the dimensions we don't care about — exactly like flattening a multi-way pivot table by summing across the unwanted axes.

#### Target-distribution side — `empirical_marginal_from_samples`

```python
# investigate_iqp_mmd_ac.py:126–137
def empirical_marginal_from_samples(X: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    k = len(subset)
    cols = X[:, list(subset)].astype(np.uint8)      # (N, k) — the k columns for S
    bit_weights = 1 << np.arange(k - 1, -1, -1, dtype=np.uint64)
    idx = (cols.astype(np.uint64) @ bit_weights).astype(np.int64)  # encode each row as int in [0, 2^k)
    counts = np.bincount(idx, minlength=2**k).astype(np.float64)
    return (counts / counts.sum()).reshape((2,) * k)
```

`X` is the `(N, n)` binary sample matrix (training data). We select only the `k` columns corresponding to the qubits in `S`, then encode each of the `N` rows as an integer in `[0, 2^k)` by dotting with the MSB-first weight vector `[2^(k-1), …, 1]`. `np.bincount` tallies how many samples landed in each of the `2^k` bins; dividing by `N` gives the empirical marginal `p_S`. The reshape to `(2,)*k` matches the shape of `marginal_from_full`'s output so the two can be compared element-wise.

The library version (`marginals.py:sample_marginal`) is the same algorithm without the final reshape.

> *Feynman explanation:* We don't have a complete probability table for the target — we only have a bag of ~5000 sample bitstrings drawn from it. To estimate the marginal on `k` qubits, we do exactly what a pollster does: ignore everything except the `k` columns we care about, read those `k` bits on each sample as a short binary number, and tally how many times each of the `2^k` possible short numbers appeared. Dividing by the total number of samples turns the tally into a frequency estimate. The more samples we have, the closer this estimate is to the true marginal — but with only ~5000 samples spread across `2^16` possible full outcomes, the *full* distribution is hopelessly sparse; it's only by projecting down to a small `k` that the counts become meaningful.

#### Bit-ordering convention — why both sides agree

Both functions encode the first element of `subset` as the most-significant bit. `marginal_from_full` achieves this by preserving the reshape axis ordering (axis `i` = qubit `i`, MSB-first in `probability_vector_exact`). `empirical_marginal_from_samples` achieves this via `bit_weights = [2^(k-1), …, 1]`, which puts `subset[0]` in the highest-order position. If one side used LSB-first and the other MSB-first, the marginal tables would be permuted and the TV values would be wrong for `k ≥ 2`. This is the convention Codex attacked as a Category-A risk and which we verified empirically (Blobs TV = 0.000000 between `iqp_bp` and `iqpopt` confirmed alignment).

> *Feynman explanation:* Both functions need to agree on what "outcome 3" means for a 2-qubit subset `{q5, q11}`. If one function calls `01` outcome 1 and the other calls it outcome 2, the two marginal tables describe the same distribution but with their rows in a different order — and subtracting them entry-by-entry gives nonsense. The convention is simple: the first qubit in the subset is always the most-significant bit, so `{q5, q11}` with values `(0, 1)` always maps to `0×2 + 1×1 = 1`. Both functions implement this with the same weight vector `[2^(k-1), …, 1]`, which is how we know the row labelling matches.

#### Assembling `M_k` — `per_order_marginal_mismatch`

```python
# investigate_iqp_mmd_ac.py:144–175
def per_order_marginal_mismatch(p_q, X_target, n, max_subsets_per_order=128, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(1, n + 1):
        subsets = list(itertools.combinations(range(n), k))   # all C(n,k) subsets
        if len(subsets) > max_subsets_per_order:
            idx = rng.choice(len(subsets), size=max_subsets_per_order, replace=False)
            subsets = [subsets[i] for i in idx]               # uniform subsample
        tvs = [tv_distance(marginal_from_full(p_q, n, S),
                           empirical_marginal_from_samples(X_target, S))
               for S in subsets]
        out[k] = {"mean_tv": ..., "max_tv": ..., "min_tv": ..., "num_subsets": ...}
```

For each order `k`, all `C(n, k)` subsets are enumerated; if that exceeds 128 a uniform random subsample is drawn (without replacement, from the full combinatorial set rather than iteratively — so every subset has equal probability of being included). For each sampled subset the two marginals are computed via the functions above and their TV distance is recorded. The result `out[k]` is a dict with `mean_tv`, `max_tv`, `min_tv` over however many subsets were used. `M_k` in the summary tables is `out[k]["mean_tv"]`.

The library version (`marginal_metrics.py:summarize_by_order`) does the same thing through the `_marginal_from_source` dispatcher, which automatically routes to `exact_marginal` or `sample_marginal` depending on whether its argument is a 1D probability vector or a 2D sample matrix — so the same function handles both sides with no branching at the call site.

> *Feynman explanation:* For order `k = 4` on 16 qubits there are `C(16, 4) = 1820` different groups of 4 qubits to check — checking all of them is feasible (and we do), but for `k = 8` there are `C(16, 8) = 12870`. Rather than checking every one, we pick 128 at random, like a random audit rather than a full census. Each audited group gets its own TV score. `M_k` is the average TV score across all audited groups — a single number that summarises "on average, how badly does the model fail to match the target when you look at any random group of `k` qubits?" A value near 0 means the model's `k`-qubit statistics are indistinguishable from the target's; a value near 0.5 means they're badly mismatched on average.

### 2.3 Why this uses `iqp_mmd`, not our `iqp_bp` training stack

The whole point was to test **the paper's own trained models** — not re-implement training. So the training step must use `iqpopt` (what the paper uses), and the AC analysis uses our `iqp_bp` validator that we already trust. The checkpoint `.npz` is the bridge. [[Codex Audit - spin_sym Export Gap|The bridge had a silent bug]] that we only caught because Codex went adversarial on it.

---

## 3. Scope and compromises (not paper-faithful)

> [!warning] The results are for a reduced surrogate, not a true paper reproduction
> We deliberately cut compute to get answers in a single afternoon. Several of the choices deviate from `configs/hyperparameters.yaml`:
>
> | Dial | Paper | This investigation | Why |
> |---|---:|---:|---|
> | `n_iters` | 10 000 | 1 000 | Wall-clock; training at `n = 16` takes ≈ 7 min per 1k iters. |
> | blobs `max_weight` | 6 | **4** | `mw = 6` with `n = 16` builds ≈ 14 900 gates; triggered JAX OOM on 16 GB RAM. Dropped to mw = 4 (2516 gates). |
> | blobs `n_ops`, `n_samples` | 1000 / 1000 | **300 / 300** | Memory — see above. |
> | ising dataset source | cached `ising_4_4_T_3_train.csv` | home-made Gibbs at `T = 2.5` | No cached CSV in repo; wrote a local sampler. |
> | blobs dataset source | `qml_benchmarks.RandomSpinBlobs` | inlined generator | Dependency conflict on jax version. |
>
> All other hyperparameters (`σ`, `init_scale`, `param_noise`, `stepsize`, `spin_sym`) match the paper for these two datasets. Scope claims accordingly.

The final-loss numbers confirm the training gap: `ising` converged nicely (0.107 → 2.6 × 10⁻⁴ over 1000 iters), `blobs` is clearly undertrained (1.62 × 10⁻² → 6.86 × 10⁻³, still descending).

---

## 4. First-pass results (later partially corrected)

Artifacts: `results/iqp_mmd_ac_investigation/*_summary.json` (v1, wrong for Ising).

### 4.1 Anti-concentration

| Dataset | scaled 2nd moment `2^n Σ q²` | `β̂(1.0)` | passes `β̂ ≥ 0.25`? |
|---|---:|---:|:---:|
| ==Ising (spin_sym=True)== | 2 942.75 | 0.078 | ❌ |
| ==Blobs (spin_sym=False)== | 43.14 | 0.168 | ❌ |
| uniform reference | 1.0 | 1.0 | ✅ |

### 4.2 Per-order marginal mismatch — first pass

| k | Ising (first pass) | Blobs |
|---:|---:|---:|
| 1  | 0.057 | 0.010 |
| 4  | 0.098 | 0.127 |
| 8  | 0.157 | 0.415 |
| 16 | 0.414 | 0.789 |

> [!tip] Interim conclusion we almost shipped
> "Both distributions fail strict AC; learned `q_θ` matches target well at low `k` but drifts monotonically with order." This is roughly right for Blobs but turns out to be **wrong for Ising** — see [[#5. Codex adversarial audit]] below.

---

## 5. Codex adversarial audit

> [!warning] What Codex found
> We ran `/codex challenge` on the investigation (adversarial mode — "try to break this"). Full transcript in `results/codex_challenge.jsonl`. Summary in [[Codex Audit - spin_sym Export Gap]]. Three ==[P1]== findings:
>
> 1. **`passes_second_moment_threshold` in `iqp_bp` is mathematically vacuous.** Threshold hardcoded to `1.0`, check passes iff `2^n Σ p(x)² ≥ 1`. That inequality holds for *every* distribution (with equality only for uniform), so the flag is always `True`. Never cite it.
> 2. **The investigation is not paper-faithful.** See [[#3. Scope and compromises (not paper-faithful)|§3]] — we already flagged this but Codex sharpened the "the paper's 8_blobs uses mw = 6, 10k iters" language.
> 3. **Checkpoint export might not represent the trained circuit.** `iqpopt` internally distinguishes `self.gates` (grouped structure) from `self.generators` (flat list). `generator_matrix_from_gates` turns each gate into one row. If any builder returns multi-generator gates, or if `spin_sym=True` alters the effective circuit semantics, the exported `(G, θ)` is a *different* model than what was trained.

Findings 1 and 2 we already knew or conceded. Finding 3 was the bombshell.

### 5.1 Verifying Codex's claim 3

Script: `scripts/_verify_checkpoint_faithful.py`. For each checkpoint:
- Load `(G, θ)` via `iqp_bp.load_iqp_checkpoint`, compute `q_bp = probability_vector_exact`.
- Rebuild `iqpopt.IqpSimulator(gates=local_gates(...), spin_sym=<original>)`, compute `q_opt = model.probs(θ)`.
- Compare: `TV(q_bp, q_opt)`, `L∞(q_bp − q_opt)`.

Result:

| Model | `spin_sym` | `TV(q_bp, q_opt)` | `L∞` | Verdict |
|---|:---:|---:|---:|---|
| Blobs | False | **0.000000** | 3.47e-17 | ✅ faithful |
| **Ising** | **True**  | **0.257637** | 6.41e-02 | ❌ **different circuit** |
| Ising (control) | False | **0.000000** | 1.19e-15 | ✅ confirms gap = spin_sym |

> [!danger] The Ising first-pass results were invalid
> `iqp_bp.IQPModel.probability_vector_exact` does the canonical IQP forward pass (`H⊗n  · exp(iφ(Z))  · H⊗n  · |0…0⟩`). It has **no handling for `spin_sym`**. When `iqpopt` is constructed with `spin_sym=True`, it averages the expectation values over both the `|0…0⟩` and `|1…1⟩` initial states — a Z₂-symmetric version of the IQP circuit with a materially different output distribution. Every scaled-second-moment, `β̂`, and `M_k` number I reported for Ising in §4 was computed on the unsymmetrised circuit.

### 5.2 What else Codex flagged (valid, not blocking)

- `β̂ ≥ 0.25` is one arbitrary choice — project spec `SMART-spec.md` uses `0.1` in one example. Say "fails at the chosen finite-n threshold," not "not AC."
- High-k `M_k` is partly confounded by target sampling noise (3k–5k samples vs. `2^16` outcomes). The gap is real but the precise endpoint is not.
- The uniform-baseline comparison (already computed) belongs in the main readout — see §6.

### 5.3 What Codex flagged that did **not** hold

- **Bit ordering:** Codex predicted off-by-one; we already matched `_basis_bits_exact` (MSB = qubit 0) with `samples_to_probability_vector` (same convention). Cross-checked empirically — blobs TV = 0 is evidence both conventions line up.
- **Grouped gates in `local_gates`:** Codex suspected `local_gates(n, max_weight)` might return grouped generators. Empirically verified: every "gate" is `[[support]]` — one support per gate. `generator_matrix_from_gates` flattening is 1-to-1 correct **for `local_gates`**. Would break for custom or `nearest_neighbour_gates`-style builders.

---

## 6. Corrected results (the real answer)

Artifacts: `results/iqp_mmd_ac_investigation/*_summary_CORRECTED.json`, `ac_and_marginals_summary.png`. Fix: compute `q_θ` via `iqpopt.IqpSimulator.probs(θ)` (which respects `spin_sym`), then feed that vector into the unchanged `check_anti_concentration` + `per_order_marginal_mismatch`. No retraining needed; the checkpoint `θ` is the same.

### 6.1 Anti-concentration (corrected)

| Dataset | scaled 2nd moment | `β̂(1.0)` | passes `β̂ ≥ 0.25`? |
|---|---:|---:|:---:|
| Ising `spin_sym=True` | ==3 856.71== | ==0.058== | ❌ |
| Blobs `spin_sym=False` | 43.14 | 0.168 | ❌ |
| empirical target (Ising) | 4 044.87 | 0.022 | ❌ |
| empirical target (Blobs) | 658.29 | 0.013 | ❌ |
| uniform reference | 1.0 | 1.0 | ✅ |

Both learned distributions are strongly concentrated relative to uniform. Ising is *more* concentrated after the fix, not less (3857 vs. 2943 before). Qualitative answer unchanged: **neither is anti-concentrated**. Note that the empirical-target scaled 2nd moments are inflated by finite sampling (a few-thousand samples over `2^16` bins), so they aren't directly comparable to `q_θ` — what matters is the magnitude vs. 1.

### 6.2 Per-order marginal mismatch (corrected)

Mean `TV(q_S, p_target_S)` over up to 128 random `k`-subsets, with uniform baseline:

| k | Ising learned | Ising uniform | Blobs learned | Blobs uniform |
|---:|---:|---:|---:|---:|
| 1  | **0.005** | 0.005 | **0.010** | 0.106 |
| 2  | 0.011 | 0.317 | 0.032 | 0.157 |
| 3  | 0.019 | 0.477 | 0.069 | 0.228 |
| 4  | **0.026** | 0.536 | **0.127** | 0.302 |
| 5  | 0.035 | 0.553 | 0.206 | 0.423 |
| 6  | 0.044 | 0.602 | 0.291 | 0.544 |
| 7  | 0.057 | 0.659 | 0.361 | 0.603 |
| 8  | **0.072** | 0.694 | **0.415** | 0.637 |
| 12 | 0.177 | 0.817 | 0.624 | 0.871 |
| 16 | **0.291** | 0.978 | **0.789** | 0.987 |

> [!success] The real Ising story
> After the `spin_sym` fix, the Ising learned distribution matches the target at **every** order `k` examined. `k = 1` TV is at MC noise floor (0.005). `k ≤ 8` TV stays under 0.075. Even `k = 16` (full joint) TV is 0.29 — roughly 3× closer to the target than uniform is (0.978).

> [!question] The Blobs story
> Learned captures low-`k` structure well (TV < 0.05 for `k ≤ 2`) but drifts quickly: TV > 0.4 by `k = 8`, TV > 0.78 at `k = 16`. Still beats uniform (0.987), but only modestly at high `k`. Plausible causes:
> 1. Training is clearly under-converged (loss still descending at 1000 iters).
> 2. `max_weight = 4` instead of 6 — the model can't directly express weight-5,6 correlations that exist in a 20-peak mixture.
> 3. Blobs is sharply multi-modal — exactly the target profile the Rudolph tension predicts is hardest for low-`σ` MMD. See [[Anti-Concentration vs Marginal Agreement#2. What Rudolph Actually Predicts]].

### 6.3 What changed from the first pass

| Metric | Ising v1 (wrong) | Ising v2 (correct) | Δ |
|---|---:|---:|---|
| scaled 2nd moment | 2 942.75 | 3 856.71 | +31% |
| `β̂(1.0)` | 0.078 | 0.058 | –26% |
| `M_1` (mean) | 0.057 | **0.005** | ≈ MC floor now |
| `M_8` (mean) | 0.157 | 0.072 | –54% |
| `M_16` (mean) | 0.414 | 0.291 | –30% |

Blobs numbers are unchanged (the export was faithful when `spin_sym=False`).

---

## 7. Answering the two questions

> [!important] Q1: Are the iqp_mmd learned distributions anti-concentrated?
> ==**No**== — at the chosen `β̂ ≥ 0.25` threshold, at `α = 1.0`, on these two datasets at `n = 16`. Both learned distributions are strongly concentrated relative to uniform (scaled 2nd moment ∈ {43, 3857} vs. 1). The result matches the intuition that a model fit to a concentrated target via low-bandwidth MMD has no pressure to spread mass out. ==Caveat==: strict BMS anti-concentration is an asymptotic-in-`n` statement; `β̂(1.0) < 0.25` at `n = 16` is "fails at this threshold," not "violates the asymptotic criterion."

> [!important] Q2: Do they coincide with the target on high-order marginals?
> ==**Ising: yes, at every order examined.**== Learned vs. target TV stays at MC noise floor for `k = 1`, under 0.08 through `k = 8`, reaches 0.29 at `k = 16` (vs. uniform 0.98). The spin-symmetric IQP circuit captures the Ising target's structure across the full marginal hierarchy.
>
> ==**Blobs: only at low orders.**== TV under 0.05 for `k ≤ 2`; exceeds 0.4 by `k = 8`; reaches 0.79 at `k = 16` (vs. uniform 0.99). Plausibly diagnosable as under-training + expressivity cap (`max_weight = 4`), not a fundamental failure of the architecture.

These two datasets tell consistent-but-different stories. The Rudolph tension predicts exactly this split: Ising in its paramagnetic phase has low effective Pauli bodyness — matchable by low-`σ` MMD. Blobs is a "high-bodyness" target — the exact case the Rudolph argument says is hard. See [[Anti-Concentration vs Marginal Agreement#2. What Rudolph Actually Predicts]].

---

## 8. What this doesn't answer (honest limits)

> [!warning] Scope boundaries
> - **Only two datasets.** The paper tests 7 (`8_blobs`, `2D_ising`, `spin_glass`, `scale_free`, `MNIST`, `genomic-805`, `dwave`). The other five have `n ∈ {256, 484, 784, 805, 1000}` — far beyond what `probability_vector_exact` can enumerate. Testing them requires Proposition-1-style `⟨Z_a⟩` estimators (Option 2 from the original scoping); not done here.
> - **Under-trained.** 1000 iters vs. paper's 10 000. Blobs especially. Converged Ising is a better data point than converged blobs.
> - **Reduced expressivity on blobs.** `max_weight = 4` not 6. Blobs result cannot falsify the paper's design — only the training choice we made.
> - **Home-made datasets.** Our Gibbs-sampled Ising (T = 2.5) and inlined spin-blobs generator replace the paper's cached CSVs. Same distribution family, not the same samples.
> - **Empirical-target AC is biased.** 3k–5k samples over `2^16` bins is sparse; `p_emp` inflates scaled 2nd moment. `q_θ` AC is on the exact model distribution — the numbers there are clean.
> - **`passes_second_moment_threshold` flag is vacuous** in `iqp_bp`. Fix-the-code TODO in [[Codex Audit - spin_sym Export Gap#3. Collateral findings (not spin_sym)]].

---

## 9. What landed on disk

```
scripts/
  investigate_iqp_mmd_ac.py       # train → export → AC + M_k (first pass)
  rerun_ac_via_iqpopt.py          # re-run AC + M_k via iqpopt.probs() (respects spin_sym)
  _verify_checkpoint_faithful.py  # TV comparison of iqp_bp vs iqpopt probs
  plot_iqp_mmd_ac_investigation.py  # reads CORRECTED summaries, emits headline plot
  _codex_parse.py                 # JSONL → readable transcript
results/iqp_mmd_ac_investigation/
  checkpoints/ising_n16_iters1000_seed666.npz
  checkpoints/spin_blobs_n16_iters1000_seed666.npz
  ising_n16_iters1000_seed666_summary.json            # WRONG (spin_sym ignored)
  ising_n16_iters1000_seed666_summary_CORRECTED.json  # valid
  spin_blobs_n16_iters1000_seed666_summary.json
  spin_blobs_n16_iters1000_seed666_summary_CORRECTED.json
  ac_and_marginals_summary.png
  rerun_via_iqpopt.log
  verify_checkpoint_faithful.log
results/codex_challenge.jsonl      # full codex transcript
```

```dataview
TABLE file.mtime as "Last Modified"
FROM "Anti-Concentration"
SORT file.mtime DESC
LIMIT 5
```

---

## 10. Process lessons (for the file report)

> [!quote] What Codex caught that I missed
> I had trusted the `iqp_mmd.checkpoint_export` bridge because it had test coverage for "flat" gate structures. That coverage missed the semantically-distinct `spin_sym=True` path. No amount of my own testing would have found it; I didn't know `iqp_bp.IQPModel` lacked spin-symmetric handling. Codex found it via cross-referencing `iqpopt` docs + the `iqp_bp` model code. Second-opinion audit was decisive.

> [!tip] What to do next time
> 1. Before believing a bridge-and-validate pipeline, write a **round-trip TV test**: compare the validator's forward pass against the trainer's native forward pass on the exact same `θ`. Would have caught spin_sym in minutes.
> 2. When importing a third-party trainer (`iqpopt`), audit every boolean knob in its `__init__` against the deterministic-side model. Each knob is a potential export corruption.
> 3. When a subagent returns fabricated results (see the 2026-04-22 chat where an earlier agent invented an "AC7–AC12 pipeline" table), force it to cite file paths + line numbers. Memory: [[User - Codex implements, not Claude]].

---

## 11. Related

- [[Anti-Concentration vs Marginal Agreement]] — the theoretical framing that made the two questions inseparable.
- [[AC7 to AC12 Implementation]] — the earlier pipeline that inspired `per_order_marginal_mismatch`.
- [[AC12 Bandwidth Sweep Results 2026-04-21]] — the `binary_mixture` / sandbox results my teammate confused with paper plots.
- [[Codex Audit - spin_sym Export Gap]] — detailed record of the bug Codex caught.
- [[Anti-Concentration]] — strict AC definitions.
- [[References#Paper 2503.02934]] — Recio-Armengol, Ahmed & Bowles. The paper under scrutiny.
- [[References#Paper 2305.02881]] — Rudolph et al., the bandwidth-bodyness argument.

---

%%
Change log:
- 2026-04-23: initial write-up after fix + Codex round.
%%
