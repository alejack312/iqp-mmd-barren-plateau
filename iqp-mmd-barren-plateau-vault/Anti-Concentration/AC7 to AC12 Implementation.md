---
title: AC7–AC12 Implementation (Learned-Distribution AC + Marginal Evolution)
aliases:
  - Supervisor Extension Implementation
  - Learned-Distribution AC Pipeline
  - Marginal Evolution Pipeline
  - AC7 AC12 Implementation
tags:
  - implementation
  - anti-concentration
  - marginals
  - bandwidth
  - training
date: 2026-04-19
---

# AC7–AC12 Implementation

> [!abstract] What this note is
> The supervisor asked two questions on 2026-04-19 (see [[Design Decisions - AC7 to AC12]]): (1) are the distributions *learned* in the Ghosh–Kim regime anti-concentrated? (2) do learned marginals match target marginals, and how does that evolve during training as a function of MMD bandwidth $\sigma$? This note explains **how the repo now satisfies those requests**, reading exactly what was built (verified against the working tree on 2026-04-19), how to run it, and what the artifacts mean. It closes with an FAQ that answers likely supervisor follow-ups.

> [!important] Experiment status
> Infrastructure: **built and tested** (35/35 tests pass; smoke run completes end-to-end).
> Long experiments `AC11` (Ghosh-Kim) and `AC12` (sigma sweep): **executed**. `AC11` results are summarized in [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]]; `AC12` results are summarized in [[AC12 Bandwidth Sweep Results 2026-04-21]].

> [!warning] Conceptual refinement — read this first
> After writing this note we realized the supervisor's framing conflates two distinct properties: strict [[Anti-Concentration|anti-concentration]] of one distribution vs **agreement on high-order marginals** between two distributions. [[References#Paper 2305.02881|Rudolph 2305.02881]] is really about the second; the learned distributions in [[References#Paper 2503.02934|Recio-Armengol/Ahmed/Bowles]] almost certainly pass strict AC, but trivially — they are *smoother than the target*. The real question is whether they match high-order marginals, and the bandwidths used in that paper (average Pauli weight $\le 17$ even at $n = 1000$) guarantee the loss could not have enforced such agreement. See [[Anti-Concentration vs Marginal Agreement]] for the full analysis and the resulting refinements to `AC11`/`AC12` framing.

---

## 1. What Was Delivered

### New modules

| Module | Purpose |
|---|---|
| [`src/iqp_bp/distributions/marginals.py`](../src/iqp_bp/distributions/marginals.py) | Exact + sampled marginals; exact + sampled Walsh/Fourier coefficients; subset enumeration with uniform sub-sampling; sampling from an exact probability vector |
| [`src/iqp_bp/distributions/marginal_metrics.py`](../src/iqp_bp/distributions/marginal_metrics.py) | `marginal_tv`, `marginal_chi_square`, `fourier_squared_error`, and the per-order stratified `summarize_by_order(...)` that produces the AC10/AC12 artifact |
| [`src/iqp_bp/training/trainer.py`](../src/iqp_bp/training/trainer.py) | `Trainer` class: SGD or Adam, exact-small-$n$ or Monte Carlo loss, per-step trajectory JSONL + `.npz` checkpoints, named RNG streams, pluggable checkpoint callback |
| [`src/iqp_bp/experiments/run_training.py`](../src/iqp_bp/experiments/run_training.py) | Runner that expands the training grid, wires the AC + marginal diagnostics callback, and writes one `runs/<setting>/` folder per resolved coordinate |
| [`src/iqp_bp/experiments/marginal_artifacts.py`](../src/iqp_bp/experiments/marginal_artifacts.py) | `write_marginal_summary(...)` — per-step sidecar JSON writer |

### Updated

- [`src/iqp_bp/cli.py`](../src/iqp_bp/cli.py) — new `run-training` subcommand.
- [`src/iqp_bp/config.py`](../src/iqp_bp/config.py) + [`configs/schema.yaml`](../configs/schema.yaml) + [`configs/base.yaml`](../configs/base.yaml) — new `training:` block validated like the other experiment blocks.
- [`src/iqp_bp/mmd/kernel.py`](../src/iqp_bp/mmd/kernel.py) — `_gaussian_tau(sigma)` lifted to a public helper so the marginal metrics module and the trainer share one locked [[Gaussian Convention|τ definition]].
- [`src/iqp_bp/mmd/gradients.py`](../src/iqp_bp/mmd/gradients.py) — `grad_mmd2_analytic` can now consume pre-sampled `a_samples`, `exp_p`, and `observable_weights` so the trainer does not re-sample observables on the gradient pass.
- [`docs/technical/anti-concentration.md`](../docs/technical/anti-concentration.md) — new "Learned-Distribution Trajectories" section.
- [`docs/technical/bandwidth-marginals.md`](../docs/technical/bandwidth-marginals.md) — new file documenting the σ-sweep contract (intentionally claims no results until `AC12` runs).

### Checked-in configs

- [`configs/experiments/training_smoke.yaml`](../configs/experiments/training_smoke.yaml) — $n=6$ product_state, 4 steps, completes in seconds. This is the `AC10` smoke test.
- [`configs/experiments/ghosh_kim_small_n.yaml`](../configs/experiments/ghosh_kim_small_n.yaml) — $n=9$ ZZ lattice, binary-mixture target, 3 bandwidths (`σ ∈ {1, 3, 9}`), 20 Adam steps, exact AC path. This is the `AC11` small-$n$ cell.
- [`configs/experiments/ghosh_kim_large_n_sampled.yaml`](../configs/experiments/ghosh_kim_large_n_sampled.yaml) — $n=20$ sparse Erdős–Rényi, Ising target, 3 bandwidths, 10 Adam steps, sample-histogram AC path. This is the `AC11` large-$n$ cell.
- [`configs/experiments/bandwidth_marginal_sweep.yaml`](../configs/experiments/bandwidth_marginal_sweep.yaml) — $n=9$, 4 bandwidths (`σ ∈ {1, 3, 9, 27}`), 20 Adam steps, exact path. This is `AC12`.

### Tests

35 tests pass across `tests/test_marginals.py`, `tests/test_marginal_metrics.py`, `tests/test_trainer.py`, `tests/test_run_training.py`, and the already-existing `tests/test_config.py`.

---

## 2. What the Two Supervisor Questions Look Like in the Pipeline

### Q1 — Are the learned distributions anti-concentrated?

Answer path: every persisted training step emits the exact (small $n$) or sampled (larger $n$) anti-concentration fields. This is the same [[Anti-Concentration|`check_anti_concentration`]] we already had for `AC1`–`AC6`, but now called on the *learned* distribution at every checkpoint of the training trajectory.

Each `trajectory.jsonl` row carries:

| Field | Meaning | From |
|---|---|---|
| `ac_scaled_second_moment` | $2^n \sum_x p(x)^2$ — the paper's primary scalar | `check_anti_concentration` |
| `ac_primary_beta_hat` | $\hat\beta(\alpha{=}1) = 2^{-n}\lvert\{x : p(x)\ge 2^{-n}\}\rvert$ | `check_anti_concentration` |
| `ac_passes_primary_threshold` | `β̂(1) ≥ 0.25` (configurable) | `check_anti_concentration` |
| `ac_passes_second_moment_threshold` | `scaled_second_moment ≥ 1.0` (configurable) | `check_anti_concentration` |
| `ac_max_probability_scaled` | $2^n \max_x p(x)$ | `check_anti_concentration` |
| `ac_beta_hat_by_alpha` | `β̂(0.5)`, `β̂(1)`, `β̂(2)` (configurable grid) | `check_anti_concentration` |
| `distribution_mode` | `"exact"` or `"sample"` (see below) | `diagnostics.distribution_mode` |

For `n ≤ exact_probability_max_n` (default 12), the AC check uses the exact probability vector. For larger $n$, the runner draws `sample_count` bitstrings from the exact probability vector (cap `sample_from_exact_probability_max_n = 20`) and runs the histogram-mode AC checker. The `AC11` large-$n$ config uses this sampled mode at $n=20$ with `sample_count=8192`.

> [!success] AC11 has been run
> The `ghosh_kim_small_n.yaml` and `ghosh_kim_large_n_sampled.yaml` sweeps populate `results/ac_ghosh_kim/.../trajectory.jsonl` with the above fields at step 0, every 5 steps, and the final step. See [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]].

### Q2 — Do learned marginals match targets, and how does that evolve with σ?

Answer path: every persisted training step computes `summarize_by_order(data, q_learned, n, orders=1..n, sigma=σ)` and attaches:

- An inline per-order breakdown on the trajectory JSONL row (`marginal_orders`).
- A sidecar JSON at `runs/<setting>/marginals/step_XXXX.json`.
- A scalar `marginal_weighted_mmd2_total` on the row, which is the kernel-weighted sum across all orders (this is exactly the MMD² that the Gaussian kernel assigns to the current mismatch).

Each order-$k$ entry carries:

| Field | Formula | Role |
|---|---|---|
| `mean_tv` / `median_tv` / `max_tv` | TV distance on the marginal histogram | Scale-free agreement |
| `mean_chi_square` / `median` / `max` | $\sum (p_S - q_S)^2 / \max(q_S, \epsilon)$ | Tail-sensitive |
| `mean_fourier_squared_error` / `median` / `max` | $(\langle Z_S\rangle_p - \langle Z_S\rangle_q)^2$ | The per-mode MMD ingredient |
| `subset_count` / `total_subsets` / `used_all_subsets` | Subset budget bookkeeping | Enumerated or uniformly sampled |
| `tau_power` | $\tau^{\lvert S\rvert}$ | [[#Rudolph Weight]] |
| `order_weight` | $\binom{n}{k}\tau^k / (1+\tau)^n$ | Probability mass that the Gaussian kernel puts on this order |
| `weighted_mmd2_contribution` | `order_weight · mean_fourier_squared_error` | Per-order MMD² contribution the kernel *actually sees* |

The σ-sweep (`AC12`) simply runs the same pipeline at four σ values in `{1, 3, 9, 27}` and lets us plot `mean_fourier_squared_error` (or TV) per order, for each σ, versus `tau_power` at that σ.

> [!tip] This directly answers "how do marginals evolve during training"
> Within one run, `marginal_orders` at step 0 vs step $N$ is the evolution. Across runs (different σ), `weighted_mmd2_contribution` per order is the bandwidth-dependent picture.

---

## 3. How to Run It

```bash
# AC10 smoke test (takes seconds, already run during verification)
python -m iqp_bp.cli run-training configs/experiments/training_smoke.yaml

# AC11 small-n exact cell
python -m iqp_bp.cli run-training configs/experiments/ghosh_kim_small_n.yaml

# AC11 large-n sampled cell
python -m iqp_bp.cli run-training configs/experiments/ghosh_kim_large_n_sampled.yaml

# AC12 bandwidth sweep
python -m iqp_bp.cli run-training configs/experiments/bandwidth_marginal_sweep.yaml
```

Each run writes:

```
results/<experiment_name>/
├── config.json                     # merged config (from P1)
├── manifest.json                   # resolved-grid manifest (from P1)
├── results.jsonl                   # one summary row per (family, σ, init, n)
└── runs/
    └── <family>__n<K>__<kernel>__<init>__<dataset>__sigma<σ>__…/
        ├── trajectory.jsonl        # one row per persisted step
        ├── checkpoints/
        │   └── step_XXXX.npz       # IQPModel checkpoint per step
        └── marginals/
            └── step_XXXX.json      # full marginal summary per step
```

The `.npz` files are the same [[Checkpoint Bridge]] format used by `AC4` — they can be fed back into `run-validation` if you want a post-hoc AC report or a plot.

---

## 4. What the Smoke Run Told Us

The `training_smoke` run ($n=6$ product_state, $\sigma=4$, small-angle init, 4 SGD steps, product-Bernoulli target) is the only real trajectory in the repo right now. Step 0 row:

| Quantity | Value | Reading |
|---|---|---|
| `loss` (MMD²) | $0.079$ | Initial mismatch |
| `ac_scaled_second_moment` | $42.7$ | Far above the `≥ 1` threshold, so AC passes on the scalar |
| `ac_primary_beta_hat` (α=1) | $0.0625$ | Only 4/64 bitstrings carry $\ge 2^{-n}$ — **fails** `β̂(1) ≥ 0.25` |
| `marginal_orders[1].mean_tv` | $0.47$ | Large single-qubit mismatch |
| `marginal_orders[1].order_weight` | $0.0854$ | Kernel weight on order 1 |
| `marginal_orders[1].weighted_mmd2_contribution` | $0.076$ | **97% of the loss is order-1** |
| `marginal_orders[2].order_weight` | $0.0033$ | Kernel weight on order 2 |
| `marginal_orders[2].weighted_mmd2_contribution` | $0.0026$ | ~3% of the loss |
| `marginal_orders[3].order_weight` | $6.95 \times 10^{-5}$ | Kernel weight on order 3 |
| `marginal_orders[3].weighted_mmd2_contribution` | $4.7 \times 10^{-5}$ | Negligible (<0.1%) |

> [!success] Rudolph's τ^|S| prediction is visible in the smoke row itself
> At $\sigma = 4$, $\tau = \tanh(1/64) \approx 0.0156$. The kernel weight on order $k$ is $\binom{n}{k}\tau^k / (1+\tau)^n$, which collapses by ~36× from order 1 to order 2 and another ~48× to order 3. The `weighted_mmd2_contribution` column shows this collapse directly. This is exactly the behavior the σ sweep (`AC12`) is designed to characterize.

Two immediate takeaways, still conditional on running the full sweeps:

1. **Mismatch rises with order** (`mean_tv`: 0.47 → 0.69 → 0.78 at orders 1–3). Low-order marginals are the most broken at init and will be the first to converge.
2. **The kernel doesn't care** about high-order mismatch at large σ: the weighted contribution is dominated by low orders. This is the trainability–expressivity trade-off the supervisor is pointing at.

---

## 5. Anticipated Supervisor Questions (FAQ)

> [!faq]+ "Are the learned distributions anti-concentrated?"
> **Short version:** yes, almost certainly — but in a way that *works against* the paper's framing, not for it. See [[Anti-Concentration vs Marginal Agreement]].
> **Why:** [[References#Paper 2503.02934|Recio-Armengol/Ahmed/Bowles]] use bandwidths that probe average Pauli weight $\le 17$ at $n$ up to 1000. Training under such a kernel cannot enforce agreement at weight $\ge 3$, and their own covariance plots (§9.5) show the learned distribution has *weaker* correlations than the target. The learned $q_\theta$ is smoother than $p$ — which is anti-concentrated by construction, trivially. The real question (what the supervisor is actually after) is whether $q_\theta$ matches $p$ on high-order marginals; that is what `AC11`/`AC12` test.
> **Infrastructure answer:** every trajectory row carries both `ac_scaled_second_moment` and the interpretable `β̂(α)` diagnostic. Plot the *gap* between target and learned AC over training — a growing gap means training is smoothing the model away from the target.
> **Empirical status:** `AC11` sweeps are complete; see [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]].

> [!faq]+ "So what's the supervisor really asking?"
> She's asking for **$M_k(\theta) \equiv \text{mean}_{\lvert a\rvert = k}(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta})^2$** plotted as a function of training step and bandwidth. That is already `mean_fourier_squared_error` on every trajectory row. The missing piece is a heatmap plot over $(k, \text{step})$ and a reference overlay of $\tau^k$ on the final step. No new code; plotting only.

> [!faq]+ "What does 'high marginal' mean in your code?"
> All orders $\lvert S\rvert \in \{1, \ldots, n\}$, stratified. `summarize_by_order` defaults `orders = range(1, n+1)`. For $n=9$ (the `AC11`/`AC12` cells), "high" means orders 5–9; for $n=20$ the large-$n$ cell caps `max_order=6` because dense enumeration at order 10+ is intractable. Nothing is silently truncated — `total_subsets` vs `subset_count` tells you whether enumeration was exhaustive or sampled.

> [!faq]+ "How is this tied to Rudolph 2305.02881?"
> The Gaussian kernel's spectral decomposition is $k_\sigma(x, y) = C \sum_S \tau^{\lvert S\rvert} \chi_S(x)\chi_S(y)$ with $\tau = \tanh(1/(4\sigma^2))$. This is Rudolph's bandwidth weight, and it already lived in [[Gaussian Convention|`mmd/kernel.py`]]. The trainer now carries σ into `summarize_by_order`, which reports $\tau^k$ as `tau_power` and $\binom{n}{k}\tau^k/(1+\tau)^n$ as `order_weight` per order. The `weighted_mmd2_contribution` field is the exact per-order MMD² mass — i.e., Rudolph's trainability prediction made observable per training step.

> [!faq]+ "How is this tied to Ghosh–Kim 2503.02934?"
> Ghosh–Kim trains IQP circuits at large $n$ and checks anti-concentration on *initializations*. Our `AC11` configs mirror their recipe — ZZ lattice or sparse ER generators, binary-mixture or Ising targets, data-independent uniform init — at $n=9$ (exact) and $n=20$ (sampled). Because we run the same AC check at *every training step*, not just init, we can also answer whether training itself breaks anti-concentration. The paper does not look at this directly.

> [!faq]+ "Why don't you reproduce the 1000-qubit regime?"
> Exact probability vectors require $\mathcal{O}(2^n)$ memory — infeasible above $n \approx 20$. We report both an exact and a sampled cell so the claim isn't dependent on one scale. The sampled cell at $n=20$ with 8192 samples gives a $\hat\beta(1)$ estimate with standard error $\sim 1/\sqrt{M} \approx 0.011$, which is much smaller than the pass/fail threshold of 0.25. We can push $n$ higher if the supervisor wants; it's a config change.

> [!faq]+ "How do you know the trainer is correct?"
> Three kinds of test (see [[Tests]]):
> 1. Unit: Fourier↔pmf identity, exact-vs-sample convergence, normalization laws.
> 2. Loss decrease: on a toy product-Bernoulli target, the trajectory loss is monotone (within MC noise) for both SGD and Adam.
> 3. Determinism: two trainer runs with the same seed produce bit-identical trajectories.
> End-to-end: `training_smoke` runs to completion and the `trajectory.jsonl` validates against the expected schema.

> [!faq]+ "What's the MC bias of the sampled AC path?"
> `β̂(1)` estimated from $M$ samples has bias $\mathcal{O}(1/M)$ and variance $\hat\beta(1-\hat\beta)/M$. The pipeline flags these rows with `distribution_mode="sample"` and `distribution_sample_count=M`. Per the [[Anti-Concentration|locked convention]], sampled AC is explicitly labeled a secondary diagnostic.

> [!faq]+ "Why Adam instead of SGD?"
> Both supported (`optimizer: sgd` or `adam`). Ghosh–Kim uses Adam for their trainability experiments, so `AC11`/`AC12` configs match that. The smoke test uses SGD to keep the math auditable.

> [!faq]+ "Does the gradient use the exact path or Monte Carlo?"
> Configurable via `training.loss_mode`: `exact_small_n` (enumerate $a$ and $z$, no sampling), `auto` (exact when $n \le$ `exact_loss_max_n`), or anything else (Monte Carlo with `num_a_samples` kernel samples and `num_z_samples` IQP samples). `AC11` small-$n$ uses exact; `AC12` uses exact at $n=9$. `AC11` large-$n$ uses sampled because $n=20$ exceeds the exact cap.

> [!faq]+ "How long will the full sweeps take?"
> Napkin math only. Per training step at $n=9$: exact loss $\mathcal{O}(2^n \cdot m)$ + exact gradient $\mathcal{O}(m \cdot 2^n \cdot A)$ where $A$ is the sampled-observable count. `AC11` small-$n$ = 20 steps × 3 σ × 1 family = 60 checkpoints. `AC12` = 20 steps × 4 σ = 80 checkpoints. Expected: a few minutes to low tens of minutes on CPU per config. `AC11` large-$n$ = 10 steps × 3 σ = 30 checkpoints of Monte Carlo-heavy work; budget an hour.

> [!faq]+ "What if training collapses the distribution?"
> That's the interesting case. The pipeline would show `ac_passes_primary_threshold: true` at step 0 and `false` at a later step, with the `marginal_weighted_mmd2_total` dropping while `max_probability_scaled` rises. Call this out in the `AC11` summary write-up in [[Anti-Concentration]].

> [!faq]+ "Can I see per-step learned marginals for one qubit?"
> Yes. Each `marginals/step_XXXX.json` has the full per-order table. For single-qubit marginals at qubit $i$, enumerate `subset = [i]` on the `.npz` checkpoint via `exact_marginal(IQPModel(...).probability_vector_exact(), [i])`. No notebook is checked in yet — this is a candidate for a `scripts/plot_marginal_evolution.py` helper as soon as real data lands.

> [!faq]+ "What's the data-dependence claim the supervisor mentioned?"
> "High marginals" reflect structure in the *target distribution*, not just the model family. For a product-Bernoulli target all high-order marginals factor into single-qubit statistics, so MMD² converges quickly even at large σ. For an Ising or binary-mixture target, the data has genuine high-order correlations that only a small-σ kernel can learn — and small-σ is where barren plateaus start. This is the trade-off space `AC12` maps out.

---

## 6. What's Still Missing

Matches handoff message, plus the additions from [[Anti-Concentration vs Marginal Agreement#6. What Changes in Our Plan|the conceptual refinement]].

- [x] Run `AC11` small-$n$ sweep -> `results/ac_ghosh_kim/small_n_exact/`.
- [x] Run `AC11` large-$n$ sampled sweep -> `results/ac_ghosh_kim/large_n_sampled/`.
- [x] Run `AC12` sigma sweep -> `results/bandwidth_marginal_sweep/`.
- [x] **Add** `compute_power_spectrum(data, max_order)` to `src/iqp_bp/distributions/marginal_metrics.py` — emit `power_spectrum.json` once per run (target does not change during training). Ties `AC12` to the data as the supervisor asked.
- [x] **Add** target `scaled_second_moment` to each run summary so the learned-vs-target AC *gap* is visible at a glance.
- [x] **Add** `scripts/plot_m_k_heatmap.py` — heatmap of $M_k$ over `(k, step)`, one per sigma, with tau^k overlay on the final-step row. This is the actual deliverable the supervisor is asking for.
- [x] Write the results subsection in [[Anti-Concentration]] with pass/fail per cell **and** the learned-vs-target AC gap.
- [x] Write [[Bandwidth Marginals]] interpretation via [[AC12 Bandwidth Sweep Results 2026-04-21]].

---

## 7. Field Reference (Trajectory Row Schema)

```text
step                               int    training step (0 = init snapshot)
loss                               float  MMD²(p_data, q_θ) at this step
theta                              list   θ vector at this step
wall_clock_sec                     float  cumulative seconds since run start
checkpoint_path                    str    path to the .npz for this step

loss_stderr                        float  (MC only) stderr across observables
loss_num_observables               int    number of Walsh modes summed

diagnostics_available              bool   false ⇒ diagnostics skipped
distribution_mode                  str    "exact" or "sample"
distribution_sample_count          int    (sample mode only) M

ac_mode                            str    "exact_probabilities" or "samples"
ac_primary_alpha                   float  α for the primary β̂ check
ac_primary_beta_hat                float  β̂(α) on learned distribution
ac_passes_primary_threshold        bool   β̂(α) ≥ β_min
ac_scaled_second_moment            float  2^n Σ p(x)²
ac_passes_second_moment_threshold  bool   ≥ second_moment_threshold
ac_max_probability_scaled          float  2^n max_x p(x)
ac_beta_hat_by_alpha               dict   {"0.5": …, "1.0": …, "2.0": …}

marginal_summary_path              str    sidecar JSON file
marginal_weighted_mmd2_total       float  Σ_k order_weight · mean(FSE_k)
marginal_orders                    list   per-order entries (see below)
```

Per-order entry inside `marginal_orders`:

```text
order                             int    |S|
subset_count                      int    subsets examined
total_subsets                     int    C(n, k)
used_all_subsets                  bool   exhaustive or sampled
mean_tv / median_tv / max_tv      float  TV distance on marginals
mean_chi_square / median / max    float  chi-square style mismatch
mean_fourier_squared_error / …    float  (〈Z_S〉_p − 〈Z_S〉_q)²
tau_power                         float  τ^|S|  (Gaussian only)
order_weight                      float  C(n, k) τ^k / (1+τ)^n
weighted_mmd2_contribution        float  order_weight · mean_FSE
```

---

## 8. Glossary

### Rudolph Weight

The Gaussian MMD kernel decomposes as

$$
k_\sigma(x, y) = C \sum_{S \subseteq [n]} \tau^{\lvert S\rvert}\, \chi_S(x) \chi_S(y), \qquad
\tau = \tanh\!\left(\frac{1}{4\sigma^2}\right)
$$

so the weight on order-$k$ Walsh modes is $\tau^k$, and the fraction of *total* kernel mass on order $k$ is $\binom{n}{k}\tau^k / (1+\tau)^n$. This is what `tau_power` and `order_weight` record. The trainability story from [[References#Paper 2305.02881|Rudolph 2305.02881]] is precisely that small $\sigma$ (large $\tau$) spreads weight into high $k$, creating a barren plateau; large $\sigma$ (small $\tau$) concentrates weight at low $k$, making training easy but the loss blind to high-order correlations.

### Learned distribution

The output probability vector $q_\theta(x) = \lvert\langle x\rvert H^{\otimes n} D_\theta H^{\otimes n} \lvert 0^n\rangle\rvert^2$ for a specific θ. In this pipeline, "learned" means θ has been updated by at least one optimizer step — not just the initialization.

---

## 9. Related

- [[Design Decisions - AC7 to AC12]] — why the scope looks the way it does
- [[Anti-Concentration]] — definitions, the `check_anti_concentration` primary-vs-secondary convention
- [[Weekly Task - Anti-Concentration]] — supervisor-facing weekly brief
- [[Presentation - Anti-Concentration]] — slide deck that the `AC11`/`AC12` results will feed
- [[Gaussian Convention]] — the locked $\tau$ formula
- [[Kernel Spectral Decomposition]] — the Walsh-basis identity
- [[Validation Runner]] — the existing AC artifact pattern reused in `AC10`
- [[Scaling Runner]] — the existing resolved-grid persistence pattern reused in `run_training.py`
- [[Checkpoint Bridge]] — the `.npz` format every persisted step writes
- [[TODO Roadmap]] — overall task list
- [[References#Paper 2503.02934]] — Ghosh–Kim
- [[References#Paper 2305.02881]] — Rudolph
