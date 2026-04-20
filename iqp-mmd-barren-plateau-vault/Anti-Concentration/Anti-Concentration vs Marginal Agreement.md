---
title: Anti-Concentration vs Marginal Agreement (the real question)
aliases:
  - AC vs Marginals
  - AC vs Marginal Agreement
  - Rudolph Tension Applied
tags:
  - theory
  - anti-concentration
  - marginals
  - bandwidth
  - critique
  - rudolph
  - ghosh-kim
  - recio-armengol
date: 2026-04-19
---

# Anti-Concentration vs Marginal Agreement

> [!abstract] What this note is
> The supervisor's 2026-04-19 ask framed the question as "were the learned distributions anti-concentrated?" — but the Rudolph bandwidth argument and the Ghosh–Kim / Recio-Armengol paper are really about **agreement on high-order marginals**, which is a *different* property. This note separates the two concepts, audits what the paper actually tests (and does not test), and re-casts [[AC7 to AC12 Implementation|`AC11`/`AC12`]] in terms of what they can genuinely measure.
>
> The conceptual split matters for how we write up results for the supervisor — conflating them will read as confused.

---

## 1. Two Properties, Not One

> [!important] Clean definitions
> **Anti-concentration** is a property of **one** distribution. **Marginal agreement** is a property relating **two** distributions.

### Anti-Concentration (Bremner–Montanaro–Shepherd, 2016)

A distribution $q$ over $\{0,1\}^n$ is anti-concentrated iff $q$ does not pile mass onto too few bitstrings. Equivalent forms:

- **Threshold form:** $\Pr_x[q(x) \ge \alpha / 2^n] \ge \beta$ for constants $\alpha, \beta > 0$.
- **Collision form:** $2^{2n} \, \mathbb{E}_x[q(x)^2] = 2^n \sum_x q(x)^2 \ge \beta' > 1$.

This is a single-distribution property. It is what [[Anti-Concentration|our `AC1`–`AC6` track]] checks, and it is the hypothesis needed for IQP classical-hardness proofs.

### Marginal Agreement (Rudolph et al., 2305.02881)

The Gaussian-kernel MMD between $p$ and $q$ decomposes as

$$
\mathrm{MMD}^2_\sigma(p, q) = C \sum_S \tau^{\lvert S\rvert} \bigl(\langle Z_S\rangle_p - \langle Z_S\rangle_q\bigr)^2
\quad \text{with} \quad
\tau = \tanh\!\left(\tfrac{1}{4\sigma^2}\right).
$$

Two distributions *agree on order-$k$ marginals* iff $\langle Z_S\rangle_p \approx \langle Z_S\rangle_q$ for every $S$ with $\lvert S\rvert \le k$. This is a relational property. It is what the supervisor is actually pointing at when she says "the learned distribution coincides with the ideal one on high marginals."

> [!warning] The two are related but not equivalent
> A distribution can be anti-concentrated yet have all the wrong high-order marginals. Equally, a distribution that matches $p$ on every marginal is identical to $p$ — at which point its anti-concentration is inherited, not independently decided. The supervisor's framing collapsed the two; our write-ups and plots need to keep them apart.

---

## 2. What Rudolph Actually Predicts

Rewrite the spectral decomposition as an expectation over a Bernoulli distribution on the Pauli weight:

$$
p_\sigma = \frac{1 - e^{-1/(2\sigma^2)}}{2}, \qquad
\mathbb{E}[\lvert S\rvert] = n \cdot p_\sigma.
$$

This gives a direct "effective bodyness" for the MMD at bandwidth $\sigma$:

| $\sigma$ regime | $p_\sigma$ | Avg Pauli weight | What the loss sees | Trainability |
|---|---|---|---|---|
| $\sigma \in \Theta(n)$ | small | low | low-body marginals | polynomial variance — trainable |
| $\sigma \in \Theta(\sqrt n)$ | moderate | $\sim \sqrt n$ | mid-order | marginal — edge of plateau |
| $\sigma \in \mathcal{O}(1)$ | $\to 1/2$ | $\sim n/2$ | high-body correlations | exponential concentration — barren plateau |

> [!tip] The operational reading
> Bandwidth $\sigma$ is a low-pass filter on the Pauli-weight axis. It does **not** choose whether the loss is trainable; it chooses **which marginals the loss can distinguish at all**. A model trained at large $\sigma$ has no gradient signal that could ever match high-order marginals — whatever high-order agreement it has is a byproduct of the model class and the init, not something the loss enforced.

This is the real statement from [[References#Paper 2305.02881|Rudolph 2305.02881]], and it applies *regardless* of whether the learned distribution is anti-concentrated.

---

## 3. Reading Recio-Armengol, Ahmed & Bowles (2503.02934) in This Light

### What bandwidth they actually used

From their Table 1 (extracted on 2026-04-19):

| Experiment | $n$ | $\sigma_1$ → avg weight | $\sigma_3$ → avg weight |
|---|---|---|---|
| D-Wave | 484 | 7.8 → **2** | 3.9 → **8** |
| MNIST | 784 | 9.9 → **2** | 3.4 → **17** |
| Scale-free | 1000 | 11.2 → **2** | 3.8 → **17** |
| Genomic | 805 | 10.0 → **2** | 4.2 → **11** |

The **largest** Pauli weight probed in the whole paper is 17, at $n$ up to 1000. That is ~2% of $n$. Nowhere close to the $\sigma \in \mathcal{O}(1)$ regime Rudolph identifies as needed for high-body correlations. Their own §9.3 concedes the bandwidths are "significantly smaller than $\sqrt{n/4}$" — at $n = 1000$ this still bounds probed weight at $\sim 20$, not $\sim 500$.

> [!important] The loss had no gradient signal above weight ~17
> Whatever high-order agreement their learned distributions have cannot have been enforced by training. This is the crux of the supervisor's implicit critique.

### What direct evidence they give on marginals

- **Weight 1:** Not really "learned" at all. Data-dependent initialization (§8.1.2) sets $\theta$ so that weight-1 marginals of $q_\theta$ match the data's bit frequencies **at init**. Single-qubit agreement is hand-set; training cannot take credit.
- **Weight 2:** Covariance plots (Figures 7, 8, 9, 11) show the IQP model reproduces the *structure* of the target covariance but with systematically *weaker* magnitudes. §9.5 admits: "tend to produce weaker correlations than the true distributions." So even at weight 2 — the order the loss most directly trains — agreement is imperfect.
- **Weight $\ge 3$:** No direct evidence. No plots of triple correlations, no cumulants, no weight-$k$ mismatch for $k \ge 3$. The aggregate MMD numbers on test sets mix weight-2 errors with the tails of the higher-weight distribution.
- **Evolution during training:** Only aggregate loss curves (Figure 3). No plot of per-order mismatch vs training step.

The last two bullets are the gaps the supervisor is pointing at. They are also exactly what [[AC7 to AC12 Implementation|our `AC10` trajectory]] is built to fill.

### What the paper implies about anti-concentration (indirectly)

- **Binary blobs:** Even 14 892 parameters with weight-$\le 6$ gates cannot reproduce the eight-mode structure; the learned distribution has those modes *plus mass elsewhere*. More spread than the target.
- **MNIST:** IQP model judged equivalent to target with pixel-flip noise $p = 0.3$. Smoother, more spread than target.
- **D-Wave, scale-free:** Learned covariances are *weaker* than ground truth — $q_\theta$ sits closer to the product of its marginals than $p$ does.

> [!success] The consistent pattern
> The learned distributions are **systematically smoother** — more anti-concentrated — than the targets. Training them under a low-bandwidth MMD erased the target's high-order structure and left something closer to a noised version of the target. On the strict Bremner–Montanaro–Shepherd definition, they would pass anti-concentration. They fail the *opposite* test: they have **too much** uniformity, not too little.

So the supervisor's Q1 ("were the learned distributions anti-concentrated?") almost certainly answers **yes** — but for a reason that undercuts the paper's framing, because it means training pushed the model toward high-entropy regions where the classical hardness hypotheses are vacuously satisfied but the target has been lost.

---

## 4. What This Means for `AC11` and `AC12`

The pipeline shipped already produces the right quantities. This note mainly changes how we *frame and plot* them.

### Re-cast `AC11` (was: "check AC on Ghosh–Kim learned distributions")

Old framing: "does training preserve anti-concentration?"

New framing: **"does training preserve *target* anti-concentration, or collapse toward uniform?"** The headline finding is likely going to be: the learned distribution passes strict AC trivially but has *higher* entropy than the target. Report both:

- `ac_scaled_second_moment` at init and end-of-training (learned trajectory) — the `AC1`–`AC6` primary scalar.
- `ac_scaled_second_moment` of the **target** (from data samples or exact when available) — this is the reference, not 1.
- **Delta:** `scaled_second_moment(target) - scaled_second_moment(learned)` per training step. Positive and growing means training is smoothing the distribution away from the target's mode structure.

### Re-cast `AC12` (was: "σ sweep of per-order TV")

Old framing: "which orders get matched, vs $\tau^k$ prediction?"

New framing sharpens to the Rudolph tension:

- Plot $M_k(\theta) \;\equiv\; \text{mean}_{\lvert a\rvert = k}\bigl(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta}\bigr)^2$ as a **heatmap** over $(k, \text{step})$, one heatmap per $\sigma$. `mean_fourier_squared_error` in [[AC7 to AC12 Implementation#7. Field Reference (Trajectory Row Schema)|the trajectory row schema]] is already $M_k$.
- Expected qualitative finding: low-$k$ rows converge fast; high-$k$ rows barely move. The "height" at which the heatmap freezes maps to the Rudolph effective-bodyness $n \cdot p_\sigma$.
- Overlay $\tau^k$ on the final-step mismatch curve. If the *learned* mismatch sits near the $\tau^k$ envelope, the kernel's low-pass was the tight constraint; if it sits above, the model class was also limiting.

### Missing piece to add: **target power spectrum**

Rudolph's prediction depends on the data too. For each target distribution, compute

$$
P_p(k) \;=\; \text{mean}_{\lvert a\rvert = k} \langle Z_a\rangle_p^2
$$

from the data samples. This is a one-time computation per dataset and is a natural diagnostic to ship alongside `AC11`/`AC12`. A dataset with a flat $P_p(k)$ across orders (e.g. binary blobs, which is essentially a weight-$n/2$ code) is inherently hard for low-$\sigma$ MMD; a dataset with $P_p(k)$ concentrated at low $k$ (e.g. a noisy Ising in its paramagnetic phase) is easy.

> [!todo] Concrete action item
> Add `compute_power_spectrum(data, max_order)` to [[AC7 to AC12 Implementation|`src/iqp_bp/distributions/marginal_metrics.py`]]. Emit it as a sidecar (`power_spectrum.json`) once per run, not per step — the target does not change during training. This is the diagnostic that ties `AC12` "to the data itself" as the supervisor asked.

---

## 5. Writing Up the Critique

If we are going to argue the paper's quantum-advantage claim is weaker than it looks, the clean form of the critique is:

1. **Training only enforces low-order marginal agreement.** Their own bandwidth choices bound the probed Pauli weight at $\lesssim 17$, regardless of $n$. (Table 1, §9.3.)
2. **The paper does not test high-order agreement.** No cumulant plots, no $M_k$ for $k \ge 3$, no per-order trajectory. (§9.4–9.5 discuss weight 2 at most.)
3. **Learned distributions are visibly smoother than targets.** Blobs mass leaks off-pattern; MNIST looks like pixel-flipped MNIST; D-Wave / scale-free covariances are weaker than ground truth. (§9.5, Figures 7–11.)
4. **The "quantum advantage" is therefore a trainability advantage, not a distributional-capture advantage.** Three of the four contested wins are cases where the classical baselines collapsed under matched hyperparameters. The IQP model won by not failing, not by capturing the target.

> [!quote] How to phrase it for the supervisor
> "On the strict Bremner–Montanaro–Shepherd sense, the learned distributions are almost certainly anti-concentrated — but trivially so, because training smoothed them toward high-entropy regions. The harder question is whether they agree with the target on high-order marginals, and the paper's own bandwidth choices guarantee that its loss could not have enforced such agreement. Our `AC11`/`AC12` pipeline tests this directly by plotting $M_k(\theta)$ as a function of training step and bandwidth."

---

## 6. What Changes in Our Plan

- No code changes needed for `AC11`/`AC12` — `summarize_by_order` already returns `mean_fourier_squared_error` per order per step. That's $M_k$.
- **Add:** `compute_power_spectrum(data, max_order)` + `power_spectrum.json` sidecar.
- **Add:** "target `scaled_second_moment`" to the per-run summary (data-side collision probability) so the learned vs target AC gap is readable at a glance.
- **Add:** a plot helper (`scripts/plot_m_k_heatmap.py` or similar) — heatmap of $M_k$ over $(k, \text{step})$, one per σ, overlay of $\tau^k$ on the final row.
- **Update:** [[AC7 to AC12 Implementation#5. Anticipated Supervisor Questions (FAQ)|the FAQ]] to include the "AC passes but not for the reason the paper needs" framing.

---

## 7. Related

- [[AC7 to AC12 Implementation]] — what the pipeline already computes (including $M_k$ via `mean_fourier_squared_error`)
- [[Design Decisions - AC7 to AC12]] — scope decisions for the 2026-04-19 extension
- [[Anti-Concentration]] — strict AC definitions, the `AC1`–`AC6` track
- [[Bandwidth Marginals]] — `AC12` artifact contract
- [[Gaussian Convention]] — the locked $\tau$ definition
- [[Kernel Spectral Decomposition]] — the Walsh-basis MMD identity
- [[Learning Task]] — the MMD training objective
- [[References#Paper 2503.02934]] — Recio-Armengol, Ahmed & Bowles (the paper under scrutiny)
- [[References#Paper 2305.02881]] — Rudolph et al. (the bandwidth-tension argument)
- [[References#Paper 2512.24801]] — Herbst, Brandić, Pérez-Salinas (strict AC definitions)
