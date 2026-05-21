---
title: Grid5000 Bandwidth Marginal Error Sweep
date: 2026-05-18
tags:
  - anti-concentration
  - marginals
  - bandwidth
  - grid5000
  - iqp-mmd
status: meeting-report
related:
  - "[[iqp_mmd AC Investigation 2026-04-23]]"
  - "[[Anti-Concentration vs Marginal Agreement]]"
  - "[[Grid5000 Native iqp_mmd AC and Marginals 2026-05-03]]"
---

# Grid5000 Bandwidth Marginal Error Sweep - 2026-05-18

![[grid5000_bandwidth_marginal_summary_2026-05-18.png]]

## One-sentence takeaway

The training-kernel bandwidth matters a lot: on native `genomic-805`, the small bandwidth setting badly hurts marginal agreement, while the paper and large bandwidths are both strong, with the paper bandwidth slightly better on higher-order marginals. The small-bandwidth failure is a warning sign, not by itself a diagnosis: it is consistent with either quantum trainability issues such as barren plateaus or classical overfitting/noisy fitting.

## What `small`, `paper`, and `large` mean

These are three training-kernel bandwidth regimes. They are not new datasets; they are three different ways of measuring the MMD training loss.

Feynman version:

> The bandwidth is the size of the lens used during training. A small bandwidth is like a microscope: it reacts strongly to fine detail. A large bandwidth is like a wide-angle view: it smooths over detail and focuses on broader structure. The paper setting is the original lens used by the paper/prior run.

We chose only three regimes because the Grid'5000 deadline was tight. The goal was to answer the scientific question, not exhaustively tune bandwidth.

| label | genomic-805 sigma | 2D_ising / 8_blobs sigma | why this value |
|---|---:|---:|---|
| `small` | `[4.26859077]` | `[0.3]` | A narrower kernel. For genomic, this is the smallest component of the paper kernel mix. For the small controls, it is half of the lower paper bandwidth. |
| `paper` | `[10.0187221, 7.70050841, 4.26859077]` | `[0.6, 1.3]` | The baseline setting from the repo's paper-reproduction hyperparameter config / prior run. This is the anchor point. |
| `large` | `[16.0]` | `[2.6]` | A wider, smoother kernel. For the small controls, it is twice the upper paper bandwidth. For genomic, `16.0` was chosen as a broad single-kernel comparison that would still be computationally feasible. |

The intended contrast is simple:

- `small`: does focusing on fine detail improve high-order marginals, or does it overfit/noisily miss global structure?
- `paper`: what does the paper-config / prior-run reference setting do?
- `large`: does smoothing help low-order agreement but lose higher-order structure?

### Which bandwidth makes high-order marginals matter more?

Small bandwidth makes high-order structure matter more in the training loss. Large bandwidth makes high-order structure matter less.

Feynman version:

> The kernel bandwidth controls how picky the training loss is. With a small bandwidth, two samples only look similar if they match in a very detailed way, so the loss pays attention to sharper, higher-order patterns. With a large bandwidth, many samples look similar after smoothing, so the loss mostly cares about broad, low-order patterns.

A useful mental model is:

| bandwidth | training pressure | expected emphasis |
|---|---|---|
| small | less smoothing | higher-order correlations matter more |
| large | more smoothing | higher-order correlations are suppressed |
| paper | mixed/reference | compromise between the two |

This is about what the loss *asks for*, not a guarantee of what training will achieve. In this sweep, the small genomic bandwidth seems to make the training problem harder or noisier: even though it should put more pressure on fine structure, it produced much worse measured marginal TV. That is the interesting result.

### What the small-bandwidth failure might mean

My supervisor's interpretation is the right cautious framing:

> When the bandwidth is small, the loss becomes very sensitive to fine-detail / higher-order structure. If the learned model then has much worse marginal error, that can be a sign of either a quantum trainability problem, such as a barren plateau, or a classical statistical problem, such as overfitting/noisy fitting. The bandwidth sweep alone cannot tell which one it is.

Feynman version:

> Small bandwidth makes the training loss picky. A picky loss can fail for two different reasons. One reason is quantum: the optimizer cannot see a useful slope because the landscape is too flat or noisy. That is the barren-plateau story. The other reason is classical: the optimizer follows tiny details in the training sample that do not become good marginals. That is the overfitting story. Our current experiment sees the failure, but it does not identify the mechanism.

So the correct claim is:

- We found that small bandwidth produces much worse marginal agreement.
- This is compatible with barren-plateau-like trainability problems.
- It is also compatible with classical overfitting or noisy fine-scale fitting.
- We need deeper analysis before saying which mechanism caused it.

## The 30-second version

We trained separate IQP-MMD models with three bandwidth regimes:

| label | purpose | intuition |
|---|---|---|
| `small` | narrow kernel | pays attention to fine detail, but can over-focus on noisy local structure |
| `paper` | paper/default kernel mix | the reference setting from the prior run |
| `large` | wide kernel | smooths the target more aggressively, often favoring low-order structure |

Then we asked: after training, do the model's marginals match the target's marginals?

The answer so far:

- `genomic-805`: `small` is much worse; `paper` and `large` are close; `paper` is slightly better at `k = 6, 8`.
- `2D_ising`: `paper` is the best all-order setting.
- `8_blobs`: all three bandwidths completed; `paper` is best at high-order marginals, while `large` is best at the easiest low-order checks.

## Feynman explanation

### What is a marginal?

Imagine the full genomic sample as an 805-column spreadsheet. Each row is one person, and each column is one SNP bit.

The full distribution asks:

> How often does every possible 805-bit pattern occur?

That is impossible to check directly because there are `2^805` possible patterns.

A marginal asks a smaller question:

> If I only look at `k` selected columns, does the model match the real data on those columns?

So:

- `k = 1` checks single SNP frequencies.
- `k = 2` checks pairwise relationships.
- `k = 8` checks small 8-SNP joint patterns.

This gives us a practical way to ask whether the model learned real structure without enumerating the full distribution.

### What is marginal TV error?

TV distance is the amount of probability mass you would need to move to turn one distribution into the other.

Feynman version:

> If the target says a certain 4-bit pattern should appear 20% of the time, but the model says 12%, some probability mass is in the wrong place. TV adds up those mismatches across all patterns.

Interpretation:

- `0.00` means perfect marginal match.
- Smaller is better.
- Values near the uniform baseline mean the model is not doing much better than guessing.

### What are `M1`, `M2`, ..., `M16`?

`M_k` means: average marginal TV error when we look at `k` variables at a time.

So:

| quantity | what it checks | plain-English meaning |
|---|---|---|
| `M1` | all 1-variable marginals | Does each individual SNP/spin/blob bit have the right frequency? |
| `M2` | all or sampled 2-variable marginals | Does the model capture pairwise relationships? |
| `M4` | 4-variable marginals | Does the model capture small joint patterns? |
| `M8` | 8-variable marginals | Does the model capture richer multi-variable structure? |
| `M16` | full 16-variable distribution for small controls | For `n=16`, how close is the full learned distribution to the empirical target? |

Feynman version:

> Imagine testing a student on a song. `M1` asks whether they know each note's overall frequency. `M2` asks whether they know which pairs of notes tend to appear together. `M8` asks whether they know longer phrases. `M16`, when available, asks whether they know the whole song. The higher `k` gets, the harder the test becomes.

For native `genomic-805`, we only report up to `M8` because exact `M16` would require looking at the full `2^805` distribution, which is impossible. For `2D_ising` and `8_blobs`, `n = 16`, so we can compute all the way through `M16`.

### What does kernel bandwidth do?

The MMD kernel is the training loss's measuring lens.

Feynman version:

> A small bandwidth is like using a microscope. It sees tiny differences, but it can also chase noise. A large bandwidth is like squinting from far away. It sees broad shape, but may miss fine details. The paper bandwidth is the compromise the original method used.

The sweep asks which lens produces learned distributions with better marginal agreement.

## Method

This follows the marginal workflow from [[iqp_mmd AC Investigation 2026-04-23]], with the same caution about spin symmetry:

- For native `genomic-805`, exact enumeration over `2^805` is impossible, so we use the Pauli marginal estimator for `k = {1,2,3,4,6,8}`. The `+/-` value is estimator uncertainty.
- For `2D_ising` and `8_blobs`, `n = 16`, so exact probabilities are feasible. We compute `q_theta` through `iqpopt.IqpSimulator.probs(theta)`, which respects `spin_sym=True` for Ising.
- For exact `n=16` controls, `M_k` is mean TV over all subsets when feasible, or up to 128 uniformly sampled subsets per order.
- Uniform baselines are included for small datasets so we can tell whether learned marginals are meaningfully better than a trivial distribution.

Canonical artifact root on Grid'5000 Nantes:

```text
/home/ajackson/iqp-mmd-barren-plateau/results/grid5000_bandwidth_sweep_final
```

Aggregate files:

```text
results/grid5000_bandwidth_sweep_final/bandwidth_marginal_summary.json
results/grid5000_bandwidth_sweep_final/bandwidth_marginal_summary.png
results/grid5000_bandwidth_sweep_final/control_marginal_l2_summary.json
```

## Native genomic-805 results

All three primary genomic bandwidths completed at 1000 iterations.

| bandwidth | sigma | final loss | wall time | M1 | M2 | M3 | M4 | M6 | M8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | `[4.26859077]` | 0.000211 | 7.60 h | 0.1218 +/- 0.0083 | 0.1897 +/- 0.0078 | 0.2486 +/- 0.0087 | 0.3063 +/- 0.0081 | 0.3949 +/- 0.0087 | 0.4548 +/- 0.0086 |
| paper | `[10.0187221, 7.70050841, 4.26859077]` | 0.000572 | 17.38 h | 0.0045 +/- 0.0006 | 0.0170 +/- 0.0017 | 0.0300 +/- 0.0028 | 0.0468 +/- 0.0027 | 0.0771 +/- 0.0031 | 0.1161 +/- 0.0030 |
| large | `[16.0]` | 0.000451 | 7.67 h | 0.0033 +/- 0.0005 | 0.0164 +/- 0.0018 | 0.0318 +/- 0.0030 | 0.0480 +/- 0.0028 | 0.0808 +/- 0.0033 | 0.1214 +/- 0.0032 |

### Genomic interpretation

The small bandwidth setting is the outlier. At `k = 8`, it has mean TV `0.4548`, while paper is `0.1161` and large is `0.1214`.

Feynman version:

> If we randomly pick 8 SNP columns and compare model vs. data, the small-bandwidth model puts much more probability mass in the wrong places. The paper and large models mostly agree with the target on these 8-column views.

Important nuance:

- `large` has lower final training loss than `paper`.
- But `paper` has slightly better high-order marginal TV.
- `small` has a low final loss but very poor marginal TV, which is exactly why this result needs diagnostic follow-up.

So the training loss alone is not enough. We need the marginal check.

Interpretation of the small-bandwidth run:

> The small-bandwidth result should be presented as a red flag, not as proof of barren plateaus. In a quantum model, this pattern could come from barren-plateau-like trainability failure. In a classical/statistical reading, it could come from overfitting or chasing noisy fine-scale structure. The sweep shows that small bandwidth is bad here; it does not yet explain why.

## 2D_ising exact control

All three Ising controls completed. They stopped early due convergence, not walltime.

| bandwidth | sigma | iters run | final loss | M1 | M2 | M4 | M6 | M8 | M16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | `[0.3]` | 1359 | 0.000457 | 0.0058 | 0.0250 | 0.0658 | 0.1041 | 0.1509 | 0.5286 |
| paper | `[0.6, 1.3]` | 1524 | 0.000111 | 0.0058 | 0.0103 | 0.0237 | 0.0430 | 0.0769 | 0.4679 |
| large | `[2.6]` | 1523 | 0.003621 | 0.0058 | 0.0101 | 0.0307 | 0.0838 | 0.1554 | 0.6774 |

### Ising interpretation

Paper bandwidth is best overall.

Feynman version:

> The Ising target has structured correlations. The paper bandwidth is the best lens for learning those correlations across many subset sizes. Large bandwidth gets the easy low-order checks right, but it loses more detail as `k` grows.

Uniform baseline reference:

| k | uniform mean TV |
|---:|---:|
| 1 | 0.0058 |
| 2 | 0.2096 |
| 4 | 0.3506 |
| 6 | 0.4420 |
| 8 | 0.5272 |
| 16 | 0.9622 |

Why the uniform baseline matters:

> Uniform is the "knows nothing" model. If learned TV is much lower than uniform TV, the IQP model learned real target structure.

The learned Ising models beat uniform strongly at every order.

## 8_blobs exact control

All three Blobs controls completed. The paper bandwidth finished at 5068 iterations and gives the best full-distribution error among the three settings.

| bandwidth | sigma | iters run | final loss | M1 | M2 | M4 | M6 | M8 | M16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | `[0.3]` | 3837 | 0.003231 | 0.0739 | 0.1277 | 0.2826 | 0.3678 | 0.4537 | 0.6662 |
| paper | `[0.6, 1.3]` | 5068 | 0.006447 | 0.0249 | 0.0579 | 0.1572 | 0.2406 | 0.3038 | 0.5300 |
| large | `[2.6]` | 1875 | 0.001673 | 0.0053 | 0.0431 | 0.1504 | 0.2715 | 0.3555 | 0.6189 |

### Blobs interpretation

The Blobs result is more nuanced than a single winner.

Feynman version:

> Blobs is a multi-peak target. The model needs to learn where the peaks are and how probability spreads around them. Large bandwidth is very good at the easiest checks, like `M1` and `M2`, because smoothing helps the model match broad structure. But as the test asks for larger groups of variables, the paper bandwidth is better: it keeps enough detail to improve `M6`, `M8`, and the full `M16` distribution.

The key comparison:

- `large` is best at `M1`, `M2`, and slightly best at `M4`.
- `paper` is best at `M6`, `M8`, and `M16`.
- `small` is worst across the reported Blobs marginals.

Uniform baseline reference:

| k | uniform mean TV |
|---:|---:|
| 1 | 0.1275 |
| 2 | 0.2220 |
| 4 | 0.4970 |
| 6 | 0.6536 |
| 8 | 0.7209 |
| 16 | 0.9868 |

All three Blobs runs beat uniform, but high-order errors remain substantial. The useful lesson is that the bandwidth affects which part of the distribution is learned best: broad low-order structure versus richer high-order structure.

## Added control metric: marginal L2 squared / MMD-style error

My supervisor pointed out that TV distance is not the only way to measure marginal error. We also computed a squared-error version for the exact `n = 16` controls.

For each subset `S`, instead of computing TV,

```text
TV(p_S, q_S) = 0.5 * sum_x |p_S(x) - q_S(x)|
```

we also compute the discrete integrated squared error:

```text
L2^2(p_S, q_S) = sum_x (p_S(x) - q_S(x))^2
```

Feynman version:

> TV asks, "How much probability mass is in the wrong place?" L2 squared asks, "How large are the squared bin-by-bin mistakes?" Big mistakes get punished more strongly because they are squared.

This is MMD-like in the sense that it is a squared discrepancy between distributions. On these finite binary marginals, the integral becomes a sum over all bit patterns in the marginal.

Important caution:

> Compare L2 squared mainly within the same `k`. As `k` changes, the number of bins changes from `2^k`, so the scale of the squared sum can change for mechanical reasons.

### 2D_ising L2 squared marginals

Smaller is better. These use the same exact `q_theta`, target samples, subset cap, and seed as the TV tables above.

| bandwidth | M1 | M2 | M4 | M6 | M8 | M16 |
|---|---:|---:|---:|---:|---:|---:|
| small | 9.09e-05 | 0.001101 | 0.001715 | 0.001242 | 0.000788 | 0.000212 |
| paper | 9.09e-05 | 0.000165 | 0.000223 | 0.000208 | 0.000198 | 0.000209 |
| large | 9.09e-05 | 0.000158 | 0.000352 | 0.000982 | 0.002296 | 0.011765 |
| uniform baseline | 9.09e-05 | 0.044974 | 0.072098 | 0.058676 | 0.042475 | 0.012284 |

Interpretation:

> The L2 squared view supports the same Ising conclusion as TV. Large is slightly best at `M2`, but it degrades badly at higher order. Paper is the best all-around setting, especially once the marginals ask for more than pairwise structure.

### 8_blobs L2 squared marginals

| bandwidth | M1 | M2 | M4 | M6 | M8 | M16 |
|---|---:|---:|---:|---:|---:|---:|
| small | 0.018076 | 0.026222 | 0.028412 | 0.022476 | 0.015673 | 0.002995 |
| paper | 0.001725 | 0.004970 | 0.008956 | 0.009943 | 0.008432 | 0.002553 |
| large | 8.55e-05 | 0.002694 | 0.009089 | 0.014175 | 0.015235 | 0.010780 |
| uniform baseline | 0.057834 | 0.080454 | 0.087327 | 0.072737 | 0.056907 | 0.025291 |

Interpretation:

> The L2 squared view also supports the Blobs TV conclusion. Large is best at the easiest low-order checks, especially `M1` and `M2`. Paper becomes better at the harder checks: `M4` is essentially tied, and paper clearly wins at `M6`, `M8`, and `M16`.

The practical takeaway from adding L2 squared:

> The conclusion is not an artifact of TV distance. When we switch to a squared-error marginal metric, the same bandwidth story remains: paper is the strongest high-order control setting, while large can look best on low-order smoothed structure.

## Main scientific readout

### What we can confidently say now

1. Bandwidth strongly affects marginal error.
2. On native genomic data, small bandwidth is clearly bad for marginal agreement.
3. Paper and large bandwidth are both strong for genomic, but paper is slightly better at higher `k`.
4. On Ising, paper bandwidth is the best all-order choice.
5. On Blobs, paper gives the best high-order and full-distribution error, while large is best at the easiest low-order checks.
6. The new L2 squared / MMD-style control metric supports the TV-based interpretation.
7. The small-bandwidth failure is consistent with either barren-plateau-like quantum trainability problems or classical overfitting/noisy fitting; this sweep cannot distinguish them.
8. Loss is not enough: the marginal curves reveal structure that final MMD loss does not.

### The simplest meeting phrasing

> We trained separate IQP-MMD models at small, paper, and large bandwidths. Then we checked whether their low-order and higher-order marginals match the target. The native genomic result is clear: small bandwidth fails badly on marginals, while paper and large work well. Paper is slightly better at higher-order genomic marginals, and the Ising and Blobs controls both support paper as the best high-order bandwidth. We checked this with TV distance and with an added L2 squared marginal error, and the story is the same. The small-bandwidth failure is a diagnostic warning sign, but not proof of barren plateaus: it could also be classical overfitting or noisy fine-detail fitting.

## Caveats

- For `genomic-805`, the marginal values are estimator-based, not exact, because exact enumeration over `2^805` states is impossible.
- For `genomic-805`, the `paper` run took much longer than `small` or `large`, so runtime is part of the practical tradeoff.
- For `2D_ising` and `8_blobs`, jobs stopped early due convergence, so iteration counts differ across bandwidths.
- This sweep identifies a bandwidth-dependent failure mode, but it does not separate quantum barren-plateau effects from classical overfitting. To make that claim, we would need diagnostics such as gradient-norm/gradient-variance curves, train-vs-validation marginal errors, multiple random seeds, and possibly loss-landscape probes.

## Next steps after the meeting

1. Use this completed sweep as the meeting result.
2. If there is time after the meeting, rerun the genomic marginal estimator with more sampled subsets to reduce uncertainty bars.
3. Add trainability diagnostics for the small-bandwidth genomic run: gradient norm, gradient variance across parameters/seeds, and loss trajectory shape.
4. Add classical generalization diagnostics: compare train vs held-out marginal errors for the same checkpoints.
5. If the supervisor wants a stronger bandwidth claim, run one or two intermediate bandwidths between paper and large for `genomic-805`.
