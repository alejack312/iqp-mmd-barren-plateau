---
title: Grid5000 Native iqp_mmd AC and Marginals
date: 2026-05-03
tags:
  - anti-concentration
  - marginals
  - grid5000
  - genomic
  - iqp-mmd
  - presentation
status: ready-to-present
---

# Grid5000 Native iqp_mmd AC and Marginals - 2026-05-03

> [!abstract] The one-minute version
> We reran the paper-style IQP-MMD experiments on Grid'5000 for `2D_ising`, `8_blobs`, and native `genomic-805`.
>
> The answer is: **the learned distributions are not anti-concentrated in the uniform / Porter-Thomas sense.** They are concentrated, because the targets are concentrated. That is not a bug in the run. It is the scientific result.
>
> The more useful finding is that the models can still match low-order marginals well. Ising is the cleanest case, genomic is surprisingly strong at low-order marginals, and blobs is the hardest case.

> [!success] Visual summary
> ![[grid5000_ac_marginals_summary_2026-05-03.png]]
>
> Left: anti-concentration scale as `log10(scaled_second_moment)`. Middle: exact all-order marginal TV curves for the two `n = 16` datasets. Right: native `genomic-805` low-order marginal TV estimates.

## How I Would Say This Out Loud

"I want to separate two questions that are easy to mix up.

First: are the learned distributions anti-concentrated, meaning spread over an exponential number of bitstrings like a typical random IQP circuit? No. They are much more concentrated than uniform.

Second: did the model learn the target structure? For that, the marginals are more informative. On Ising, the learned distribution tracks the target very closely across marginal orders. On blobs, it improves a lot over uniform but still misses a lot of the higher-order structure. On native genomic-805, exact enumeration is impossible, but the Pauli estimator says the low-order marginals are quite good through order 8.

So the clean takeaway is: IQP-MMD is learning concentrated empirical distributions, not preserving anti-concentration. The quantum-hardness style anti-concentration story does not survive this training setup in the naive form. But the model does learn useful low-order structure."

---

## Grid'5000 Context

### What Grid'5000 Is

Grid'5000 is a French national research testbed for running controlled experiments on real compute clusters. Instead of using my laptop, I reserve a machine on one of their sites, copy the project there, install the environment, and run the experiment on the reserved node.

How to say it:

"Grid'5000 is basically a research computing platform. The important point is that this was not run on my local machine. We used a dedicated Grid'5000 node at the Nancy site, so the native genomic experiment had enough CPU memory and wall time to finish."

Why we needed it:

- Local exact enumeration is fine for `n = 16`, but not for `n = 805`.
- Native genomic training builds a circuit with `805` qubits and `324415` gates.
- The genomic checkpoint alone is about `252 MB`.
- The full genomic job took about 8 hours, with about 6.5 hours spent in training and about 1.5 hours in estimator diagnostics.

### How We Used Grid'5000

The workflow was:

1. Package the project code, configs, scripts, and genomic CSV locally.
2. Copy the payload to Grid'5000 through the access node.
3. Connect to the Nancy frontend.
4. Reserve a compute node with OAR.
5. Create a Python environment on the cluster.
6. Run the three datasets.
7. Copy the result directory back to the local machine.

The actual target was the Nancy site:

| item | value |
|---|---|
| Grid site | Nancy |
| account | `ajackson` |
| frontend | `fnancy` / Nancy frontend |
| scheduler | OAR |
| genomic job id | `6378392` |
| assigned host | `gros-7.nancy.grid5000.fr` |
| job result | `Terminated`, exit code `0` |

There was one practical complication: the high-memory `abaca` resources were not available with this account's privilege level, so we used the default Nancy `gros` queue instead. That was still enough for the native genomic run.

The genomic batch job ran this command through a small shell wrapper:

```bash
python scripts/grid5000_run_iqp_mmd_ac.py \
  --dataset genomic-805 \
  --iters 1000 \
  --out-dir results/grid5000_iqp_mmd_ac
```

The wrapper also set:

```bash
export PYTHONPATH="$PWD/src"
```

That mattered because the Grid runner imports the local `iqp_bp` package from `src/`.

How to say it:

"Operationally, Grid'5000 gave us a clean remote machine where the job could keep running even if my laptop disconnected. We submitted the native genomic run as a batch job through OAR. That is why the earlier overnight laptop restart did not matter once the proper batch job was submitted: the computation was happening on the Nancy node, not on my laptop."

The final local copy came from:

```powershell
scp -r ajackson@access.grid5000.fr:nancy/iqp-mmd-barren-plateau/results/grid5000_iqp_mmd_ac .\results\
```

That copied back the exact small-`n` results and the native `genomic-805` estimator artifacts.

---

## What Was Actually Run

All outputs are under:

```text
results/grid5000_iqp_mmd_ac/
```

| dataset | n | max_weight | spin_sym | iterations requested | evaluation path |
|---|---:|---:|:---:|---:|---|
| `2D_ising` | 16 | 4 | true | 10000 | exact `q_theta`, exact AC, all-order `M_k` |
| `8_blobs` | 16 | 6 | false | 10000 | exact `q_theta`, exact AC, all-order `M_k` |
| `genomic-805` | 805 | 2 | false | 1000 | Pauli estimator AC, low-order marginal estimator |

Important detail:

- Ising and blobs are exact because `2^16 = 65536` outcomes is manageable.
- Genomic is native full `n = 805`. It is **not** the earlier local 16-SNP projection.
- Genomic exact enumeration was correctly skipped because `2^805` is impossible.

Training notes:

| dataset | training behavior |
|---|---|
| `2D_ising` | converged early at 1523/10000 steps |
| `8_blobs` | converged early at 4763/10000 steps |
| `genomic-805` | ran full 1000/1000 steps; did not hit convergence criterion |

The early stopping for Ising and blobs is normal: the trainer stopped when the convergence criterion fired. It does not mean the job failed.

### Is It Bad That Genomic Did Not Converge?

Short answer: **not automatically**.

The genomic log says:

```text
Training has not converged after 1000 steps
```

That means the run did **not** satisfy the trainer's formal stopping criterion by the final convergence check. It does **not** mean the job failed, crashed, or learned nothing.

The loss still dropped substantially:

| quantity | value |
|---|---:|
| initial loss | about `0.00449` |
| final loss | about `0.000591` |
| reduction | about `87%` |

And the post-training marginal diagnostics are meaningful:

| k | mean TV |
|---:|---:|
| 1 | `0.0051` |
| 2 | `0.0171` |
| 4 | `0.0468` |
| 8 | `0.1155` |

How to say it:

"The native genomic model did not trigger the formal convergence criterion within the 1000-step budget, but it trained substantially. The loss dropped by almost an order of magnitude, and the low-order marginals are good through order 8."

### What The Paper Did

In Sec. 8.1.3 of Recio-Armengol, Ahmed, and Bowles (`2503.02934v2`), the authors describe the IQP training strategy:

- Use the squared MMD loss.
- Estimate each training update with `|A| = |Z| = 1000`.
- Use the training data as `X`.
- Estimate gradients with automatic differentiation in JAX.
- Train with ADAM through IQPopt.
- Stop after a predetermined number of steps **or** when a convergence criterion is met.

For genomic, their setup was:

| item | paper setup |
|---|---:|
| qubits | `805` |
| parameters | `324415` |
| gate set | all-to-all single- and two-qubit IQP gates |
| bandwidths | `10.0`, `7.7`, `4.2` |
| implied Pauli weights | about `2`, `4`, `11` |
| dataset size | `5008` haplotypes |
| train/test split | test ratio `1/3` |

Their discussion says convergence was reached smoothly in a small number of steps for all problems, but the paper text does not give a detailed per-dataset convergence threshold. That matters because our log line is a specific IQPopt stopping-rule statement, while the paper's statement is a higher-level description of the training curves.

How to compare our run honestly:

> We matched the paper's native genomic scale and main training budget: `n = 805`, `324415` gates, `1000` steps, ADAM, `1000` Pauli operators, `1000` expectation samples, and the paper's genomic bandwidths. Our run completed the full step budget but did not trip the convergence flag. Since the loss fell strongly and the marginal diagnostics are good, this should be presented as a full-budget run that improved substantially but did not formally early-stop.

---

## Slide 1 - The Headline

> [!important] Headline
> The learned IQP-MMD distributions are **concentrated**, not anti-concentrated.

Use this wording:

"If anti-concentration means something like uniform or Porter-Thomas scale, then these trained models do not have it. Their probability mass is much more concentrated. But the targets are also concentrated, so the real question is not 'did the learned model stay anti-concentrated?' The real question is 'did it match the target's concentration and marginals?'"

Reference points:

| distribution type | scaled second moment |
|---|---:|
| uniform | `1` |
| Porter-Thomas-like | about `2` |
| concentrated empirical target | much larger than `2` |

The scalar is:

$$
2^n \sum_x p(x)^2
$$

Large values mean fewer effective outcomes. The useful mental model is:

$$
\text{effective support} = \frac{2^n}{\text{scaled second moment}}
$$

---

## Slide 2 - Anti-Concentration Results

| dataset | learned scaled second moment | target scaled second moment | what it means |
|---|---:|---:|---|
| `2D_ising` | `826.90` | `806.01` | learned matches target concentration very closely |
| `8_blobs` | `938.54` | `1658.44` | learned is concentrated, but less concentrated than target |
| `genomic-805` | about `10^238.43` | empirical target about `10^238.63` | learned is extremely concentrated, close to target scale |

How to say it:

"Uniform would be 1. Porter-Thomas would be order 2. So 800 or 1600 is not anti-concentrated in that sense. For genomic, the number is enormous because the space is enormous. We have 805 bits, but only about five thousand observed training rows. The empirical target itself is therefore extremely concentrated."

### Why Genomic Has A Huge Number

The genomic train CSV has:

| quantity | value |
|---|---:|
| shape | `(5008, 805)` |
| unique rows | `5008` |
| max duplicate count | `1` |
| empirical effective support | `5008` |
| empirical target log10 scaled second moment | `238.63` |
| learned estimator log10 scaled second moment | `238.43` |

Talk track:

"The giant genomic AC number is not automatically a numerical failure. The empirical data itself has support around five thousand rows inside a `2^805` space. So any distribution that actually resembles the empirical target will have a massive scaled second moment. In fact, the learned estimate is slightly more spread out than the empirical target, but it is in the same order-of-magnitude regime."

### Important Caveat About The Existing Flag

Some summaries contain:

```text
passes_second_moment_threshold = true
```

Do **not** present that as "passes anti-concentration." In this code path, the threshold is `>= 1`, and every probability distribution has scaled second moment at least `1`. The magnitude is what matters.

Use this sentence:

"The boolean second-moment flag is not the scientific result here. The number is the result."

---

## Why Compare Marginals To Uniform?

The uniform distribution is the "learned nothing" baseline.

If a model outputs every bitstring with equal probability, then it has no information about the target data. It has no Ising correlations, no blob modes, and no genomic SNP structure. So when we compare the learned model to the target, we also compare uniform to the same target to ask:

> "Is the trained IQP model actually closer to the data than a completely structureless sampler?"

How to say it:

"Uniform is not the goal. It is the sanity-check baseline. If the learned model is no better than uniform on a marginal, then training did not learn that marginal. If the learned model is much closer than uniform, then the circuit has captured real structure from the data."

For marginals, this is especially useful because the target may be concentrated or correlated in complicated ways. The uniform baseline tells us how much of the TV error is trivial:

- At `k = 1`, uniform only checks whether each bit has frequency near 50/50.
- At `k = 2`, uniform has no pairwise correlations.
- At high `k`, uniform has none of the target's joint structure.

So the uniform curve is the "zero-structure" curve. The learned curve should sit below it. The bigger the gap, the more structure the model learned.

Plainest version:

> Uniform is what we would get if the model just guessed randomly. We compare against it to show that the trained model is not just random guessing; it is actually closer to the target data.

---

## Slide 3 - Marginal Agreement: Ising

Lower mean TV is better. Uniform is the baseline: what happens if the model learned nothing except the sample space.

| k | learned mean TV | uniform mean TV | improvement over uniform |
|---:|---:|---:|---:|
| 1 | `0.0058` | `0.0058` | `0.0%` |
| 2 | `0.0103` | `0.2096` | `95.1%` |
| 3 | `0.0167` | `0.3163` | `94.7%` |
| 4 | `0.0237` | `0.3506` | `93.2%` |
| 6 | `0.0430` | `0.4420` | `90.3%` |
| 8 | `0.0769` | `0.5272` | `85.4%` |
| 10 | `0.1378` | `0.6018` | `77.1%` |
| 12 | `0.2440` | `0.7017` | `65.2%` |
| 16 | `0.4679` | `0.9622` | `51.4%` |

How to say it:

"Ising is the cleanest result. The learned model is dramatically better than uniform at every nontrivial order. At order 2, the mean TV error drops from about 0.21 to 0.01. Even at the full 16-bit marginal, it cuts the uniform error roughly in half. This is the strongest case that IQP-MMD is learning the target distribution rather than just smoothing everything away."

Short version:

> Ising: concentrated, but correctly concentrated; marginals are strong.

---

## Slide 4 - Marginal Agreement: 8 Blobs

| k | learned mean TV | uniform mean TV | improvement over uniform |
|---:|---:|---:|---:|
| 1 | `0.0248` | `0.1275` | `80.6%` |
| 2 | `0.0583` | `0.2220` | `73.7%` |
| 3 | `0.0996` | `0.3148` | `68.4%` |
| 4 | `0.1579` | `0.4970` | `68.2%` |
| 6 | `0.2422` | `0.6536` | `62.9%` |
| 8 | `0.3046` | `0.7209` | `57.8%` |
| 10 | `0.3753` | `0.8400` | `55.3%` |
| 12 | `0.4358` | `0.8911` | `51.1%` |
| 16 | `0.5323` | `0.9868` | `46.1%` |

How to say it:

"Blobs is harder. The learned distribution is still much better than uniform, so it is learning something. But the errors are much larger than Ising, especially as the marginal order goes up. That fits the intuition: blobs has a more global mode structure, so low-bandwidth MMD has a harder time enforcing the full high-order pattern."

Short version:

> Blobs: learned structure, but high-order mode structure is only partially captured.

---

## Slide 5 - Marginal Agreement: Native Genomic-805

For genomic, we cannot enumerate all outcomes, so these are estimator diagnostics over random subsets.

| k | learned vs target mean TV | max TV | subsets |
|---:|---:|---:|---:|
| 1 | `0.0051` | `0.0273` | 128 |
| 2 | `0.0171` | `0.1219` | 128 |
| 3 | `0.0297` | `0.2210` | 128 |
| 4 | `0.0468` | `0.1431` | 128 |
| 6 | `0.0776` | `0.2046` | 128 |
| 8 | `0.1155` | `0.2266` | 128 |

How to say it:

"Genomic is the full native 805-bit run. We cannot make exact claims about the whole distribution because exact enumeration is impossible, but the low-order marginals look good. Order 1 is around half a percent mean TV. Order 2 is about 1.7 percent. By order 8 it is about 11.5 percent. That is a smooth degradation, not a collapse."

Short version:

> Genomic: full native `n = 805`; low-order marginals are strong through `k = 8`.

---

## Slide 6 - The Scientific Interpretation

Use this as the conclusion:

"The conclusion is not that IQP-MMD preserves anti-concentration. It does not, at least not on these trained empirical targets. The learned models are concentrated because the data is concentrated.

The better conclusion is that IQP-MMD can learn low-order structure while producing a distribution whose concentration scale is close to the target. That is good for generative modeling, but it weakens the naive quantum-hardness story, because the usual anti-concentration hypothesis is not what survives training."

### Dataset-by-dataset takeaway

| dataset | takeaway |
|---|---|
| `2D_ising` | best overall fit; learned concentration and marginals closely track target |
| `8_blobs` | learns low-order and some global structure, but misses substantial high-order mode detail |
| `genomic-805` | native large-n run completed; low-order marginals are good, AC estimator says target-scale concentration |

### The clean thesis sentence

> IQP-MMD learns concentrated empirical structure; it does not preserve anti-concentration as a generic hardness property.

---

## Likely Supervisor Questions

### "So are the learned distributions anti-concentrated?"

Answer:

"No, not in the usual uniform or Porter-Thomas sense. Their scaled second moments are far above constant scale. But that is expected because the targets are also concentrated. The more relevant comparison is learned versus target, not learned versus uniform."

### "Is that bad?"

Answer:

"It depends what claim we are making. It is bad for a simple quantum-hardness story that relies on anti-concentration after training. It is not bad for generative modeling if the target itself is concentrated. For modeling, matching the target concentration and marginals is the point."

### "Why is genomic so enormous?"

Answer:

"Because `2^805` is astronomically large, and the empirical target has only `5008` observed rows. The target effective support is around `5008`, so when we scale by the full `2^805` space, the second moment is necessarily about `10^238`."

### "Can we trust genomic if it is estimator-based?"

Answer:

"We should present it as estimator evidence, not exact enumeration. The exact route is impossible at `n = 805`. The estimator used 2000 Pauli samples and 2000 expectation samples, and its relative standard error on the AC estimate is about 5.5 percent. The low-order marginal estimates are direct random-subset diagnostics through `k = 8`."

### "What is the strongest result?"

Answer:

"Ising is the strongest exact result. It matches the target concentration almost exactly and improves marginal TV dramatically over uniform across every order."

### "What is the weakest result?"

Answer:

"Blobs is the weakest. It is still much better than uniform, but high-order errors remain large. That suggests the eight-mode structure contains global information that the training setup only partially captures."

### "What should we do next?"

Answer:

"The next useful experiment is not just another AC scalar. It is a target power spectrum or per-order Pauli mismatch plot, so we can say which orders the data actually contains and which orders training learns."

---

## Source Artifacts

Local files copied back from Grid'5000:

```text
results/grid5000_iqp_mmd_ac/2D_ising/2D_ising_n16_iters10000_seed666_summary.json
results/grid5000_iqp_mmd_ac/2D_ising/2D_ising_n16_iters10000_seed666_summary.png
results/grid5000_iqp_mmd_ac/8_blobs/8_blobs_n16_iters10000_seed666_summary.json
results/grid5000_iqp_mmd_ac/8_blobs/8_blobs_n16_iters10000_seed666_summary.png
results/grid5000_iqp_mmd_ac/genomic-805/genomic-805_summary.json
results/grid5000_iqp_mmd_ac/genomic-805/estimator.json
results/grid5000_iqp_mmd_ac/genomic-805/checkpoint.npz
results/grid5000_iqp_mmd_ac/genomic-805/losses.csv
```

Grid run facts:

| item | value |
|---|---|
| Grid site | Nancy |
| genomic job id | `6378392` |
| genomic host | `gros-7.nancy.grid5000.fr` |
| genomic job state | `Terminated`, exit code `0` |
| genomic wall time | about 8 hours |
| genomic training time | about 6.5 hours |
| genomic estimator time | about 1.5 hours |

---

## Related Notes

- [[Anti-Concentration]]
- [[Anti-Concentration vs Marginal Agreement]]
- [[Pauli Estimator Scale Up 2026-04-24]]
- [[Genomic iqp_mmd AC Investigation 2026-04-29]]
- [[iqp_mmd AC Investigation 2026-04-23]]
- [[Bandwidth Marginals]]
- [[Kernel Spectral Decomposition]]
- [[Learning Task]]
