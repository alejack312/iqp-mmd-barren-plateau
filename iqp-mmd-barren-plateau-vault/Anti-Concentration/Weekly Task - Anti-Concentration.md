---
title: Weekly Task — Anti-Concentration
aliases:
  - Anti-Concentration Task
  - This Week
tags:
  - task
  - weekly
  - anti-concentration
  - presentation
---

# Weekly Task — Anti-Concentration

> [!important] The Task (one paragraph)
> Start running the code. Next week, present what **anti-concentration** is, and explain how you would write a function that checks whether a probability distribution is anti-concentrated. Reference paper: [arXiv:2512.24801](https://arxiv.org/abs/2512.24801). The goal after that is to take output distributions from **trained** IQP circuits and check whether they stay anti-concentrated after training.

This page is the one-stop prep doc. Read it in order. Every section is written as simply as possible. If someone at the meeting asks "what about X?", the [[#Likely Questions|FAQ]] at the bottom probably has it.

---

## 1. What Is Anti-Concentration, In One Sentence?

> **Anti-concentration means "the probability isn't piled up on a few outcomes — it's spread out across many of them."**

That's it. The technical definitions just make "spread out" precise.

### A Physical Picture

Imagine you have $N = 2^n$ boxes (one per possible $n$-bit string) and you drop a bag of sand of total weight 1 into them. Three extreme cases:

| Case | How the sand sits | Anti-concentrated? |
|---|---|---|
| **Uniform** | Every box gets exactly $1/N$ of the sand | ✅ Yes — maximally spread |
| **Delta** | One box gets **all** the sand, the rest get 0 | ❌ No — maximally concentrated |
| **Half and half** | A constant fraction of the boxes each hold roughly $1/N$; the rest hold 0 | ✅ Yes — still "enough" spread |

The key insight: anti-concentration does **not** require uniformity. It only requires that a **constant fraction** of the boxes have **at least roughly the uniform share** of sand.

> [!tip] Rule of thumb
> **Concentrated** = "a few outcomes hog all the probability."
> **Anti-concentrated** = "a lot of outcomes each have a fair share."

---

## 2. Why Do We Care? (The Quantum Advantage Story)

This is the part that matters for the meeting, because the "why" is the whole reason the project cares about anti-concentration.

### The Promise of IQP

[[IQP Circuits|IQP circuits]] are believed to be **classically hard to sample from**. Under plausible complexity assumptions (Bremner–Jozsa–Shepherd), efficiently simulating the output distribution of a generic IQP circuit would collapse the polynomial hierarchy.

Translation: **if you can draw typical samples from an IQP circuit output, you're doing something a classical computer provably can't do fast**. That's the whole point of using IQP circuits as generative models.

### The Catch

The hardness proofs need **anti-concentration as a hypothesis**. Roughly: the proof says "sampling is hard *provided* the distribution is spread out enough that the typical output carries meaningful probability mass."

If after training the IQP circuit collapses into a spiky distribution (a few outcomes hold all the mass), a classical computer can just **guess the spike** — no quantum advantage left.

### The Concrete Worry in Our Project

We're using the [[Project Overview|"train on classical, deploy on quantum"]] paradigm:

1. **Training** runs classically via the [[IQP Expectation|classical IQP estimator]] and the [[MMD Loss|MMD² loss]].
2. **Inference** means sampling from the trained circuit on a real quantum device.

For this to be worthwhile, **the trained distribution must still be anti-concentrated after training**. If training pushes $\theta$ into a corner where the distribution becomes spiky, the quantum deploy step loses its motivation.

> [!important] The core question of your task
> After training, is the learned output distribution $q_\theta(x)$ still anti-concentrated?
> If yes → quantum sampling advantage is (plausibly) preserved.
> If no → we trained ourselves out of the regime where the whole IQP idea makes sense.

See [[IQP Classical Sampling]] and [[Anti-Concentration]] for the longer writeups.

---

## 3. The Two Definitions From the Paper

The reference paper ([arXiv:2512.24801](https://arxiv.org/abs/2512.24801)) gives **two equivalent definitions**. You should know both cold because supervisors love asking "what's the difference?".

Let $p$ be a probability distribution over $\{0,1\}^n$ and let $x \in \{0,1\}^n$ be drawn uniformly.

### Definition A — Threshold Form

$$
\Pr_{x \sim U}\!\left[p(x) \ge \frac{\alpha}{2^n}\right] \ge \beta
$$

**In plain English:** a constant fraction $\beta$ of the $2^n$ possible outcomes have probability at least $\alpha$ times the uniform baseline $1/2^n$. Both $\alpha$ and $\beta$ are constants **independent of $n$**.

- $\alpha = 1$ is the natural choice: "at least the uniform share"
- $\alpha = 0.5$ is the most generous choice: "at least half the uniform share"
- $\alpha = 2$ is stricter: "at least twice the uniform share"

For a **uniform** distribution, every $x$ has $p(x) = 1/2^n$ exactly, so the fraction above is **1 whenever $\alpha \le 1$** and **0 whenever $\alpha > 1$**.

For a **delta** distribution ($p(x^*) = 1$), only **one** outcome out of $2^n$ has probability above the threshold, so the fraction is $1/2^n$ — which shrinks to 0 as $n$ grows. **Not anti-concentrated.**

### Definition B — Second-Moment Form

$$
2^{2n} \cdot \mathbb{E}_{x \sim U}[p(x)^2] \ge \beta' > 1
$$

**In plain English:** the "scaled second moment" has to exceed 1 by a constant.

Let's unpack it:

- $\mathbb{E}_x[p(x)^2]$ averages $p(x)^2$ over uniformly random $x$
- For the **uniform distribution**, $p(x) = 2^{-n}$ so $\mathbb{E}_x[p(x)^2] = 2^{-2n}$, and $2^{2n} \cdot 2^{-2n} = 1$
- Anti-concentration says this quantity must **exceed 1 by a constant**

So Definition B is really saying: "the distribution is at least as spread out as uniform, up to a constant factor." It's a single scalar, super clean.

### Why They're Equivalent (Intuition)

They both measure "is the distribution close to the uniform?" but from different angles:

- **Definition A** is a **counting statement** — how many outcomes are heavy enough?
- **Definition B** is a **norm statement** — how flat is the distribution?

The second moment $\sum_x p(x)^2$ (also called **collision probability** — the chance that two independent draws collide) is minimized by the uniform distribution and blows up for spiky ones. So "second moment close to uniform" implies "mass is spread" implies "many outcomes are above the threshold."

> [!tip] The easy way to remember it
> - Delta distribution: threshold check → 0. Second moment → $2^n$ (huge). **Fails**.
> - Uniform: threshold check → 1. Second moment → 1. **Passes** (barely — it's the boundary).
> - Typical "spread" distribution: threshold check → constant. Second moment → constant $> 1$. **Passes**.

---

## 4. The Exact Finite-$n$ Identities

This is the math the code actually computes. From [`docs/technical/anti-concentration.md`](../docs/technical/anti-concentration.md):

$$
\mathbb{E}_{x \sim U}[p(x)^2] = 2^{-n} \sum_x p(x)^2
$$

So:

$$
2^{2n} \cdot \mathbb{E}_x[p(x)^2] = 2^n \sum_x p(x)^2
$$

This is the **exact deterministic scalar** used in the code. The `scaled_second_moment` field that the checker reports is literally this number.

Similarly, the threshold statistic for finite $n$ becomes:

$$
\hat{\beta}(\alpha) = 2^{-n} \cdot \left|\{x : p(x) \ge \alpha \cdot 2^{-n}\}\right|
$$

i.e. "what fraction of the $2^n$ bitstrings have probability above $\alpha/2^n$?" It's literally a count divided by $2^n$.

> [!info] The two numbers the code reports
> - `scaled_second_moment = 2^n * sum_x p(x)^2`
> - `beta_hat(alpha) = fraction of bitstrings with p(x) >= alpha/2^n`
>
> **If `scaled_second_moment` is close to 1 and `beta_hat(1.0)` is a reasonable fraction (say ≥ 0.25), the distribution is anti-concentrated.**

See [[Anti-Concentration]] for the detailed writeup.

---

## 5. How You'd Write the Function (The Simple Version)

Here's the pseudocode anyone can understand:

```python
def check_anti_concentration(p, alpha=1.0, beta_min=0.25):
    """
    p: a probability vector of length 2^n (sums to 1).
    alpha: threshold multiplier (default 1.0 = uniform baseline).
    beta_min: minimum acceptable fraction for 'passing' (default 0.25).

    Returns: a dict with the two diagnostics and pass/fail booleans.
    """
    n = int(round(log2(len(p))))          # recover qubit count from length
    uniform_baseline = 2 ** (-n)          # i.e. 1 / 2^n

    # Diagnostic 1 — threshold form
    threshold = alpha * uniform_baseline
    beta_hat = mean(p >= threshold)       # fraction of x with p(x) above threshold
    passes_threshold = beta_hat >= beta_min

    # Diagnostic 2 — second-moment form
    collision_probability = sum(p ** 2)
    scaled_second_moment = (2 ** n) * collision_probability
    passes_second_moment = scaled_second_moment >= 1.0   # boundary case = uniform

    return {
        "n": n,
        "beta_hat": beta_hat,
        "scaled_second_moment": scaled_second_moment,
        "passes_threshold": passes_threshold,
        "passes_second_moment": passes_second_moment,
    }
```

That's the entire function. Three lines of actual math:

1. `beta_hat = mean(p >= alpha / 2**n)`
2. `scaled_second_moment = 2**n * sum(p**2)`
3. Compare both against thresholds.

### Supporting Diagnostics Worth Reporting

Even though the two above are the core, the current code also reports:

| Diagnostic                | Formula                           | What it tells you                                                |
| ------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| `collision_probability`   | $\sum_x p(x)^2$                   | Probability two i.i.d. draws are equal                           |
| `max_probability_scaled`  | $2^n \max_x p(x)$                 | How big the biggest bin is, relative to uniform                  |
| `effective_support`       | $1 / \sum_x p(x)^2$               | "How many outcomes is the distribution effectively spread over?" |
| `effective_support_ratio` | $\text{effective\_support} / 2^n$ | Same but as a fraction of the full state space                   |

`effective_support` is especially intuitive: a uniform distribution has effective support $2^n$; a delta has effective support 1. Anything in between tells you roughly how many bins the mass is actually using.

---

## 6. Where the Probability Vector Comes From

Writing the check is the easy half. The harder half is getting $p(x)$ for a trained IQP circuit in the first place.

### Path 1 — Exact, small $n$

For an IQP circuit

$$
|\psi(\theta)\rangle = H^{\otimes n} \cdot D_\theta \cdot H^{\otimes n}|0^n\rangle, \quad D_\theta|z\rangle = e^{-i\phi(z)}|z\rangle
$$

with diagonal phase $\phi(z) = \sum_j \theta_j (-1)^{z \cdot g_j}$, you can compute $p(x)$ **exactly** in four steps:

1. Enumerate every $z \in \{0,1\}^n$ (costs $2^n$)
2. Build the phase vector $d(z) = e^{-i\phi(z)}$
3. Apply the [[Walsh-Hadamard Transform|Walsh-Hadamard transform]] → amplitudes $a(x)$
4. Square to get $p(x) = |a(x)|^2$

This is exactly what [`IQPModel.probability_vector_exact`](../src/iqp_bp/iqp/model.py) already does. It's capped at $n \le 20$ because it's exponential in $n$.

> [!warning] Why the small-$n$ cap matters for your presentation
> The exact path **only works up to ~20 qubits**. Above that, you can't enumerate $\{0,1\}^n$. For larger circuits you'd need sample-based diagnostics — which give you an **empirical histogram**, not the true distribution. That's a valid secondary check, but it's weaker evidence because of sampling noise.

### Path 2 — Sample-based, large $n$

For large $n$, you sample bitstrings from the trained circuit (running it on a simulator or real quantum hardware), build a histogram, and apply the same two diagnostics on the histogram:

```python
samples = sample_iqp(trained_theta, G, num_samples=100000)
p_empirical = build_histogram(samples)  # but p_empirical is NOT the true p
result = check_anti_concentration(p_empirical)
```

Caveats:

- The histogram is missing all the $x$ you didn't happen to sample
- `beta_hat` is biased downward (you underestimate the support)
- `scaled_second_moment` is biased upward (empty bins look artificially empty)
- You need a **lot** of samples for the estimate to converge at large $n$ — but you can't get a lot of samples from hardware cheaply

**For your task this week, stick to Path 1 (exact, small $n$).** That's where the deterministic validation pipeline lives. The [[Validation Runner]] already handles it.

---

## 7. What's Already in the Code

> [!tip] Good news
> The checker function **already exists**. Your task is to understand it, know how to call it, and be able to talk through what it does. You are not being asked to write it from scratch.

### Key files

| File | Purpose |
|---|---|
| [`src/iqp_bp/iqp/model.py`](../src/iqp_bp/iqp/model.py) | `IQPModel.probability_vector_exact()` — computes $p(x)$ via FWHT |
| [`src/iqp_bp/experiments/run_validation.py`](../src/iqp_bp/experiments/run_validation.py) | `check_anti_concentration(probabilities, …)` — the actual checker |
| [`tests/test_anti_concentration.py`](../tests/test_anti_concentration.py) | Unit tests, including uniform + delta sanity checks |
| [`docs/technical/anti-concentration.md`](../docs/technical/anti-concentration.md) | Locked definitions and repo decisions |
| [`docs/papers/2512.24801v1.pdf`](../docs/papers/2512.24801v1.pdf) | The reference paper |

### The existing signature

```python
def check_anti_concentration(
    probabilities,                          # length 2**n
    *,
    alphas=(0.5, 1.0, 2.0),                # grid of threshold multipliers
    primary_alpha=1.0,                      # the alpha used for pass/fail
    beta_min=0.25,                          # minimum acceptable beta_hat
    second_moment_threshold=1.0,            # minimum acceptable second moment
    atol=1e-12,                             # normalization tolerance
) -> dict:
```

It returns a dict with every diagnostic you could possibly want:

- `n`, `num_outcomes`, `uniform_probability`
- `collision_probability`, `scaled_second_moment`, `passes_second_moment_threshold`
- `max_probability`, `max_probability_scaled`
- `effective_support`, `effective_support_ratio`
- `primary_alpha`, `primary_beta_hat`, `passes_primary_threshold`
- `threshold_checks` — list of one entry per $\alpha$ in the grid, with `beta_hat` and `passes_beta_threshold`

### Minimal example you can run at the meeting

```python
import numpy as np
from iqp_bp.experiments import check_anti_concentration

# Case 1: uniform (boundary case — passes trivially)
p_uniform = np.full(8, 1 / 8)
print(check_anti_concentration(p_uniform)["passes_primary_threshold"])   # True

# Case 2: delta (maximally concentrated — should fail)
p_delta = np.zeros(8); p_delta[0] = 1.0
print(check_anti_concentration(p_delta)["passes_primary_threshold"])     # False

# Case 3: a real IQP model
from iqp_bp.iqp.model import IQPModel
from iqp_bp.hypergraph.families import make_hypergraph

G = make_hypergraph("lattice", n=4, m=4, dimension=2, range_=1)
theta = np.random.default_rng(0).normal(0, 0.1, size=G.shape[0])
model = IQPModel(G=G, theta=theta)
p_iqp = model.probability_vector_exact()
print(check_anti_concentration(p_iqp))
```

See [[Validation Runner]] and [[Anti-Concentration]] for deeper details.

---

## 8. What To Actually Do This Week

> [!todo] Concrete checklist
> - [x] **Install and run the pipeline**. Try `iqp-bp run-validation configs/experiments/validation.yaml` or the `scaling_ac_smoke` config with $n = 4$ or 9. — ran `PYTHONPATH=src python -m iqp_bp.cli run-validation configs/experiments/validation.yaml`; artifacts in `results/validation/complete_graph_n6_seed7.{json,csv,png}`. Note: `pip install -e .` fails because `iqpopt` is not on PyPI, so use `PYTHONPATH=src` instead.
> - [x] **Reproduce the unit tests** in `tests/test_anti_concentration.py` — run `pytest tests/test_anti_concentration.py -v`. Read each test and make sure you understand *why* it passes. — 4/4 pass via `PYTHONPATH=src python -m pytest tests/test_anti_concentration.py -v`.
> - [x] **Call `probability_vector_exact` yourself** on a small IQP model you build manually. Plot $p(x)$ as a bar chart for $n = 3$ or 4 so you can literally see the shape. — see `scripts/explore_anti_concentration.py`; bar chart at `results/exploration/probability_vectors_n3.png`.
> - [x] **Feed the vector into `check_anti_concentration`** and print the dict. Match each field to one of the paper definitions. — same script prints every field with an inline mapping; raw dicts at `results/exploration/anti_concentration_n3.json`.
> - [x] **Try at least three cases** — uniform, delta, and an IQP model with random $\theta$ — and compare their diagnostics side by side. — uniform / delta / `complete_graph` n=3 random-θ all in the same script and the side-by-side table on Slide 4.
> - [x] **Read the paper** (2512.24801). At minimum: the definitions and the physical intuition. The proofs are not needed for next week. — definitions are §II.B Eqs. (2)–(3) on page 3 (threshold form and second-moment form); intuition: $p_C(x) \in \Theta(2^{-n})$, Porter–Thomas as paradigmatic example.
> - [x] **Draft 3–5 slides** (or a whiteboard sketch) covering: (1) what anti-concentration is, (2) the two definitions, (3) the function you'd write, (4) an example run, (5) how this ties to IQP sampling hardness. — [[Presentation - Anti-Concentration]].

---

## 9. Likely Questions

These are the questions that could land on you at the meeting. Having a one-line answer for each is usually enough — longer answers are in the linked pages.

### Conceptual

**Q: What is anti-concentration in one sentence?**
A: A probability distribution over $\{0,1\}^n$ is anti-concentrated if a constant fraction of the $2^n$ outcomes each carry at least a constant multiple of the uniform probability $1/2^n$. In plain words: the mass is spread out, not piled up.

**Q: Why do we care?**
A: IQP circuit sampling hardness proofs assume anti-concentration. If training pushes the learned distribution out of that regime, the quantum sampling advantage disappears — a classical computer could just guess the big spikes. See [[IQP Classical Sampling]].

**Q: What's the difference between the two definitions?**
A: The threshold form counts "how many bitstrings are above a uniform multiple of probability mass," the second-moment form measures "how flat is the distribution, normalized to uniform." They're equivalent up to constants. The second moment is one scalar (compact), the threshold form is interpretable (you get an actual fraction).

**Q: Is uniform anti-concentrated?**
A: Yes — it's the **boundary case**. `scaled_second_moment = 1` exactly, and `beta_hat(alpha) = 1` for any $\alpha \le 1$. Uniform barely satisfies the definition with $\beta' > 1$ being a strict inequality, but in practice the code uses $\ge 1$ as the pass condition.

**Q: Is a delta distribution anti-concentrated?**
A: No. Only one outcome is above any positive threshold, so `beta_hat` = $1/2^n$ → 0. And `scaled_second_moment = 2^n`, which **passes** the second-moment threshold trivially but **fails** the threshold form. This is actually the clearest example of why reporting both diagnostics matters.

**Q: Wait, a delta "passes" the second-moment check?**
A: The second moment is $2^n \cdot 1 = 2^n$, which is certainly $\ge 1$. So a naive second-moment check alone is misleading. The paper's scalar is "$\ge \beta' > 1$" but passing that says "not worse than uniform," not "actually anti-concentrated." This is why the code reports **both** diagnostics — the threshold form is the human-interpretable sanity check.

**Q: How does anti-concentration relate to barren plateaus?**
A: They're **separate questions**. Barren plateau = "can I train the circuit at all?" Anti-concentration = "is the trained distribution still quantum-useful?" A model can pass one and fail the other. See [[Anti-Concentration#Why the Split Matters]].

**Q: How does anti-concentration relate to classical simulability?**
A: If $p$ is very concentrated, you don't need to run the circuit — just guess the big outcomes. If $p$ is anti-concentrated, typical samples each have small probability, and recovering them is harder. This is the mechanism behind IQP hardness proofs. See [[IQP Classical Sampling]].

**Q: What are typical values of $\alpha$ and $\beta$ in the paper?**
A: The paper takes them as "some positive constants independent of $n$." In the repo, the default `alphas` grid is $(0.5, 1.0, 2.0)$ with `primary_alpha = 1.0`, `beta_min = 0.25`, `second_moment_threshold = 1.0`. These are threshold choices — we report diagnostics at every alpha and let the viewer decide what's "enough."

### Implementation

**Q: How do you get $p(x)$ from a trained circuit?**
A: For small $n$, you use the exact path: enumerate $z$, build the diagonal phase vector $d(z) = e^{-i\phi(z)}$, apply the Walsh-Hadamard transform, square. This is `IQPModel.probability_vector_exact()`. For large $n$ you fall back to sample histograms, but those are biased and only give a secondary check.

**Q: What's the max $n$ you can check exactly?**
A: 20 in the default safety cap, but that's already $2^{20} \approx 10^6$ probabilities — you can push it higher if you have RAM. Realistically, the exact regime is $n \le 16$ for quick experiments.

**Q: What if the probability vector has tiny negative entries from floating-point roundoff?**
A: The code zeros anything with magnitude below `atol=1e-12` before validating normalization. See `_coerce_probability_vector`.

**Q: Do you need to sum over all $x$ or can you sample?**
A: For the exact path, yes, you sum over all $x$ — that's the point. For sample-based diagnostics, you use the empirical histogram, but the result is biased and noisy. The paper's definitions are over the full distribution.

**Q: Why does the scalar form use $2^n$ and not $2^{2n}$ in the code?**
A: Because $2^{2n} \cdot \mathbb{E}_x[p^2] = 2^{2n} \cdot 2^{-n} \sum_x p^2 = 2^n \sum_x p^2$. The $2^n$ version is what you actually compute — it's the paper's scaled second moment in its most compact form.

**Q: What's "collision probability"?**
A: $\sum_x p(x)^2$ is the probability that two independent draws from $p$ come out equal. It's the unscaled second moment. For uniform it's $1/2^n$. The "scaled second moment" is this times $2^n$.

**Q: What's "effective support"?**
A: $1 / \sum_x p(x)^2$. For uniform it equals $2^n$; for a delta it equals 1. Intuitively, it's the number of outcomes the distribution is "effectively" spread over. Great intuition-builder, not a formal anti-concentration metric.

**Q: Can the function be fooled?**
A: Kind of. A distribution that's uniform over half the space and zero on the other half has `scaled_second_moment = 2` (passes) and `beta_hat(1.0) = 0.5` (passes for beta_min ≤ 0.5). That's still anti-concentrated by the definition. But a distribution with 100 huge spikes out of $2^{20}$ bins can have a **large** second moment and still fail the threshold form — which is why reporting both is important.

### Project-Level

**Q: Is this part of the barren plateau study or something else?**
A: It's a **parallel track**. Barren plateau asks "can we train?", anti-concentration asks "did training produce something quantum-useful?". Both must pass for the "train on classical, deploy on quantum" paradigm to make sense.

**Q: What do you do after training a model?**
A: (1) Extract the trained parameters $\theta$ and generator matrix $G$. (2) Wrap them in `IQPModel`. (3) Call `probability_vector_exact()` for small $n$. (4) Feed into `check_anti_concentration`. (5) Save the JSON + CSV artifacts and generate plots. The [[Validation Runner]] already does all of this.

**Q: What's the relationship to the MMD loss?**
A: MMD compares expectation values $\langle Z_a \rangle$ between data and model — it's about **parity correlations**, not about **where mass sits**. A distribution can match all low-order parities while still being spiky, or match no parities while being spread out. So anti-concentration is a genuinely orthogonal check. See [[MMD Loss]].

**Q: Why run this on trained models, not random ones?**
A: Random IQP circuits with generic $\theta$ are typically anti-concentrated by construction — that's why the sampling hardness argument works in the first place. The open question is whether **training** pushes the parameters to a place where anti-concentration is lost. That's what the supervisor wants you to check.

**Q: Will you use any real data?**
A: Not yet. For the first pass, `product_bernoulli` or a small Ising target is plenty. Real datasets (like MNIST) are in the `iqp_mmd` package but aren't needed for this check.

**Q: How would you present a result?**
A: A plot of `beta_hat(alpha)` vs $\alpha$ on a log-x axis, with the `beta_min` threshold drawn as a horizontal line. A bar chart of $p(x)$ for small $n$ to visualize the shape. And the `scaled_second_moment` scalar printed next to it. The [[Validation Runner]] already emits all three.

---

## 10. TL;DR

1. **Anti-concentration** = probability distribution is spread out over $\{0,1\}^n$, not concentrated on a few outcomes.
2. **Why it matters** = the quantum sampling hardness of IQP circuits assumes this; without it, classical simulation catches up.
3. **Two equivalent definitions** = threshold form (count of bitstrings above $\alpha/2^n$) and second-moment form ($2^n \sum p^2 \ge \beta' > 1$).
4. **The function** = compute those two scalars and compare against thresholds. Already exists as `check_anti_concentration` in `run_validation.py`.
5. **Where $p(x)$ comes from** = `IQPModel.probability_vector_exact()` via the Walsh-Hadamard transform, for $n \le 20$.
6. **Your job this week** = run the code, walk through the math, be ready to explain it all and show a concrete example.

---

## Related

- [[Anti-Concentration]] — the full technical writeup inside this vault
- [[IQP Classical Sampling]] — the hardness backdrop that motivates caring about this
- [[IQP Model]] — where `probability_vector_exact` lives
- [[Walsh-Hadamard Transform]] — the FWHT that converts IQP phases to amplitudes
- [[Validation Runner]] — the runner that automates the whole pipeline
- [[MMD Loss]] — why anti-concentration is *not* the same as matching parities
- [[Barren Plateaus]] — the companion question about trainability
- [[References#Paper 2512.24801]] — the reference paper
- [[Research Questions]] — where this fits in the broader project
