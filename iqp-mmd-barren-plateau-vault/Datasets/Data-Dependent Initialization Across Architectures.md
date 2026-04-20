---
title: Data-Dependent Initialization Across Families
aliases:
  - Data-Dependent Initialization Across Architectures
tags:
  - theory
  - init
  - trainability
  - warm-start
  - literature-review
  - families
---

# Data-Dependent Initialization Across IQP Families

One concrete question: if we pick our IQP circuit parameters from the training data instead of random numbers, does the trick work the same way for every connectivity family this project sweeps? Short answer: yes. Longer answer: the *recipe* works for all of them, but what the recipe actually *does* looks very different depending on the family.

Everything below is grounded in the papers in [`docs/papers/`](../docs/papers/). See [[References]] for citations, [[Initialization Schemes]] for the I1/I2/I3 schemes the code implements, and [[Families MOC]] for the SMART four families.

---

## What you need to know first (plain English)

If you are not already deep in the IQP barren plateau literature, read this section. Every term the rest of the note uses is defined here, and I try to give you the picture before the formulas.

**IQP circuit.** A restricted kind of quantum circuit. Picture it like this: start every qubit in the same state, apply a bunch of phase rotations that each touch a specific group of qubits, then measure. The useful math fact is that the phase rotations all commute with each other. That means you can reason about them one at a time without worrying about the order they are applied in, which is a very unusual and convenient property. See [[IQP Circuits]] for the formal version.

**Parameters ($\theta$).** The knobs we turn during training. Each phase rotation has one knob. Together they form a vector $\theta \in \mathbb{R}^m$ with $m$ components, one per rotation.

**Generator matrix $G$.** A binary matrix of shape $(m, n)$ that says *which qubits each rotation touches*. Row $j$ is a string of 0s and 1s. If $G_{j,i} = 1$, qubit $i$ participates in rotation $j$; if $G_{j,i} = 0$, it does not. Picture a spreadsheet where columns are qubits, rows are rotations, and a cell is checked off when that rotation touches that qubit.

**Hamming weight of a row.** Just the number of 1s in the row. Weight 1 means a single-qubit rotation. Weight 2 means two qubits are entangled by that rotation. Weight 3 means a three-qubit term. The weight is telling you how "many-body" the interaction is.

**Family (as used in this project).** A rule for building $G$ as the number of qubits grows. "Product state" makes $G$ the identity matrix, so every rotation touches exactly one qubit. "2D lattice" puts rows on neighboring pairs on a grid. "Complete graph" puts rows on every possible pair of qubits. "Sparse Erdős–Rényi" flips a coin for each pair and only keeps the edges that come up heads. See [[Families MOC]].

**Initialization (or warm start).** The $\theta$ you pick at step 0, before any gradient descent. The classic choice is random, either uniform on $[-\pi, \pi]$ or a small Gaussian. "Data-dependent" means we set $\theta_0$ using statistics of the training data instead of rolling dice.

**Barren plateau.** The failure mode this whole line of work is trying to avoid. It means the gradient variance shrinks exponentially with the number of qubits, so optimizers cannot tell a good descent direction from noise. On a barren plateau, training just does not scale. See [[Barren Plateaus]].

**Parity expectation $\langle Z_{g_j}\rangle_p$.** For a generator $g_j$ and a data distribution $p$, this is the average value of $(-1)^{x \cdot g_j}$ over samples $x$ drawn from $p$. Two concrete cases make it less mysterious:

- If $g_j$ is a single 1 at position $i$, then $(-1)^{x \cdot g_j} = (-1)^{x_i}$, and the average is $1 - 2 P(x_i = 1)$. That is $+1$ when bit $i$ is usually 0 and $-1$ when it is usually 1.
- If $g_j$ is two 1s at positions $i, k$, then you are averaging $(-1)^{x_i + x_k}$, which is $+1$ when the two bits *agree* and $-1$ when they *disagree*. So the expectation is measuring how often the two bits agree across the dataset.

In short: the parity expectation is a single number that summarizes the data from the point of view of one generator. That number is what the project's current init recipe plugs straight into $\theta_j$.

**Covariance.** A standard stat measure. It is positive when two features move together, negative when they move in opposite directions, and near zero when they are unrelated. Recio-Armengol et al. 2025's recipe uses feature covariance as an initialization value for two-qubit rotations.

**"Model class" vs "family".** These are easy to mix up in the literature. A *model class* is a choice of parameterized model at the highest level (IQP vs energy-based model vs RBM vs HEA vs ...). A *family*, in this project, is a shape of the $G$ matrix inside a fixed model class. Recio-Armengol et al. 2025's famous claim about data-dependent init being an IQP strength is a statement about *model classes*. The question this note answers is one level finer: inside the IQP model class, does that strength travel to every family the project sweeps?

---

## TL;DR

Yes, data-dependent initialization is possible for every IQP connectivity family in the sweep. There is no family-level obstruction. Both the Recio-Armengol et al. 2025 recipe and the project's current lightweight I3 boil down to "compute a scalar from the data for each generator," and that scalar is well-defined no matter which family you are using.

What changes from family to family:

1. **Which data statistics the init consumes.** Product state uses only per-feature means. 2D lattice uses covariances between neighbors on the grid. Erdős–Rényi uses covariances on a random subset of pairs. Complete graph uses every pairwise covariance.
2. **How well the init describes the training data.** If you use the product state family with the Recio-Armengol et al. 2025 recipe and set every other parameter to zero, the model reproduces the training set's single-qubit marginals exactly. If you use the complete graph family, the init is instead trying to reproduce the full pairwise correlation structure. In between, the fit is limited by the architecture.
3. **Whether the init is rescuing training or just accelerating it.** Uniform init is expected to hit a barren plateau on the complete graph family, so data-dependent init has the most to gain there. Product state and lattice do not plateau under uniform init, so data-dependent init is more about starting closer to the answer than about saving training from collapsing.
4. **Whether the closed-form Recio-Armengol et al. 2025 recipe even applies.** All four SMART families have generators of Hamming weight 1 or 2, so the recipe applies to every row. The legacy families (`bounded_degree`, `dense`, `community`, `symmetric`) sometimes use weight-3 or higher rows, and Recio-Armengol et al. 2025's recipe has no closed-form case for those.

One thing worth getting straight up front. Recio-Armengol et al. 2025 says data-dependent init is a strength "that may not be generally present in other models." They mean other *model classes*, things like classical energy-based models and RBMs. Inside the IQP model class, the strength carries over to every family in the sweep without fuss.

---

## What "data-dependent init" means in practice

Two recipes are on the table.

### Recipe A: Recio-Armengol et al. 2025, marginal-matching

This is the recipe from Sec. 8.1.2 of Recio-Armengol et al. 2025. It splits by generator weight, and each case has a clean interpretation.

**Weight-1 generator on qubit $j$.** Set $\theta_j = \arcsin(\sqrt{\bar x_j})$, where $\bar x_j$ is the average value of feature $j$ in the training data. Why that specific formula? Because the single-qubit phase rotation, when every other parameter is zero, produces a Bernoulli distribution on qubit $j$ whose probability of being 1 is exactly $\bar x_j$. So the recipe is saying: "start as a factorized guess that already gets the single-qubit frequencies right." You can check the formula by working through $|\langle 1|R_x(\theta_j)|+\rangle|^2$ and seeing it equals $\sin^2(\theta_j)$; setting that equal to $\bar x_j$ gives the arcsin.

**Weight-2 generator on qubits $(j, k)$.** Set $\theta_{jk} \propto \mathrm{cov}(x_j, x_k)$, after mapping the features to $\pm 1$. The picture is simpler than the formula looks. In an IQP circuit, a two-body generator flips a pair of bits coherently. If two features tend to move together in the data, the rotation should fire harder. If they move in opposite directions, it should fire the other way. If they are uncorrelated, the initial angle is zero. Same slogan as the weight-1 case: "start as a guess that already gets the pair correlations right."

**Weight 3 or higher.** Recio-Armengol et al. 2025 has no closed form. They fall back to a random small-angle Gaussian with the standard deviation as a hyperparameter.

The reason this recipe works at all is the commuting property of IQP generators I mentioned in the setup. Because they commute, the output marginals factor cleanly when the other knobs are off, and you can read the intended meaning off the parameters one generator at a time. In an architecture without that property (like HEA) a single parameter is tangled up with every other parameter through the non-commuting algebra, and this clean interpretation evaporates.

### Recipe B: the project's lightweight I3

The current code does something different. Weight-agnostic. Simpler. From `src/iqp_bp/mmd/mixture.py:30` and `src/iqp_bp/experiments/run_scaling.py:274`:

```python
return scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
```

In math, this says: for every row $g_j$ of $G$, compute the parity expectation on the training data and scale it by a small constant.

$$
\theta_j = s \cdot \langle Z_{g_j}\rangle_p = s \cdot \mathbb{E}_{x \sim p_\text{data}}\big[(-1)^{x \cdot g_j}\big]
$$

The default is $s = 0.1$.

The important property is that this is *one formula* that works for any row, no matter its weight. You do not have to split weight-1 from weight-2 from weight-3. The formula is asking a single question ("how often does an even-vs-odd parity pattern show up under this generator in the training data?") and scaling the answer.

Recipe A is richer and has a cleaner interpretation. Recipe B is lighter and handles every possible $G$ uniformly. The two recipes do not agree numerically, even on the weight-1 and weight-2 cases they both handle, because they are computing different things. Recipe A uses a mean or a covariance; Recipe B uses a parity average. For the full story on why the project started with Recipe B see [[Initialization Schemes#I3 Data-Dependent]] and [[Implementation Choices#Why the init layer changed too]].

---

## Family by family: what the init actually does

All four SMART families have generators of Hamming weight 1 or 2 (see [[Families MOC]]). What changes is *which* rows show up, and that changes what the init is actually computing.

### Product state family

Every row of $G$ has weight 1, so $G$ is just the identity matrix. Each qubit gets its own single-qubit rotation, nothing entangles. See [[Product State Family]].

Under Recipe A, $\theta_j = \arcsin(\sqrt{\bar x_j})$, and the resulting IQP state reproduces the training set's single-qubit marginals exactly. That is the best product-distribution fit you can get from this architecture. The statistics the init consumes are $n$ numbers: the feature means. There are no correlations to consume, because the architecture has no parameter that touches a pair of qubits.

Under Recipe B, $\theta_j$ is a scaled linear function of the same marginal.

This family is the easy case. There is no 2-body structure to miss, so the init can match the data as well as the architecture allows. Uniform init does not hit a barren plateau here (see [[Scope Lock]]), which makes data-dependent init mostly about starting closer to the target rather than about rescuing training.

### 2D ZZ lattice family

Every row has weight 2, one per nearest-neighbor pair on an open-boundary $L \times L$ grid. That gives $m = 2L(L-1)$ rotations. See [[ZZ Lattice Family]].

Under Recipe A, $\theta_{jk} \propto \mathrm{cov}(x_j, x_k)$, but only for pairs $(j, k)$ that live on a lattice edge. If two features are strongly correlated in the data but their qubits are not neighbors on the grid, the init sees nothing, because there is no parameter for that pair. Recipe B is the same idea with a parity average instead of a covariance.

The init ends up consuming $O(n)$ pair correlations: a sparse, geometrically-shaped slice of the full covariance matrix.

A caveat that is easy to miss. This only *means* anything if the feature ordering is consistent with the lattice layout. If you are training on pixels and the qubit layout corresponds to pixel positions, the init is exploiting real image-like structure. If you are training on arbitrary tabular data where feature 7 and feature 8 are in no way adjacent in the world, the init still runs, but you are scaling the rotation between two randomly-paired things and calling it "geometric structure."

Uniform init works fine on the lattice family, so data-dependent init here is an accelerator, not a rescuer. The thing to measure is how much faster training converges, not whether training works at all.

### Sparse Erdős–Rényi family

Every row has weight 2, one per edge sampled under $\text{Bernoulli}(p = c/n)$ with target average degree $c$. The row count varies from draw to draw. See [[Erdos-Renyi Family]].

Same formulas as the lattice case, just on a different set of edges. The statistics consumed are 2-body correlations on a *random* subset of pairs, with size about $cn/2$ in expectation.

Here is the Erdős–Rényi twist that the other families do not have. The sampled edge set is RNG-dependent. Two different seeds with the same spec see *different subsets of the covariance matrix*. The quality of the init ends up depending on whether the sampled edges happen to hit the high-covariance pairs in the data. You can, in principle, get lucky or unlucky.

This opens a natural question the project is well-positioned to ask. Does the init waste information whenever the sampled edges miss the important correlations? And if so, could a data-aware edge sampler fix it? No paper in the bundle addresses this.

Under uniform init, sparse ER at $c \approx 2$ is not expected to plateau. Dense ER at $p \approx 0.5$ is (see [[Scope Lock]]).

### Complete graph family

Every row has weight 2, one per $\binom{n}{2}$ pair. No edges are missing, so this family never drops a pairwise correlation on the floor. See [[Complete Graph Family]].

Recipe A sets $\theta_{jk} \propto \mathrm{cov}(x_j, x_k)$ for every pair, so the full empirical covariance matrix flows straight into $\theta$. Recipe B is the corresponding parity-average version.

This is the family where uniform init is expected to hit a barren plateau under every kernel setting (again, see [[Scope Lock]]). It is also the family where Recio-Armengol et al. 2025 directly observed the rescue effect in their Fig. 12 ablation on all-to-all architectures. If data-dependent init is going to save trainability anywhere in this sweep, the complete graph is where we should see it save it.

### Legacy families (weight ≥ 3)

`bounded_degree`, `dense`, `community`, and `symmetric` can produce rows with Hamming weight 3 or higher (see [[Hypergraph Families]]). They are not in the primary sweep, but they are a useful illustration of where the two recipes diverge.

Recipe A does not cover weight ≥ 3 at all. It falls back to a random small-angle Gaussian for anything bigger than a pair, which amounts to "no closed-form data map." So if a legacy family mixes weight-2 and weight-3 rows, Recipe A gives a *partial* data-dependent init: the first two weight classes consume data, the rest are noise.

Recipe B still works, because its only ingredient is $\langle Z_{g_j}\rangle_p$, and that is defined for any generator regardless of weight. This is the one real architectural advantage the project's lightweight I3 has over Recio-Armengol et al. 2025's recipe.

That said, all four SMART families have weight ≤ 2, so the distinction only shows up if the project later decides to sweep a legacy family.

> [!note] The recipe split only matters outside the SMART four
> All four SMART families have weight ≤ 2, so Recipe A and Recipe B both work uniformly across the primary sweep. The split only matters if the project eventually sweeps the legacy higher-order families.

---

## Cross-family comparison

| Family | Rows | What the init uses | Size of stat set | BP under uniform? | Init expected to matter? |
|---|---|---|---|---|---|
| Product state | weight-1, $m = n$ | feature means $\{\bar x_j\}$ | $n$ | No | Mildly, acceleration |
| 2D lattice | weight-2, $m = 2L(L-1)$ | pair correlations on lattice edges | $O(n)$ | No | Mildly, acceleration |
| ER sparse ($c \approx 2$) | weight-2, $m \approx cn/2$ | pair correlations on sampled edges | $O(cn/2)$ | No | Medium, sample-dependent |
| Complete graph | weight-2, $m = \binom{n}{2}$ | full pair correlation matrix | $O(n^2)$ | Yes | Most (potential rescue) |
| (Legacy higher-order) | weight ≥ 3 | Recipe B only; Recipe A fails | varies | family-dependent | family-dependent |

Something jumps out of this table if you read down the "BP" and "matters" columns together. The families where the init is expected to matter most, complete graph and dense, are also the families where the init has the most raw data to consume. That is not a coincidence. Recio-Armengol et al. 2025's recipe is parameterized by gate connectivity, not by a fixed "side-information budget." A richer architecture gives the init more parameters to fill with data statistics, so it both needs the help more and is better equipped to receive it.

The less flattering way to say the same thing: the init can only carry the data as far as the architecture lets it. The 2D lattice family can only express pair correlations between grid neighbors, no matter how much off-lattice correlation the dataset contains. The complete graph family can express all of it. Data-dependent init cannot sneak in correlations that the architecture has no parameters for, no matter how clever the recipe is.

---

## Where the literature helps and where it stops

**Recio-Armengol et al. 2025 (2503.02934)** is the paper that gives us Recipe A and shows empirically that it beats uniform init on datasets that look like the complete-graph case. The paper also makes a model-class-level claim: data-dependent init is an IQP strength that is not generally available in classical generative models like RBMs or EBMs. Since all four SMART families are members of the IQP model class, that model-class claim carries across all of them.

**Mhiri 2025 (2502.07889)** provides the general warm-start framework. Any parameter setting with non-vanishing curvature seeds a polynomially-shrinking "patch" of guaranteed gradients around it. The framework is architecture-agnostic *within* non-commuting circuit families like HEA, HVA, and UCC, but it does not cover commuting IQP circuits. Recio-Armengol et al. 2025 flags exactly this in Sec. 9.2. So Mhiri 2025 supports the *existence* of patches around a data-dependent init point conceptually, but the family-by-family patch-radius bounds in that paper are not for the families in this project.

**Larocca 2024 (2405.00781, Sec. VII.D)** gives the survey-level taxonomy of warm-start strategies. It also raises the warning that matters most here: a smartly initialized circuit can land in a region that is classically simulable. If that happens, there is no quantum advantage left, which connects this whole question to [[IQP Classical Sampling]].

**Rudolph 2023 (2305.02881, Supplemental Note II B)** is the sanity check. The paper shows that naive data-dependent inits, like "initialize at a single training bitstring," fail on general distributions, because the model has exponentially low probability on anything except that one bitstring. Only cluster-structured data escapes the failure. This is the baseline any real IQP init has to beat.

**What the literature does not tell us.** None of the papers measure data-dependent init across different IQP connectivity families. Recio-Armengol et al. 2025 uses one architecture per experiment and does not vary connectivity. Mhiri 2025's patch-radius table is for HEA, HVA, and UCC, not for lattice or Erdős–Rényi. And no paper in the bundle addresses the specific Erdős–Rényi question of how random edge sampling interacts with data-driven init quality. That family-by-family comparison is exactly the gap this project can fill.

---

## What this means for the project

A few practical takeaways.

Recipe B, the project's lightweight I3, is the right default. It applies to all four SMART families and also to the legacy higher-order families, without any casework. The cost is that you lose Recipe A's clean "match the single-qubit marginals exactly" interpretation in exchange for that uniformity.

A natural ablation is Recipe A versus Recipe B at fixed family. If Recipe A wins on the product state family, where its interpretation is cleanest, but the two are close on lattice, ER, and complete graph, that would be evidence the weight-agnostic version is a safe substitute. If Recipe A wins on complete graph, where Recio-Armengol et al. 2025 claims the rescue effect is largest, that would be evidence the lighter version is leaving something on the table.

The complete graph family is the crown-jewel test case. It is the family where the init is expected to matter most, where Recio-Armengol et al. 2025's claim is strongest, and where Recipe A can use the full covariance matrix. If data-dependent init fails here, something is deeply wrong with the strategy. If it works here, the result is a direct replication of Recio-Armengol et al. 2025 in the MMD setting this project actually studies.

Erdős–Rényi is the cleanest stress test for Recipe B, because the edge set is random and neither recipe can "choose" good edges. Both are at the mercy of the sample, which makes the head-to-head fair.

2D lattice is the cleanest architectural-alignment test. If you use a dataset with real geometric structure, like a 2D Ising model or a binary mixture on a grid, the lattice family can actually exploit it. If the alignment breaks (arbitrary feature ordering), the init still runs, but it does not mean much.

Small scoping warning. If the project ever sweeps the legacy higher-order families, Recipe A is not a drop-in. You would either have to use Recipe B, or use Recipe A with the small-angle fallback on the weight ≥ 3 rows. Worth flagging before proposing the extension.

---

## Related

- [[Initialization Schemes]] — the three schemes I1/I2/I3 this project implements
- [[Families MOC]] — the SMART four and the legacy families
- [[Product State Family]]
- [[ZZ Lattice Family]]
- [[Erdos-Renyi Family]]
- [[Complete Graph Family]]
- [[Hypergraph Families]]
- [[Generator Matrix]] — the binary matrix the init is keyed off of
- [[Barren Plateaus]] — the failure mode the init is trying to avoid
- [[Gradient Variance]] — the scalar the [[Scaling Runner]] measures
- [[Research Questions]] — which outcome the family-by-family comparison is evidence for
- [[Implementation Choices#Why the init layer changed too]] — why the project has a lightweight I3
- [[References#Recio-Armengol et al. 2025 — Scaling GQML]] — Recipe A (the paper earlier drafts mislabeled as "Rudolph 2025")
- [[References#Mhiri 2025]] — the warm-start framework
- [[References#Larocca 2024]] — the taxonomy
- [[References#Rudolph 2023]] — the naive-data-init baselines
