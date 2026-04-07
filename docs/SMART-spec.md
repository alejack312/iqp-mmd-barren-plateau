## **SMART Execution Specification (Mar 18, 2026 → May 15, 2026\)**

### **Project Name**

**IQP–MMD Trainability & Barren Plateau Study (Theory \+ Hypothesis \+ Qiskit \+ Forge)**

### **Time Window**

**Start:** Wednesday, Mar 18, 2026
**End:** Friday, May 15, 2026
**Duration:** \~8 weeks

---

# **1\) SMART Summary**

## **S — Specific**

From Feb 22 to May 15, we will:

1. Lock the precise problem definition (IQP circuit family, MMD kernel, parameter distribution, and gradient target).  
2. Implement a reproducible computational pipeline:  
   * Hypothesis-driven circuit generator (hypergraph families).  
   * Classical IQP expectation engine (for ⟨Zₐ⟩ and MMD² mixture) .  
   * Gradient/variance estimator (per-parameter and aggregate).  
   * Full classical training loop for the MMD objective, with quantum execution reserved for sampling from the learned IQP model at inference time.  
3. Execute scaling experiments across 4 circuit families × 3 kernel types × ≥ 3 initialization schemes (including small-angle), with primary focus on Gaussian kernel × all 4 connectivity families before extending to other kernels.
4. Validate selected regimes in Qiskit:  
   * Statevector baseline  
   * Shot-based estimation  
   * Noise-model runs  
5. Use Forge to search for structural “plateau-inducing” patterns and encode formal structural statements (finite-n).  
6. Write a thesis-quality report with interpretable conclusions.

## **M — Measurable**

By May 15, you will have:

* **Derivations:** 1 complete derivation note for ∂θ MMD² in the IQP–MMD decomposition framework.  
* **Pipeline:** 1 repo with end-to-end reproducible experiments \+ config-driven runs.  
* **Experiments:** 4 circuit families × 3 kernel types × ≥ 3 initialization schemes × ≥ 6 n-values × ≥ 100 random seeds per setting (or justified alternative). Phase 1: Gaussian kernel × 4 connectivity families (complete); Phase 2: Laplacian + multi-scale Gaussian.
* **Scaling outputs:** ≥ 12 publication-quality figures (variance vs n per kernel; kernel comparison; shot/noise effect; init scheme comparison).
* **Qiskit validation:** ≥ 3 comparisons (exact vs statevector vs shots; with/without noise).  
* **Forge results:** ≥ 2 structural findings (counterexample or invariant) \+ serialized Forge models.  
* **Writing:** Final report/thesis draft (target 40–70 pages) \+ 10–15 slide deck.

## **A — Achievable**

Achievable because core expectation values for parameterized IQP circuits used in MMD² training are classically estimable efficiently using the den Nest-style formulation utilized in the “train on classical, deploy on quantum” approach . Qiskit validation is limited to small/moderate n where simulators are feasible.

## **R — Relevant**

Directly answers whether IQP+MMD generative learning avoids barren plateaus, and how/when shot noise or hardware noise reinstates trainability barriers, while relating to IQP hardness results and generative-model trainability theory .

## **T — Time-Bound**

Weekly/biweekly deliverables below. Final submission package ready **by May 15, 2026**.

---

# **2\) Methodology (Operational Definition)**

## **2.1 Objects of Study**

### **IQP Model Class (parameterized IQP circuits)**

Use the definition and structure consistent with :

* Gates: ( \\exp(i \\theta\_j X^{g\_j}) ) with generator bitmask ( g\_j \\in {0,1}^n )  
* Measurement in computational basis.

Operational umbrella: use IQP circuits as generative models trained against data with the MMD loss entirely on classical hardware, then sample from the trained circuit on quantum hardware so that inference remains aligned with the IQP sampling-hardness motivation.

### **Loss**

Squared MMD with kernel k, written in mixture-of-Z-words form:
\[
\\text{MMD}^2(p,q\_\\theta)=\\mathbb{E}\_{a \\sim P\_k}\\left\[(\\langle Z\_a\\rangle\_p \- \\langle Z\_a\\rangle\_{q\_\\theta})^2\\right\]
\]
(Operationally based on Proposition-style decomposition described in .)

We study three kernel families — the choice of kernel determines P_k and the spectral weighting of the Z-word mixture:

| Kernel | k(x,y) | Notes | Priority |
|---|---|---|---|
| Gaussian | exp(-‖x-y‖²/2σ²) | Primary; σ is bandwidth axis | Phase 1 |
| Laplacian | exp(-‖x-y‖/σ) | Heavier tails than Gaussian | Phase 2 |
| Multi-scale Gaussian | Σ_i w_i exp(-‖x-y‖²/2σ_i²) | Mixture of Gaussians; captures multi-scale correlations | Phase 2 |

**Study order:** The Gaussian kernel is analysed exhaustively across all four circuit families first. Laplacian and multi-scale Gaussian are added once the Gaussian regime is fully characterised. For each kernel, we derive the explicit MMD² expression as a function of the IQP circuit family and write the gradient ∂_{θ_i} MMD² in closed form.

### **Gradient Target**

* Per-parameter gradient: ( \\partial\_{\\theta\_i}\\text{MMD}^2 )  
* Aggregate gradient norm proxy: ( \\mathbb{E}*i\[\\text{Var}(\\partial*{\\theta\_i} \\mathcal{L})\] ) or median across i.

### **Parameter and Circuit Distributions**

* θ initialization schemes (all three are primary axes, not optional):
  * **Uniform** U[-π,π] — stress test / worst-case
  * **Small-angle** N(0,σ_θ²) with σ_θ ∈ {0.01, 0.1, 0.3} — main trainability hypothesis
  * **Data-dependent** (covariance-based) — inspired by structured init literature

  For each (kernel × connectivity) combination, we compare gradient variance across all three inits to determine whether small-angle initialization suppresses or avoids barren plateaus independently of circuit structure.
* Circuit families (hypergraph generators (g\_j)):
  * **product state** — single-qubit Z rotations only (degree-1 hyperedges; baseline, no entanglement)
  * **2D lattice** — nearest-neighbour ZZ interactions on a square lattice
  * **sparse Erdős–Rényi** — random hyperedges with bounded expected degree
  * **complete graph** — all-to-all ZZ interactions (dense limit)

### **Datasets p(x)**

Use 3 targets (start simple, then structured):

1. Product Bernoulli (easy baseline)  
2. Ising-like synthetic (pairwise correlations)  
3. Real binary dataset (small) or structured synthetic mixture.

### **Output-distribution anti-concentration target**

In addition to gradient-variance scaling, we will evaluate whether the trained IQP output
distribution itself is anti-concentrated. For a distribution `p_theta` on `{0,1}^n`, the
reference criterion is:
\[
\Pr_{x \sim \mathrm{Unif}(\{0,1\}^n)}\left[p_\theta(x) \ge \alpha 2^{-n}\right] \ge \beta
\]
for constants `alpha, beta > 0` independent of `n`.

Operationally, the code will report the finite-`n` threshold curve
\[
\hat{\beta}_{p_\theta}(\alpha) = 2^{-n}\left|\{x \in \{0,1\}^n : p_\theta(x) \ge \alpha 2^{-n}\}\right|
\]
for a small fixed grid such as `alpha in {0.25, 0.5, 1.0, 2.0}`. A run passes the empirical
anti-concentration check at threshold pair `(alpha, beta_min)` only when
`beta_hat_p_theta(alpha) >= beta_min`.

Deterministic implementation plan:

* **Exact regime (primary):** for trained IQP circuits with `n <= 16-20`, reconstruct the full
  probability vector `p_theta(x)` and evaluate `beta_hat_p_theta(alpha)` exactly.
* **Sample regime (secondary):** for larger `n`, estimate the occupied mass pattern from generated
  samples and report diagnostics only; this does not replace the exact small-`n` claim.
* **Supporting diagnostics:** report collision probability `sum_x p_theta(x)^2`, scaled
  `2^n sum_x p_theta(x)^2`, `max_x 2^n p_theta(x)`, and effective support size
  `1 / sum_x p_theta(x)^2`.

Planned deterministic API:

* `check_anti_concentration(probabilities, alpha=0.5, beta_min=0.1, atol=1e-12) -> dict`

The return payload should include `n`, `alpha`, `beta_hat`, `passes_threshold`,
`collision_probability`, `scaled_l2`, `max_probability_scaled`, and `effective_support_ratio`.

This evaluation belongs on the deterministic experiment / validation side. Training may still
happen through `iqp_mmd`, but the anti-concentration claim and serialized evidence should be
produced by `iqp_bp` after loading trained parameters or generated bitstrings.

---

# **3\) Weekly / Biweekly Execution Plan (Deliverables)**

## **Week 1 (Mar 18–Mar 24): Scope Lock \+ Repo Skeleton**

**Deliverables**

* D1.1: 2-page “Scope Lock” memo:
  * exact model definition, gradient definitions, experiment grid
  * IQP connectivity families: product state, 2D lattice, sparse Erdős–Rényi, complete graph
  * explicit MMD² loss written out for Gaussian kernel (primary) and each connectivity family; stubs for Laplacian and multi-scale Gaussian
  * statement of the main barren plateau question per (connectivity, bandwidth σ, init) triple under Gaussian kernel
* D1.2: Repo structure \+ reproducibility scaffold:  
  * `src/iqp/`, `src/mmd/`, `src/experiments/`, `configs/`, `notebooks/`, `forge/`  
  * deterministic seeding, logging, results serialization (JSONL/Parquet)  
* D1.3: Minimal working pipeline (MWP):  
  * generate hypergraph → compute ⟨Z\_a⟩qθ (single a) → compute one gradient estimate

**Method**

* Implement classical ⟨Z\_a⟩ estimator via uniform-z Monte Carlo per .  
* Confirm matches brute-force statevector for n ≤ 10 (sanity check).

---

## **Week 2 (Mar 25–Mar 31): Correctness Validation Suite**

**Deliverables**

* D2.1: Unit tests:  
  * expectation estimator correctness vs exact enumeration (n ≤ 12\)  
  * gradient correctness vs finite differences (small n)  
* D2.2: Baseline experiment script:  
  * run for n ∈ {8, 10, 12, 14, 16} on 2 circuit families  
* D2.3: “Validation Report” with 3 plots:  
  * estimator error vs samples  
  * gradient error vs samples  
  * runtime vs n

**Method**

* Implement clean interfaces:  
  * `iqp_expectation(theta, hypergraph, a, num_z_samples)`  
  * `mmd2(theta, hypergraph, dataset, kernel, kernel_params, num_a_samples, num_z_samples)` — kernel is pluggable (Gaussian, Laplacian, polynomial, linear)
  * `grad_mmd2(theta, ...)` via autodiff (JAX) or analytic partials.

---

## **Weeks 3–4 (Apr 1–Apr 14): Scaling Experiments v1 (Classical Exact Regime)**

**Deliverables (biweekly)**

* D4.1a: Anti-concentration workstream completed by Apr 8, 10:30am:
  * write a technical note `docs/technical/anti-concentration.md` explaining the paper-level
    definition, the finite-n surrogate used in this repo, and the limits of sample-only checks
  * add an exact small-n probability-vector extraction path for trained IQP circuits
  * add a deterministic anti-concentration checker and JSON/CSV result schema on the validation side
* D4.2a: Add anti-concentration figures for selected trained checkpoints:
  * threshold curves for `2^n p_theta(x)`
  * collision-probability and max-probability trends
* D4.3a: Add an interim interpretation question:
  * do the trained output distributions remain anti-concentrated in the regimes that appear trainable?

* D4.1: Experiment grid v1 completed (Phase 1 — Gaussian kernel):
  * Circuit families: 4 (product state, 2D lattice, sparse Erdős–Rényi, complete graph)
  * Kernel: Gaussian with σ ∈ {0.1, 0.5, 1.0, 2.0, 5.0} (bandwidth sweep)
  * Inits: 3 (uniform, small-angle N(0,σ_θ²) with σ_θ ∈ {0.01,0.1,0.3}, data-dependent)
  * n values: at least 6 (e.g., 16, 24, 32, 48, 64, 96)
  * seeds per setting: ≥ 50
* D4.2: Scaling figures v1:
  * Var(∂θ_i L) vs n for each (family, σ) pair under Gaussian kernel
  * Connectivity comparison plots: for fixed σ, how does circuit family change scaling?
  * Bandwidth comparison plots: for fixed connectivity, how does σ change scaling?
  * Init comparison plots: for fixed (connectivity, σ), how does small-angle init change scaling?
  * Aggregate fits (exponential vs polynomial)
* D4.3: Interim interpretation memo:
  * identify candidate “trainable regimes” and “plateau regimes” per connectivity × bandwidth
  * answer: “does this loss function exhibit a barren plateau?” for each (connectivity, σ) pair
  * choose 2 regimes for Qiskit validation; decide whether to proceed to Laplacian/multi-scale Gaussian

**Method**

* Use Hypothesis to generate hypergraphs with controlled statistics.  
* Enforce comparable parameter counts across n (e.g., m \= c·n or m \= c·n log n).  
* Use fixed computational budgets:  
  * num\_a\_samples (mixture): e.g., 256–2048  
  * num\_z\_samples: tuned to stabilize variance estimates.
* Route trained-parameter snapshots into the deterministic validation runner:
  * exact probabilities first for small `n`
  * sample-histogram diagnostics second for larger `n`

---

## **Week 5 (Apr 15–Apr 21): Qiskit Pipeline \+ Cross-Checks**

**Deliverables**

* D6.1: Qiskit circuit generator:  
  * hypergraph → parameterized IQP circuit builder  
  * supports exporting QASM \+ transpilation settings  
* D6.2: Qiskit validation report (noise-free \+ shots):  
  * exact classical vs statevector vs shot-based  
  * n ≤ 20 (statevector feasible), shots ∈ {1k, 10k, 100k}  
* D6.3: “Shot-induced plateau” measurement:  
  * plot gradient SNR vs n for fixed shot budgets  
  * estimate shots needed to keep relative error constant

**Method**

* Implement (e^{i\\theta X^{g}}):  
  * apply H on support qubits to map X→Z  
  * implement multi-qubit Z-rotation on parity (use CNOT ladder \+ RZ \+ uncompute)  
  * apply H back  
* For gradients in Qiskit:  
  * use parameter-shift for small n or finite-difference on loss estimator  
  * keep measurement strategy aligned with MMD² estimator.

---

## **Week 6 (Apr 22–Apr 28): Scaling Experiments v2 (Expanded \+ Noise Models)**

**Deliverables**

* D8.1: Experiment grid v2 completed (Phase 2 — Laplacian + multi-scale Gaussian):
  * Circuit families: all 4 (same grid as v1)
  * Kernel types: add Laplacian and multi-scale Gaussian (compare against Gaussian baseline from v1)
  * Inits: all 3 (including small-angle sweep over σ_θ values)
  * n values: extend to ≥ 128 (classical) where feasible
* D8.2: Noise model experiments (Qiskit Aer):
  * at least 2 noise models (depolarizing + readout; optionally amplitude damping)
  * compare variance scaling with noise on/off, per kernel type
* D8.3: Consolidated figure set v2 (target ≥ 12 figures, including kernel-comparison and init-comparison panels)

**Method**

* Add a “noise study” axis:  
  * measure how Var(∂L) changes with noise rate ε  
* Tie findings to known issues: noise can flatten landscapes in VQAs; quantify if it dominates here.

---

## **Week 7 (Apr 29–May 5): Forge Modeling Sprint**

**Deliverables**

* D9.1: Forge model encoding:  
  * qubits, hyperedges, overlap graph, degree constraints  
* D9.2: 2 structural results:  
  * either invariant statements (always true in bounded-degree families up to n≤N)  
  * or minimal counterexample circuits showing plateau-like overlap patterns  
* D9.3: “Structural Lemmas” note (3–6 pages) connecting Forge findings to empirical regimes

**Method**

* Forge used for:  
  * finite model finding: “find smallest circuit where overlap pattern exceeds threshold”  
  * search: minimal instances where all generators share support, etc.  
* Output: serialized instances that can be fed back into Python/Qiskit.

---

## **Week 8 (May 6–May 11): Writing \+ Synthesis (Draft Complete)**

**Deliverables**

* D11.1: Full draft v1 (complete narrative):  
  * Intro \+ background  
  * Methods (classical \+ Hypothesis \+ Qiskit \+ Forge)  
  * Results (v1+v2)  
  * Discussion: reconcile hardness vs trainability  
* D11.2: Reproducibility appendix:  
  * configs, run commands, compute budgets, seeds  
* D11.3: Slide deck v1 (12–15 slides)

**Method**

* Write while final experiments run only if strictly needed.  
* Ensure each claim is backed by a figure/table or a derivation.

---

## **Final Days (May 12–May 15): Finalization \+ Submission Package**

**Deliverables**

* D12.1: Final report/thesis (PDF) with:  
  * clean figures, captions, references  
  * limitations \+ future work  
* D12.2: Final code release tag:  
  * `v1.0` with reproducible scripts  
* D12.3: Final slide deck \+ 1-page executive summary

**Method**

* Freeze experiments by May 11.
* Use May 12–15 exclusively for editing, formatting, and packaging.

---

# **4\) Detailed Execution Standards (How we will run each component)**

## **4.1 Classical IQP Expectation Engine**

* Inputs: (θ, hypergraph G, observable mask a, budget Bz)  
* Output: ⟨Z\_a⟩ estimate with CI  
* Implementation:  
  * uniform z sampling  
  * vectorize phase computation  
  * stable batching to avoid memory blowups  
* Acceptance:  
  * matches exact enumeration within tolerance for n≤12  
  * stable variance estimates across ≥3 independent runs

## **4.2 MMD² Estimation**

* Kernel is a pluggable parameter; P_k(a) is derived from the chosen kernel's spectral decomposition
* Supported kernels: Gaussian (primary, bandwidth σ), Laplacian, multi-scale Gaussian (mixture weights w_i, bandwidths σ_i)
* For each a sampled from P_k(a):
  * estimate ⟨Z_a⟩_p from dataset samples
  * estimate ⟨Z_a⟩_{q_θ} from IQP expectation engine
  * accumulate squared difference weighted by kernel spectral coefficients
* Acceptance: MMD² computed with each kernel matches brute-force for n≤10

## **4.3 Gradient Computation**

Primary: autodiff through Monte Carlo estimator (JAX) for classical pipeline.  
Secondary: parameter-shift / finite differences for Qiskit validation.

Acceptance:

* gradient sanity (finite diff) on small n  
* variance estimates converge with increased budgets

## **4.4 Statistical Protocol**

For each (family, kernel, n, init, kernel_params):

* sample S circuits (or fixed hypergraph with varying θ, as defined)  
* sample T θ seeds  
* compute gradient variance across θ (and optionally across circuits)

Minimum defaults:

* seeds T ≥ 50 (prefer 100\)  
* use median-of-means if heavy tails observed

---

# **5\) Reporting Structure (What the final document will contain)**

1. Background (IQP hardness \+ MMD \+ barren plateaus)  
2. Problem definition (exact scaling question per kernel and connectivity)
3. MMD² loss derivation for Gaussian kernel (primary) across all four connectivity families; Laplacian and multi-scale Gaussian where complete
4. Derivation of gradient estimator (kernel-parametric form, with bandwidth σ as explicit axis)
5. Hypothesis-driven experiment design
6. Classical scaling results (connectivity × kernel × init grid)
7. Anti-concentration diagnostics for trained output distributions
8. Qiskit validation (shots + noise)
9. Forge structural findings
10. Synthesis: when/why plateaus occur — role of kernel, connectivity, initialization, and output-distribution shape
11. Limitations and future work

---

# **6\) Success Criteria (Pass/Fail)**

By May 15, 2026, the project is “successful” if:

* You can state a clear conclusion of the form:
  **”Under connectivity family X with kernel K (and bandwidth σ) and initialization I, Var(∂L) scales [exponentially/polynomially] with n; shot/noise effects [do/do not] reinstate exponential suppression.”**

  The conclusion must be stated for each of the four connectivity families under the Gaussian kernel (primary), and where time permits, for Laplacian and multi-scale Gaussian.

and you can support it with:

* ≥ 12 scaling plots (covering bandwidth × connectivity comparison under Gaussian; kernel comparison panels where available; init comparison)
* ≥ 1 exact small-`n` anti-concentration study on trained IQP output distributions, plus a documented
  sample-based follow-up for larger `n`
* ≥ 1 Qiskit shot/noise study
* ≥ 1 structural insight from Forge or a minimal counterexample
* A clear answer to: “does the choice of bandwidth (and kernel family) determine whether this MMD loss exhibits a barren plateau?”
* A clear answer to: “in the regimes that train best, are the learned output distributions still
  anti-concentrated, or do they become sparse / structured?”

---

# **7\) Immediate Next Actions (Apr 7 → Apr 8, 10:30am)**

1. Write `docs/technical/anti-concentration.md`:
   * explain the paper-level anti-concentration criterion
   * define the finite-`n` threshold statistic `beta_hat_p_theta(alpha)`
   * explain why exact probability vectors are the primary validation object and sample histograms are secondary
2. Add exact small-`n` output-probability extraction for deterministic validation:
   * input: trained IQP circuit / parameter snapshot
   * output: normalized probability vector over `{0,1}^n`
   * target regime: `n <= 16-20`
3. Add a deterministic checker function on the `iqp_bp` side:
   * `check_anti_concentration(probabilities, alpha=0.5, beta_min=0.1, atol=1e-12)`
   * return threshold result plus collision-probability and max-probability diagnostics
4. Extend the validation runner so it can:
   * load trained IQP parameters or generated bitstrings
   * run the anti-concentration check
   * serialize JSON/CSV outputs for downstream plots and memo writing
5. Add curated deterministic tests:
   * uniform distribution should pass
   * delta / highly concentrated distributions should fail
   * sample-based estimator should approach the exact checker on toy cases
6. Produce the first anti-concentration note for the week:
   * one exact small-`n` trained example
   * one sample-based larger-`n` diagnostic example
   * a short memo stating whether the learned distribution looks anti-concentrated or sparse

Hard deadline for items 1-6 above: **Wednesday, Apr 8, 2026 at 10:30am**.

If you want, I can also produce:

* A concrete `configs/` schema (YAML) for all experiment axes,
* A minimal folder blueprint for the repo,
* A template for the thesis/report structure with section-by-section bullet points.

---

# **8\) Key References**

1. Rudolph, M. S., Lerch, S., Thanasilp, S., Kiss, O., Shaya, O., Vallecorsa, S., Grossi, M., & Holmes, Z. (2023). *Trainability barriers and opportunities in quantum generative modeling*. arXiv:2305.02881. — Establishes that implicit MMD loss avoids the barren-plateau flavour introduced by explicit losses (e.g., KL divergence).

2. Larocca, M., Thanasilp, S., Wang, S., Sharma, K., Biamonte, J., Coles, P. J., Cincio, L., McClean, J. R., Holmes, Z., & Cerezo, M. (2024). *Barren Plateaus in Variational Quantum Computing*. Nature Reviews Physics 7, 174–189. arXiv:2405.00781. — Review framing BPs as average-case, curse-of-dimensionality statements that do not preclude trainable valleys.

3. Mhiri, H., Puig, R., Lerch, S., Rudolph, M. S., Chotibut, T., Thanasilp, S., & Holmes, Z. (2025). *A unifying account of warm start guarantees for patches of quantum landscapes*. arXiv:2502.07889. — Shows absence of BPs for perturbations around favourable starting points; motivates small-angle initialization study.
