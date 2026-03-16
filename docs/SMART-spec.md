## **SMART Execution Specification (Feb 22, 2026 → May 15, 2026\)**

### **Project Name**

**IQP–MMD Trainability & Barren Plateau Study (Theory \+ Hypothesis \+ Qiskit \+ Forge)**

### **Time Window**

**Start:** Sunday, Feb 22, 2026  
**End:** Friday, May 15, 2026  
**Duration:** \~12 weeks

---

# **1\) SMART Summary**

## **S — Specific**

From Feb 22 to May 15, we will:

1. Lock the precise problem definition (IQP circuit family, MMD kernel, parameter distribution, and gradient target).  
2. Implement a reproducible computational pipeline:  
   * Hypothesis-driven circuit generator (hypergraph families).  
   * Classical IQP expectation engine (for ⟨Zₐ⟩ and MMD² mixture) .  
   * Gradient/variance estimator (per-parameter and aggregate).  
3. Execute scaling experiments across ≥ 6 circuit families and ≥ 3 initialization schemes.  
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
* **Experiments:** ≥ 6 circuit families × ≥ 6 n-values × ≥ 100 random seeds per setting (or justified alternative).  
* **Scaling outputs:** ≥ 8 publication-quality figures (variance vs n; shot/noise effect).  
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

### **Loss**

Squared MMD with Gaussian kernel bandwidth σ. Use the mixture-of-Z-words form:  
\[  
\\text{MMD}^2(p,q\_\\theta)=\\mathbb{E}*{a \\sim P*\\sigma}\\left\[(\\langle Z\_a\\rangle\_p \- \\langle Z\_a\\rangle\_{q\_\\theta})^2\\right\]  
\]  
(Operationally based on Proposition-style decomposition described in .)

### **Gradient Target**

* Per-parameter gradient: ( \\partial\_{\\theta\_i}\\text{MMD}^2 )  
* Aggregate gradient norm proxy: ( \\mathbb{E}*i\[\\text{Var}(\\partial*{\\theta\_i} \\mathcal{L})\] ) or median across i.

### **Parameter and Circuit Distributions**

* θ initialization schemes:  
  * i.i.d. uniform on (\[-\\pi,\\pi\]) (stress test)  
  * small-angle normal ( \\mathcal{N}(0,\\sigma\_\\theta^2) )  
  * data-dependent initialization (covariance-based) inspired by  
* Circuit families (hypergraph generators (g\_j)):  
  * bounded-degree k-local  
  * Erdos–Renyi hyperedges  
  * lattice-local  
  * dense/complete-ish  
  * community-structured  
  * symmetry-constrained (global bitflip symmetry class; optional)

### **Datasets p(x)**

Use 3 targets (start simple, then structured):

1. Product Bernoulli (easy baseline)  
2. Ising-like synthetic (pairwise correlations)  
3. Real binary dataset (small) or structured synthetic mixture.

---

# **3\) Weekly / Biweekly Execution Plan (Deliverables)**

## **Week 1 (Feb 22–Feb 28): Scope Lock \+ Repo Skeleton**

**Deliverables**

* D1.1: 2-page “Scope Lock” memo:  
  * exact model definition, loss form, gradient definitions, experiment grid  
* D1.2: Repo structure \+ reproducibility scaffold:  
  * `src/iqp/`, `src/mmd/`, `src/experiments/`, `configs/`, `notebooks/`, `forge/`  
  * deterministic seeding, logging, results serialization (JSONL/Parquet)  
* D1.3: Minimal working pipeline (MWP):  
  * generate hypergraph → compute ⟨Z\_a⟩qθ (single a) → compute one gradient estimate

**Method**

* Implement classical ⟨Z\_a⟩ estimator via uniform-z Monte Carlo per .  
* Confirm matches brute-force statevector for n ≤ 10 (sanity check).

---

## **Week 2 (Feb 29–Mar 6): Correctness Validation Suite**

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
  * `mmd2(theta, hypergraph, dataset, sigma, num_a_samples, num_z_samples)`  
  * `grad_mmd2(theta, ...)` via autodiff (JAX) or analytic partials.

---

## **Weeks 3–4 (Mar 7–Mar 20): Scaling Experiments v1 (Classical Exact Regime)**

**Deliverables (biweekly)**

* D4.1: Experiment grid v1 completed:  
  * Circuit families: 4  
  * Inits: 2  
  * n values: at least 6 (e.g., 16, 24, 32, 48, 64, 96\)  
  * seeds per setting: ≥ 50  
* D4.2: Scaling figures v1:  
  * Var(∂θ\_i L) vs n for each family  
  * aggregate plots \+ fits (exponential vs polynomial)  
* D4.3: Interim interpretation memo:  
  * identify candidate “trainable regimes” and “plateau regimes”  
  * choose 2 regimes for Qiskit validation

**Method**

* Use Hypothesis to generate hypergraphs with controlled statistics.  
* Enforce comparable parameter counts across n (e.g., m \= c·n or m \= c·n log n).  
* Use fixed computational budgets:  
  * num\_a\_samples (mixture): e.g., 256–2048  
  * num\_z\_samples: tuned to stabilize variance estimates.

---

## **Weeks 5–6 (Mar 21–Apr 3): Qiskit Pipeline \+ Cross-Checks**

**Deliverables (biweekly)**

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

## **Weeks 7–8 (Apr 4–Apr 17): Scaling Experiments v2 (Expanded \+ Noise Models)**

**Deliverables (biweekly)**

* D8.1: Experiment grid v2 completed:  
  * Circuit families: ≥ 6  
  * Inits: ≥ 3 (include data-dependent init)  
  * n values: extend to ≥ 128 (classical) where feasible  
* D8.2: Noise model experiments (Qiskit Aer):  
  * at least 2 noise models (depolarizing \+ readout; optionally amplitude damping)  
  * compare variance scaling with noise on/off  
* D8.3: Consolidated figure set v2 (target ≥ 8 figures)

**Method**

* Add a “noise study” axis:  
  * measure how Var(∂L) changes with noise rate ε  
* Tie findings to known issues: noise can flatten landscapes in VQAs; quantify if it dominates here.

---

## **Week 9 (Apr 18–Apr 24): Forge Modeling Sprint**

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

## **Weeks 10–11 (Apr 25–May 8): Writing \+ Synthesis (Draft Complete)**

**Deliverables (biweekly)**

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

## **Week 12 (May 9–May 15): Finalization \+ Submission Package**

**Deliverables**

* D12.1: Final report/thesis (PDF) with:  
  * clean figures, captions, references  
  * limitations \+ future work  
* D12.2: Final code release tag:  
  * `v1.0` with reproducible scripts  
* D12.3: Final slide deck \+ 1-page executive summary

**Method**

* Freeze experiments by May 9\.  
* Use May 9–15 exclusively for editing, formatting, and packaging.

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

* Use mixture sampling over a \~ Pσ(a) per  
* For each a:  
  * estimate ⟨Z\_a⟩p from dataset samples  
  * estimate ⟨Z\_a⟩qθ from IQP expectation engine  
  * accumulate squared difference

## **4.3 Gradient Computation**

Primary: autodiff through Monte Carlo estimator (JAX) for classical pipeline.  
Secondary: parameter-shift / finite differences for Qiskit validation.

Acceptance:

* gradient sanity (finite diff) on small n  
* variance estimates converge with increased budgets

## **4.4 Statistical Protocol**

For each (family, n, init, σ):

* sample S circuits (or fixed hypergraph with varying θ, as defined)  
* sample T θ seeds  
* compute gradient variance across θ (and optionally across circuits)

Minimum defaults:

* seeds T ≥ 50 (prefer 100\)  
* use median-of-means if heavy tails observed

---

# **5\) Reporting Structure (What the final document will contain)**

1. Background (IQP hardness \+ MMD \+ barren plateaus)  
2. Problem definition (exact scaling question)  
3. Derivation of gradient estimator  
4. Hypothesis-driven experiment design  
5. Classical scaling results  
6. Qiskit validation (shots \+ noise)  
7. Forge structural findings  
8. Synthesis: when/why plateaus occur  
9. Limitations and future work

---

# **6\) Success Criteria (Pass/Fail)**

By May 15, 2026, the project is “successful” if:

* You can state a clear conclusion of the form:  
  **“Under these circuit families and initializations, Var(∂L) scales \[exponentially/polynomially\] with n; shot/noise effects \[do/do not\] reinstate exponential suppression.”**

and you can support it with:

* ≥ 6 scaling plots  
* ≥ 1 Qiskit shot/noise study  
* ≥ 1 structural insight from Forge or a minimal counterexample

---

# **7\) Immediate Next Actions (Today → 72 hours)**

1. Create `ScopeLock.md` with:  
   * circuit families list  
   * dataset plan  
   * exact metrics (which variance, over what randomness)  
2. Create repo skeleton \+ configs  
3. Implement smallest end-to-end run:  
   * generate hypergraph  
   * compute ⟨Z\_a⟩qθ  
   * compute MMD² estimate  
   * compute one gradient estimate  
4. Add correctness test for n≤10 via brute force

If you want, I can also produce:

* A concrete `configs/` schema (YAML) for all experiment axes,  
* A minimal folder blueprint for the repo,  
* A template for the thesis/report structure with section-by-section bullet points.

