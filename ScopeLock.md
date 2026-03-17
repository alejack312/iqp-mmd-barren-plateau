# ScopeLock — IQP–MMD Barren Plateau Study
**Locked:** Mar 18, 2026 | **Scope Owner:** Week 1 deliverable (D1.1)

---

## 1. Model Definition

### 1.1 IQP Circuit Family

Parameterized IQP circuit with n qubits and m generators:

```
|ψ(θ)⟩ = H^⊗n · exp(i Σ_j θ_j X^{g_j}) · H^⊗n |0⟩^n
```

- **Gates:** exp(i θ_j X^{g_j}), generator bitmask g_j ∈ {0,1}^n
- **Measurement:** computational basis → binary string x ∈ {0,1}^n
- **Output distribution:** q_θ(x) = |⟨x|ψ(θ)⟩|²

### 1.2 Classical Expectation Estimator

For observable Z_a = ⊗_i Z_i^{a_i}:

```
⟨Z_a⟩_{q_θ} = E_{z ~ U({0,1}^n)} [cos(Φ(θ, z, a))]
```

where the phase is:

```
Φ(θ, z, a) = 2 · Σ_j θ_j · (a · g_j mod 2) · (-1)^{z · g_j}
```

This is efficiently estimable classically via Monte Carlo over z (den Nest-style).

---

## 2. Loss Function — MMD² for Each Kernel

The MMD² loss in mixture-of-Z-words form:

```
MMD²(p, q_θ) = Σ_{a ∈ {0,1}^n} w_k(a) · (⟨Z_a⟩_p - ⟨Z_a⟩_{q_θ})²
             = E_{a ~ P_k} [(⟨Z_a⟩_p - ⟨Z_a⟩_{q_θ})²]
```

where P_k(a) ∝ w_k(a) are the spectral weights of kernel k.

### 2.1 Kernel Families and Spectral Weights

All kernels use ±1 encoding: x_i ∈ {±1}, Hamming distance H(x,y) = |{i : x_i ≠ y_i}|.

#### Gaussian — k(x,y) = exp(−H(x,y) / σ²)

Spectral weights (factored over qubits):

```
w_G(a; σ) ∝ tanh(1/σ²)^|a|
```

- Weight decays exponentially with Hamming weight |a|
- σ controls effective cutoff order; small σ → low-order terms dominate
- P_k biased toward low-weight Z-words
- **Primary kernel.** σ ∈ {0.5σ_med, σ_med, 2σ_med} where σ_med = median pairwise distance

#### Laplacian — k(x,y) = exp(−√H(x,y) / σ)

```
w_L(a; σ) ∝ (tanh(1/(2σ)))^|a|  [approx; exact via Walsh–Hadamard transform]
```

- Heavier tails than Gaussian in spectral domain
- More weight on intermediate-weight Z-words

#### Polynomial — k(x,y) = (x · y / n + c)^d

For degree d, inner product x·y/n ∈ [−1,1]:

```
w_P(a; d, c) = [z^{|a|}] (z + c)^d evaluated at z = 1/n · (weight-|a| contribution)
```

- Exactly degree-d kernel: only Z-words of weight ≤ d have nonzero weight
- d=1 → linear; d=2 → quadratic; d=3 → cubic
- Captures interaction structure up to order d

#### Linear — k(x,y) = x · y / n

```
w_lin(a) = 1/n  if |a| = 1,  else 0
```

- Only weight-1 Z-words (single-qubit observables)
- Degenerate baseline; gradient reduces to single-qubit expectation differences

### 2.2 Explicit MMD² Per (Kernel, Connectivity)

The organizing question for each (kernel K, connectivity family F) pair:

> **"Does Var_{θ~D}[∂_{θ_i} MMD²_K(p, q^F_θ)] decay exponentially in n?"**

Shorthand: BP(K, F) = YES means exponential decay confirmed; NO means polynomial or constant.

---

## 3. Gradient Target

### 3.1 Per-Parameter Gradient

```
∂_{θ_i} ⟨Z_a⟩_{q_θ} = −2 · (a · g_i mod 2) · E_{z~U}[sin(Φ(θ,z,a)) · (-1)^{z·g_i}]
```

```
∂_{θ_i} MMD²(p, q_θ) = −2 · E_{a~P_k}[(⟨Z_a⟩_p − ⟨Z_a⟩_{q_θ}) · ∂_{θ_i}⟨Z_a⟩_{q_θ}]
```

### 3.2 Metric — Gradient Variance

**Primary metric:**

```
V(i; θ_dist, F, K, n) = Var_{θ ~ θ_dist} [∂_{θ_i} MMD²(p, q^F_θ)]
```

**Aggregate:**

```
V_agg(F, K, n) = (1/m) Σ_i V(i; ...)   [mean over parameters]
                = median_i V(i; ...)     [if heavy tails detected]
```

**What randomness is averaged over:**
- θ ~ θ_dist (parameter initialization scheme)
- z ~ U({0,1}^n) (uniform bitstring for IQP expectation Monte Carlo)
- a ~ P_k (Z-word sampling for MMD² mixture)
- Circuit randomness: fixed hypergraph instance per seed, or average over random instances

**Scaling hypothesis to test:**

```
log V_agg(F, K, n) ~ −α(F,K) · n + β(F,K) · log n + c(F,K)
```

- α > 0, β ≈ 0 → exponential decay (barren plateau)
- α = 0, β < 0 → polynomial decay (mild plateau)
- α = 0, β = 0 → constant (no plateau)

---

## 4. Circuit Families

Six hypergraph families with controlled connectivity statistics. All generate g_j ∈ {0,1}^n.

### F1: k-Local Bounded-Degree
- **Structure:** Each generator g_j has |g_j| ≤ k (k-local, weight bounded)
- **Parameters:** k ∈ {2, 3, 4}; number of generators m = c·n
- **Hypothesis:** Locality → commuting structure → avoids BP
- **Generating rule:** Random k-subsets of qubits; degree per qubit bounded by Δ_max

### F2: Erdős–Rényi Hyperedges
- **Structure:** Each g_j includes each qubit independently with probability p_ER
- **Parameters:** p_ER ∈ {2/n, log(n)/n, 0.5}
- **Hypothesis:** Sparse p_ER ~ 2/n behaves like bounded-degree; dense p_ER ~ 0.5 → BP
- **Generating rule:** Bernoulli(p_ER) per qubit per generator

### F3: Lattice-Local (1D / 2D)
- **Structure:** Generators on nearest-neighbor pairs/triples in 1D chain or 2D grid
- **Parameters:** interaction range r ∈ {1, 2}
- **Hypothesis:** Physical locality of interactions suppresses BP
- **Generating rule:** Consecutive qubits in 1D; nearest neighbors in √n × √n grid

### F4: Dense / Complete-ish
- **Structure:** Each generator acts on Θ(n) qubits
- **Parameters:** Expected weight E[|g_j|] = n/2
- **Hypothesis:** High overlap between generators → BP
- **Generating rule:** Bernoulli(0.5) per qubit per generator

### F5: Community-Structured
- **Structure:** n qubits partitioned into B blocks; generators mostly intra-block
- **Parameters:** B ∈ {2, 4, 8}; intra-block probability 0.8, inter-block 0.05
- **Hypothesis:** Block structure creates partially-decoupled gradients → partial BP mitigation
- **Generating rule:** Mixed Bernoulli with block membership

### F6: Symmetry-Constrained (Global Bitflip)
- **Structure:** Generators symmetric under X^⊗n (global bitflip symmetry)
- **Parameters:** Enforce parity: each generator has even weight
- **Hypothesis:** Symmetry constraint changes spectral structure → qualitatively different scaling
- **Generating rule:** Sample g_j, flip one qubit to enforce |g_j| even if needed

---

## 5. Dataset Plan

Three target distributions p(x) for ⟨Z_a⟩_p:

### D1: Product Bernoulli (Baseline)
- p(x) = Π_i Bernoulli(0.5) — independent qubits
- ⟨Z_a⟩_p = 0 for all |a| ≥ 1 (all higher-order correlations vanish)
- Purpose: simplest possible target; isolates circuit structure from data structure
- Regime: tests whether gradient variance is data-independent

### D2: Ising-Like Synthetic
- p(x) ∝ exp(−β Σ_{⟨ij⟩} J_{ij} x_i x_j)
- J_{ij} sampled from N(0, 1/n); coupling graph = 2D grid
- ⟨Z_a⟩_p computed from Monte Carlo or exact for small n
- Purpose: structured pairwise correlations; tests how data correlations interact with gradient landscape

### D3: Structured Synthetic Binary Mixture
- p(x) = (1/K) Σ_{k=1}^K N(μ_k, ε²I) projected to {0,1}^n
- K=4 modes; μ_k are random binary vectors; ε = 0.1 (low noise)
- Purpose: multi-modal target; closer to real data; hardest for MMD² to capture with shallow IQP

**Scope note:** D1 is the primary experimental target for scaling (most controllable). D2 and D3 used for comparison in later weeks.

---

## 6. Initialization Schemes

All three are primary axes (not optional). For each (F, K) combination, all three inits are compared.

### I1: Uniform
- θ_i ~ U[−π, π]
- Worst-case / stress test initialization
- Expected to maximize BP severity

### I2: Small-Angle
- θ_i ~ N(0, σ_θ²)
- σ_θ ∈ {0.01, 0.1, 0.3} (three sub-settings)
- Main trainability hypothesis: small angles suppress BP via linearization
- For σ_θ → 0: ⟨Z_a⟩_{q_θ} ≈ first-order Taylor; gradients determined by linear response

### I3: Data-Dependent (Covariance-Based)
- θ initialized from empirical covariance of dataset D2 or D3
- θ_j ∝ (1/|D|) Σ_{x~p} (a · g_j mod 2) · (-1)^{x·g_j} for representative a
- Purpose: structured init aligned with target; tests if data-aware initialization avoids BP

---

## 7. Experiment Grid

Full grid: 6 families × 4 kernels × 3 inits × 6 n-values × ≥100 seeds

| Axis | Values |
|---|---|
| Circuit family (F) | F1–F6 |
| Kernel (K) | Gaussian, Laplacian, Polynomial (d=2), Linear |
| Kernel bandwidth (σ) | 3 values per kernel |
| Initialization (I) | I1 (Uniform), I2 (Small-angle, 3 σ_θ), I3 (Data-dep) |
| n (qubit count) | 16, 24, 32, 48, 64, 96 (+ 128 if time permits) |
| Seeds per setting | ≥ 100 (θ seeds); circuit seeds separate |
| Dataset target | D1 (primary), D2 and D3 (comparison) |

**Computational budget per setting:**
- num_a_samples (mixture): 512 (default); tunable up to 2048 for variance stability
- num_z_samples (IQP expectation): 1024 (default); scaled up to 4096 for large n

---

## 8. Barren Plateau Questions Per (Kernel, Connectivity) Pair

For each cell in the 6×4 grid, the organizing question:

```
BP(F_i, K_j)?  →  Var_{θ~I1}[∂_{θ} MMD²_{K_j}(p, q^{F_i}_θ)] = Θ(exp(−α·n))?
```

**Secondary questions per cell:**
1. Does I2 (small-angle) change the scaling exponent α?
2. Does I3 (data-dep) suppress variance relative to I1?
3. Does increasing σ in Gaussian/Laplacian change whether BP occurs?
4. Does degree d in polynomial kernel create a threshold effect?

**Expected outcomes by family:**
- F1 (bounded-degree): No BP expected for k ≤ 3 with Gaussian; TBD with polynomial
- F2 (ER sparse): No BP at p ~ 2/n; BP expected at p ~ 0.5
- F3 (lattice): No BP expected; gradient decays at most polynomially
- F4 (dense): BP expected for all kernels
- F5 (community): Partial BP; block-dependent; investigate kernel interaction
- F6 (symmetric): Unknown; symmetry may redistribute gradient mass

---

## 9. Success Criterion

By May 15, 2026: for each of the 24 (F, K) pairs, produce a stated conclusion of the form:

> "Under connectivity family F_i with kernel K_j and initialization I, Var(∂L) scales [exponentially / polynomially / constant] with n; small-angle init [does / does not] change this; shot/noise effects [do / do not] reinstate exponential suppression."
