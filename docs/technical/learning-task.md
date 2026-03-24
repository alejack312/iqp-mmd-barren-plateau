# The learning task

## What the model is

An IQP circuit with n qubits and m parameterized gates. Measurement gives a binary string of length n, drawn from the circuit's output distribution q_θ. Training adjusts the m parameters θ to make q_θ match some target distribution p.

Generating samples from q_θ requires running the actual circuit — on quantum hardware, or a statevector simulator for small n. Training does not. The MMD loss and its gradients are computable entirely classically, via:

```
⟨Z_a⟩_{q_θ} = E_{z~U({0,1}^n)}[ cos(Φ(θ, z, a)) ]
```

This is exact, not an approximation. See `docs/technical/iqp-classical-sampling.md` for the full derivation.

## What the training loop does

Given N binary strings drawn from p, the goal is to find θ minimizing:

```
MMD²_σ(p, q_θ) = C · Σ_{S ⊆ [n]} τ^|S| (⟨Z_S⟩_p − ⟨Z_S⟩_{q_θ})²
```

This is the weighted ℓ² distance between the Fourier spectra of p and q_θ. The Gaussian kernel with bandwidth σ gives each mode S a weight τ^|S| = tanh(1/σ²)^|S|, which decays exponentially with Hamming weight. Low-order correlations (single-qubit marginals, pairwise correlations) dominate the loss; high-order terms contribute very little unless σ is small. See `docs/technical/mmd-gaussian-fourier.md` for why this factorization falls out of the kernel structure.

In practice, the training loop estimates this with Monte Carlo:
1. Sample K Z-word masks a from the kernel's spectral distribution P_σ
2. For each a, compute ⟨Z_a⟩_p from the dataset (average parity of bits in the mask) and ⟨Z_a⟩_{q_θ} from the cosine formula
3. Average the squared differences
4. Differentiate with respect to θ

Everything runs on a CPU or GPU. The quantum circuit is only needed after training, to generate new samples.

## The three datasets

All datasets generate binary strings of length n on-the-fly at runtime — no files, no preprocessing.

**product_bernoulli** — each bit is independently Bernoulli(0.5). Every higher-order Fourier coefficient is exactly zero: ⟨Z_a⟩_p = 0 for all |a| ≥ 1. This makes it the cleanest test case for the barren plateau question. The "right answer" is known analytically, and the gradient depends purely on how well the circuit reproduces an uncorrelated product distribution. There's no structure in the data that could confound the circuit's gradient behavior.

**ising** — samples from a synthetic Ising model p(x) ∝ exp(−β Σ_{⟨ij⟩} J_{ij} x_i x_j) where the couplings J_{ij} ~ N(0, 1/n) are random and the interaction graph is a 2D grid. The data has non-zero pairwise correlations: ⟨Z_{e_i + e_j}⟩_p ≠ 0 for neighboring qubits i, j. This tests whether circuits with local connectivity can learn physically local correlations.

**binary_mixture** — K Gaussian clusters binarized per dimension, with each sample coming from one of K modes. The distribution is multi-modal and concentrated on K distinct regions of {0,1}^n. Capturing this with an IQP circuit likely requires high-order correlations, making it the hardest target for shallow circuits.

For the barren plateau scaling study, product_bernoulli is the primary target — most controllable, most analytically tractable. Ising and mixture are used for comparison once the product_bernoulli regime is fully characterized.

## What "success" means

There are two separate questions here that often get conflated.

**Generative modeling success:** the trained circuit's samples are hard to distinguish from the data. In practice: MMD² converges during training, and the learned distribution's single-qubit marginals and pairwise correlations match those of p.

**Scaling success (no barren plateau):** gradient variance does not decay exponentially in n. If:

```
Var_{θ~D}[∂_{θ_i} MMD²] ~ C · 2^{−α·n}   with α > 0
```

then gradients become exponentially small as the system grows. Even with perfect optimization machinery, learning becomes infeasible for large n — you can't distinguish a useful gradient from noise. If variance decays polynomially (α = 0, polynomial in n) or stays constant, training scales.

These goals are related but not the same. A circuit can train successfully at n = 16 but still have a barren plateau that prevents training at n = 64. The project's main output is a fitted scaling law for each (family, kernel, init) combination:

```
log Var ~ −α(F,K) · n + β(F,K) · log n + c(F,K)
```

α > 0 means exponential decay (barren plateau). α = 0, β < 0 is a polynomial decay (milder issue). α = β = 0 means gradient variance stays roughly constant — the trainable regime.

## Data-dependent initialization

The uniform and small-angle inits set θ without looking at the data. Data-dependent initialization uses training data statistics to start at a better point.

At θ = 0, the IQP circuit outputs |0⟩^⊗n deterministically (no gates applied, just H·H = I). Every Fourier coefficient is ⟨Z_a⟩_{q_0} = 1, since (-1)^{a·0} = 1 for any a. The data has Fourier coefficients spread across [−1, 1]. The initial MMD loss is therefore large regardless of the target distribution.

The design intent (from ScopeLock Section 6 and arXiv:2503.02934) is to set θ_j from the dataset's statistics:

```
θ_j  ∝  (1/N) Σ_{x ∈ D}  (a · g_j mod 2) · (−1)^{x · g_j}
```

for a representative Z-word a. This computes how much generator j's response (−1)^{g_j·x} correlates with the data, weighted by whether generator j is "relevant" to mode a (the factor (a·g_j mod 2)). If the data has a strong positive parity response along generator j's support, the init sets θ_j positive from the start.

The effect: instead of starting at θ = 0 (all Fourier modes = 1, maximum mismatch with most datasets), the circuit begins partially aligned with the target's low-order Fourier structure. It is a covariance-based warm start.

The theoretical motivation comes from the warm start literature (arXiv:2502.07889, Mhiri et al.), which shows that perturbations around favorable starting points can avoid exponential gradient suppression even when random initialization produces a barren plateau. Whether a data-dependent warm start constitutes a "favorable starting point" for IQP circuits, and for which circuit families this holds, is one of the questions the experiments aim to answer.

**Current status:** the config key `init.scheme: data_dependent` is reserved. The runner raises `NotImplementedError` until the implementation is written. The formula above describes the intended behavior.
