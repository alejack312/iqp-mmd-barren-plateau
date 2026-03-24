# MMD² with a Gaussian kernel: the Fourier picture

The existing doc [iqp-classical-sampling.md](./iqp-classical-sampling.md) explains *how* we compute
the MMD loss classically. This one explains *why* that works — specifically, why the
Gaussian-kernel MMD² decomposes naturally into a sum over Pauli Z-string expectation values, and
what that decomposition means for trainability and circuit design.

---

## The simple version first

Forget the formulas for a moment.

You have two bags of binary strings — the training data and your quantum model's samples. You
want to measure how different they are. The obvious way is to count how often each string appears
in each bag and compare. With n bits there are $2^n$ possible strings, so for n = 50 that's over
a quadrillion entries. You can't store that, let alone compare it.

So instead, ask simpler questions. Pick a random subset of bit positions — say, positions 2, 5,
and 9. For each sample, look only at those three bits and ask: is an odd or even number of them
equal to 1? Write down +1 for odd, −1 for even. Average those ±1 values across all samples in
bag A, then do the same for bag B. If the averages differ, the bags disagree on something
involving bits 2, 5, 9 jointly. That ±1 quantity is a **parity check** — cheap to compute from
raw samples, no probabilities needed.

The MMD loss does this for many random subsets at once and averages the squared differences. If
two distributions agree on the parity check for every possible subset, they're identical.

Now, checking all $2^n$ subsets equally is back to the original problem. The Gaussian kernel is
a way of weighting them: small subsets get high weight, large subsets get exponentially less. So
the loss mostly tests single-bit averages and two-bit correlations, occasionally three-bit ones,
rarely anything larger. The bandwidth $\sigma$ controls how fast the weights drop — wider $\sigma$
means you only care about the small, simple correlations.

The connection to quantum is: a parity check on subset S is *exactly* what the Pauli operator
$Z_S = Z \otimes Z \otimes \cdots$ measures. Run the quantum circuit, measure, and each shot
gives $(-1)^{\text{parity of selected bits}}$. Average those shots and you have $\langle Z_S \rangle$.
So the quantum expectation value and the statistical parity average are the same number.

That's the whole story. Because the MMD loss breaks into parity averages, and because IQP
circuits have a closed-form classical expression for those averages, you can train without
touching a quantum computer. Sample random subsets, compute parity averages from data and from
the formula, take squared differences, differentiate. That's it.

Everything below is the same argument written in proper math.

---

## Walsh characters are Pauli Z eigenvalues

Start with n-bit strings x ∈ {0,1}^n. For any subset S ⊆ {1,...,n} (encoded as a bitmask
a ∈ {0,1}^n), define the **parity function**:

$$\chi_S(x) = (-1)^{a \cdot x} = (-1)^{\sum_{i \in S} x_i}$$

These are the Walsh characters — the Fourier basis functions for {0,1}^n. They take values ±1
and are orthonormal under the uniform measure:

$$\frac{1}{2^n} \sum_{x \in \{0,1\}^n} \chi_S(x)\, \chi_T(x) = \delta_{S,T}$$

The quantum connection is immediate: the Pauli Z-string operator $Z_S = \bigotimes_{i \in S} Z_i$
has eigenvalue $\chi_S(x)$ on any computational basis state $|x\rangle$:

$$\langle x | Z_S | x \rangle = (-1)^{a \cdot x} = \chi_S(x)$$

So for any distribution p over bitstrings, the **Fourier coefficient at mode S** and the
**expectation value of Z_S** are the same thing:

$$\hat{p}(S) = \mathbb{E}_{x \sim p}[\chi_S(x)] = \mathbb{E}_{x \sim p}[\langle x | Z_S | x \rangle] = \langle Z_S \rangle_p$$

This is the bridge. Any question about the Fourier structure of the distribution p is
equivalently a question about which Pauli Z observables have non-zero expectation under p, and
vice versa.

Every distribution p has a unique Boolean Fourier expansion:

$$p(x) = \sum_{S \subseteq [n]} \hat{p}(S)\, \chi_S(x) = \sum_{a \in \{0,1\}^n} \langle Z_a \rangle_p\, (-1)^{a \cdot x}$$

The coefficients $\langle Z_a \rangle_p$ completely characterize p. Matching all of them between two
distributions is equivalent to matching the distributions exactly.

---

## Factoring the Gaussian kernel

The Gaussian kernel on binary strings (as used in this codebase) is:

$$k_\sigma(x, y) = \exp\!\left(-\frac{H(x, y)}{\sigma^2}\right)$$

where $H(x,y) = \#\{i : x_i \neq y_i\}$ is the Hamming distance. The key is to factor this over
the individual bits. Each bit contributes a factor:

$$k^{(i)}_\sigma(x_i, y_i) = \exp\!\left(-\frac{|x_i - y_i|}{\sigma^2}\right) = \begin{cases} 1 & x_i = y_i \\ e^{-1/\sigma^2} & x_i \neq y_i \end{cases}$$

Switch to ±1 encoding via $s_i = 1 - 2x_i$, $t_i = 1 - 2y_i$. Then $s_i t_i = +1$ when $x_i = y_i$
and $-1$ when $x_i \neq y_i$, so:

$$k^{(i)}_\sigma = \frac{1 + e^{-1/\sigma^2}}{2} + s_i t_i \cdot \frac{1 - e^{-1/\sigma^2}}{2}$$

Write this as $A_\sigma(1 + \tau\, s_i t_i)$ where:

$$A_\sigma = \frac{1 + e^{-1/\sigma^2}}{2}, \qquad \tau = \frac{1 - e^{-1/\sigma^2}}{1 + e^{-1/\sigma^2}} = \tanh\!\left(\frac{1}{2\sigma^2}\right)$$

So $\tau \in (0, 1)$ is the key parameter — it controls how fast the Fourier weights decay with
mode order. (The code uses the closely related $\tau' = \tanh(1/\sigma^2)$, which differs by
a reparametrization of $\sigma$ but has the same structural role.)

Multiplying over all n bits:

$$k_\sigma(x, y) = \prod_{i=1}^n A_\sigma(1 + \tau\, s_i t_i) = A_\sigma^n \prod_{i=1}^n (1 + \tau\, s_i t_i)$$

Expand the product:

$$\prod_{i=1}^n (1 + \tau\, s_i t_i) = \sum_{S \subseteq [n]} \tau^{|S|} \prod_{i \in S} s_i t_i = \sum_{S \subseteq [n]} \tau^{|S|}\, \chi_S(x)\, \chi_S(y)$$

where the last step uses $\prod_{i \in S} s_i \cdot \prod_{i \in S} t_i = \chi_S(x)\chi_S(y)$.

The full **Fourier-basis expansion of the Gaussian kernel** is:

$$\boxed{k_\sigma(x, y) = C \sum_{S \subseteq [n]} \tau^{|S|}\, \chi_S(x)\, \chi_S(y)}$$

where $C = A_\sigma^n$ is a normalization constant. The kernel is diagonal in the Walsh basis —
it only couples mode S on the x-side to the same mode S on the y-side, never mixing different
modes. The weight on each mode depends only on the Hamming weight $|S|$, not on which specific
qubits are in S. Since $\tau < 1$, higher-order modes are exponentially down-weighted. How fast
they decay depends on $\sigma$: a larger bandwidth gives a smaller $\tau$ and a steeper spectral
dropoff, so the kernel tests fewer Fourier modes.

---

## MMD² as a weighted sum over Z-word differences

Plug the kernel expansion into the MMD²:

$$\text{MMD}^2_\sigma(p, q) = \mathbb{E}_{x,x' \sim p}[k(x,x')] - 2\,\mathbb{E}_{x \sim p,\, y \sim q}[k(x,y)] + \mathbb{E}_{y,y' \sim q}[k(y,y')]$$

By linearity, each of the three terms factors over the Walsh basis:

$$\mathbb{E}_{x,x' \sim p}[k(x,x')] = C \sum_S \tau^{|S|} \langle Z_S \rangle_p^2$$

$$\mathbb{E}_{x \sim p,\, y \sim q}[k(x,y)] = C \sum_S \tau^{|S|} \langle Z_S \rangle_p \langle Z_S \rangle_q$$

Assembling them:

$$\boxed{\text{MMD}^2_\sigma(p, q) = C \sum_{S \subseteq [n]} \tau^{|S|} \left(\langle Z_S \rangle_p - \langle Z_S \rangle_q\right)^2}$$

This is the central formula. **The Gaussian-kernel MMD² is a weighted $\ell^2$ distance between the
Fourier spectra of p and q.** It measures how much the two distributions disagree on each
Z-string expectation value, with lower-order correlations weighted more heavily.

Note what this does *not* require: probabilities, density estimation, or any access to the full
$2^n$ distribution table. The loss is entirely determined by the expectations
$\langle Z_S \rangle_p$ and $\langle Z_S \rangle_q$ — which can both be estimated from samples.

---

## The spectral sampling picture

The sum over $2^n$ modes is intractable directly. But its structure suggests a natural Monte
Carlo strategy. Define the probability distribution:

$$P_\sigma(S) \propto \binom{n}{|S|} \tau^{|S|}$$

which is the normalized version of the weights. Then:

$$\text{MMD}^2_\sigma(p, q) \propto \mathbb{E}_{S \sim P_\sigma}\!\left[\frac{\left(\langle Z_S \rangle_p - \langle Z_S \rangle_q\right)^2}{\text{(combinatorial factor)}}\right]$$

To sample $S \sim P_\sigma$ efficiently:

1. Sample the **Hamming weight** $w \sim P(w) \propto \binom{n}{w} \tau^w$ — this is a
   binomial-like distribution over $\{0, 1, \ldots, n\}$.
2. Sample **which $w$ qubits** uniformly at random from all $\binom{n}{w}$ subsets of size $w$.

This two-step procedure is exactly what `gaussian_sample_a` in `mmd/kernel.py` implements.

The training loop then alternates between:
- Drawing a batch of modes $S_1, \ldots, S_K \sim P_\sigma$ (Z-word sampling)
- Estimating $\langle Z_{S_k} \rangle_p$ from the dataset (mean parity)
- Estimating $\langle Z_{S_k} \rangle_{q_\theta}$ from the IQP circuit formula (cosine formula)
- Averaging the squared differences and differentiating with respect to $\theta$

---

## Computing $\langle Z_S \rangle_{q_\theta}$ without a quantum computer

For IQP circuits, the Fourier coefficient $\langle Z_S \rangle_{q_\theta}$ has a closed classical
form. This is what makes the whole framework feasible. The formula (derived in the companion doc
[iqp-classical-sampling.md](./iqp-classical-sampling.md)) is:

$$\langle Z_S \rangle_{q_\theta} = \mathbb{E}_{z \sim \text{Uniform}(\{0,1\}^n)}\!\left[\cos\!\left(\Phi(\theta, z, S)\right)\right]$$

where the phase is:

$$\Phi(\theta, z, S) = 2 \sum_{j=1}^m \theta_j \cdot (S \cdot g_j \bmod 2) \cdot (-1)^{z \cdot g_j}$$

The inner quantity $(S \cdot g_j \bmod 2)$ determines whether mode S "sees" generator $g_j$: it is
1 iff S has odd overlap with the support of generator $j$, and 0 otherwise. Generators with even
overlap with S drop out of the phase entirely.

In matrix form: if $G$ is the $m \times n$ generator matrix (rows are the generators), then the
indicator vector $(S \cdot g_j \bmod 2)_{j=1}^m$ is just $G \cdot S \bmod 2$. So the phase is:

$$\Phi(\theta, z, S) = 2\, (G S \bmod 2)^T \cdot \big[(-1)^{Gz \bmod 2} \odot \theta\big]$$

Everything here is classical binary arithmetic and dot products.

---

## How circuit structure determines Fourier mode access

Not all Fourier modes $\langle Z_S \rangle_{q_\theta}$ are tunable. Mode S is **inert** — independent
of $\theta$ — when $\Phi(\theta, z, S) = 0$ for all $\theta$ and $z$. This happens when
$(S \cdot g_j \bmod 2) = 0$ for **every** generator $j$, i.e., S lies in the null space of
$G^T$ over $\mathbb{F}_2$:

$$\text{Inert modes} = \{S \in \{0,1\}^n : G S \equiv 0 \pmod{2}\}$$

The structure of inert modes depends directly on the circuit connectivity:

**Product state** ($G = I_n$, single-qubit Z gates):
$G S = S$ mod 2, so $S$ is inert only when $S = 0$. Every non-trivial Z-string has odd overlap
with at least one generator. The circuit can tune *all* Fourier modes — but it has only n
parameters (one per qubit), so modes with $|S| > 1$ are tunable only through correlated
combinations of single-qubit phases.

**2D lattice** (ZZ pairs on nearest neighbors, $g_{ij} = e_i + e_j$):
$(S \cdot g_{ij} \bmod 2) = (S_i + S_j) \bmod 2$, which is 1 iff exactly one of $\{i, j\}$ is
in S. Modes where S has even degree in the interaction graph (every qubit in S has an even number
of neighbors also in S) are inert.

**Complete graph** (all $\binom{n}{2}$ ZZ pairs):
Any S with $|S| \geq 2$ has many generators with odd overlap, giving many parameters controlling
the mode. Single-qubit modes $|S|=1$ need a single-qubit Z gate if one is included, otherwise
they may also be inert.

**Erdős–Rényi or bounded-degree** (random sparse hyperedges):
The inert subspace is determined by the $\mathbb{F}_2$ row-null space of $G$. For a random sparse
$G$, this is typically small (just $\{0\}$ for most draw), but structured families can have
nontrivial null spaces.

The practical implication: **the circuit can only improve the MMD by adjusting the modes it has
access to**. If the Gaussian kernel puts significant weight on modes that are inert for the given
circuit family, training hits a floor regardless of optimization quality.

---

## What bandwidth does to the loss

$\tau = \tanh(1/(2\sigma^2))$ decreases from 1 to 0 as $\sigma$ increases. This directly controls
which Fourier modes the loss cares about:

**Small $\sigma$ (narrow kernel, $\tau \approx 1$):**
All modes contribute roughly equally. MMD approaches total variation distance.
The loss "sees" all correlations — $k$-body terms for every $k$. But the gradient
is dominated by whatever the circuit can most easily tune, and high-body terms
(which are harder to affect with local gates) get equal treatment to low-body ones.
Per [Rudolph et al. 2023, arXiv:2305.02881], this regime can produce untrainable losses
for generic circuit families because the effective observable is global.

**Large $\sigma$ (broad kernel, $\tau \approx 1/(2\sigma^2) \ll 1$):**
Only low-order modes ($|S| = 1, 2, \ldots$) contribute appreciably. The loss is
**low-bodied**: it measures discrepancy in marginals and low-order correlations. Gradient
variance scales better because the effective observable is local. The same paper shows that
for $\sigma \in \Theta(\sqrt{n})$ (so $\tau \sim 1/n$), the MMD is trainable for
shallow IQP circuits.

The catch: a low-bodied loss can be fooled. Two distributions that agree on all $k$-th order
marginals but differ in $(k{+}1)$-th order correlations will look identical to the MMD with
a wide-enough kernel. Whether this matters depends on the dataset.

So $\sigma$ controls a genuine trade-off between expressivity and trainability. This is specific
to the MMD — explicit losses like KL divergence have no analogous tuning knob and fail for
entirely different reasons (probability estimation from samples).

---

## The gradient formula, in Fourier language

Differentiating $\text{MMD}^2_\sigma(p, q_\theta)$ with respect to parameter $\theta_i$:

$$\partial_{\theta_i} \text{MMD}^2 = -2C \sum_S \tau^{|S|} \left(\langle Z_S \rangle_p - \langle Z_S \rangle_{q_\theta}\right) \partial_{\theta_i} \langle Z_S \rangle_{q_\theta}$$

And the per-mode gradient is (from `mmd/gradients.py`):

$$\partial_{\theta_i} \langle Z_S \rangle_{q_\theta} = -2\, (S \cdot g_i \bmod 2)\, \mathbb{E}_{z \sim U}\!\left[\sin(\Phi(\theta, z, S)) \cdot (-1)^{z \cdot g_i}\right]$$

If generator $i$ has even overlap with S, its gradient is zero — it cannot affect mode S. The
total gradient variance $\text{Var}_\theta[\partial_{\theta_i} \mathcal{L}]$ therefore depends on
three things: how many modes have odd overlap with $g_i$ (fixed by the hypergraph), what
$\tau^{|S|}$ is for those modes (fixed by the bandwidth), and how large
$|\langle Z_S \rangle_p - \langle Z_S \rangle_{q_\theta}|$ is for them (depends on initialization
and target data).

The barren plateau question for this project — whether gradient variance decays exponentially in
$n$ — translates directly to: as we scale up $n$, do the Fourier modes that drive the gradient
collapse to a measure-zero set, or remain accessible?

---

## Summary

| Concept | In terms of Z-strings / Fourier | In terms of circuits |
|---|---|---|
| Gaussian kernel | Diagonal in Walsh basis with weights $\tau^{\|S\|}$ | — |
| MMD²(p, q) | Weighted $\ell^2$ distance between Fourier spectra | Measurable from samples |
| Mode weight | $\tau^{\|S\|}$, depends only on $\|S\|$ | Controlled by bandwidth $\sigma$ |
| Mode access | Z-words with odd overlap with some $g_j$ | Determined by hypergraph $G$ |
| Trainability | Gradient variance over active, low-weight modes | Depends on $G$ + $\sigma$ interaction |

The full derivation in one chain:

$$k_\sigma(x,y) = C\sum_S \tau^{|S|}\chi_S(x)\chi_S(y)$$

$$\Rightarrow\quad \text{MMD}^2_\sigma(p,q) = C\sum_S \tau^{|S|}\!\left(\langle Z_S \rangle_p - \langle Z_S \rangle_q\right)^2$$

$$\Rightarrow\quad \langle Z_S \rangle_{q_\theta} = \mathbb{E}_z\!\left[\cos\!\left(\Phi(\theta,z,S)\right)\right]$$

The Gaussian kernel's product structure over bits is what collapses the $2^n$-dimensional MMD
expression into a sum over Z-word expectation values. IQP circuits are special because those
expectation values are classically computable via the cosine formula. Neither property holds
for generic kernels or generic circuits.

---

## Further reading

- arXiv:2305.02881 — Rudolph et al., *Trainability barriers and opportunities in quantum
  generative modeling*: derives trainability conditions (Theorems 2 and 3 in that paper), proves
  that the MMD is an expectation value of a quantum observable, and relates body-ness to barren
  plateau behavior.
- arXiv:2503.02934 — Recio-Armengol, Ahmed, Bowles, *Train on classical, deploy on quantum*:
  the paper this codebase implements; demonstrates 1000-qubit training and analyzes
  data-dependent initialization as a way to avoid the barren plateau regime from the start.
- [iqp-classical-sampling.md](./iqp-classical-sampling.md): the companion doc, which covers
  the cosine formula for $\langle Z_S \rangle_{q_\theta}$ in detail.
