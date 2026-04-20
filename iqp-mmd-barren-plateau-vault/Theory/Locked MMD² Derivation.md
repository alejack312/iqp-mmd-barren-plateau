---
title: Locked MMD² Derivation
tags:
  - theory
  - convention
  - lock
---

# Locked MMD² Derivation

The specific analytic form of the MMD² loss this project is committed to, how to get from the standard definition down to something a classical algorithm can estimate, where the Gaussian bandwidth $\sigma$ enters, where the IQP connectivity $G$ enters, and how all of this feeds into a variance expression you can actually reason about.

"Locked" means the team has fixed a concrete formulation for the current study phase. It is not a theorem, it is a shared contract between the theory files and the code. See [[Theory-Implementation Parity]] and [[Gaussian Convention]] for the full history of why the contract exists.

---

## Why a lock

> [!important] One formula, everywhere
> - Theory needs one unambiguous formula for proofs and scaling arguments.
> - Implementation needs one unambiguous formula for `sample_a()`, kernel weights, and estimators.
> - If these drift, the Monte Carlo estimator is no longer sampling from the kernel it claims to use.

The project went through a period where different files quietly used different meanings of `sigma` and different Walsh decay constants. The lock is the discipline that closed that drift.

---

## The locked Gaussian formula

For the Gaussian kernel case, the form the code implements is:

$$
\mathrm{MMD}^2_\sigma(p, q_\theta; G) \;=\; C \sum_{a \subseteq \{0,1\}^n} \tau^{|a|}\Big(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta}\Big)^2
$$

with:

- Gaussian kernel $k(x, y) = \exp(-H(x,y)/(2\sigma^2))$, where $H$ is the Hamming distance
- Walsh decay constant $\tau = \tanh(1/(4\sigma^2))$
- Normalization $C$ such that $\sum_a \tau^{|a|}/C^{-1} = 1$ after choosing a sampling distribution
- Sampling distribution over Z-word masks $a$: $P(a) \propto \tau^{|a|}$, with weight-stratified sampling

All three of `gaussian_kernel`, `gaussian_spectral_weights`, and `gaussian_sample_a` agree on this convention. See [[Gaussian Convention]].

The rest of this note explains *why* this formula looks the way it does, starting from the textbook MMD and walking down to the form above.

---

## How the MMD loss becomes an observable

This is the derivation from Recio-Armengol et al. 2025 ([2503.02934](https://arxiv.org/abs/2503.02934), Sec. 3.3 and Sec. 4) rewritten in plain terms. The goal is to take an object that looks like an expensive distribution-matching integral and reshape it into a *classical observable you can estimate with Monte Carlo over bitstrings*.

### Step 1: the textbook definition

The squared maximum mean discrepancy between two distributions $p$ and $q$ on $\{0,1\}^n$ is (Recio-Armengol et al. 2025, Eq. 2):

$$
\mathrm{MMD}^2(p, q) \;=\; \mathbb{E}_{x,x' \sim p}[k(x, x')] \;-\; 2\,\mathbb{E}_{x \sim p,\, y \sim q}[k(x, y)] \;+\; \mathbb{E}_{y, y' \sim q}[k(y, y')]
$$

This is a distance between $p$ and $q$ measured in the "feature space" of the kernel $k$. If the kernel is *characteristic* (which the Gaussian is), then $\mathrm{MMD}^2 = 0$ if and only if $p = q$.

This formula is exactly the wrong shape for our purposes. It requires pairs of samples from $q$, and $q$ is an IQP output distribution that we cannot sample efficiently on a classical computer. If you try to estimate it by simulating the circuit, you are back to exponential cost. We need a different form.

### Step 2: expand the kernel into parity characters (Walsh decomposition)

The Gaussian kernel on bit strings, $k(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))$, is a function of the Hamming distance $H(x,y) = \|x-y\|^2$. Any such function has a *Walsh decomposition*: it can be written as a sum over "parity characters" $\chi_a(x) = (-1)^{a\cdot x}$ indexed by subsets $a \in \{0,1\}^n$:

$$
k(x, y) \;=\; \sum_{a \in \{0,1\}^n} w_k(a)\,\chi_a(x)\,\chi_a(y)
$$

For the Gaussian, the weights factor nicely across qubits and come out as

$$
w_k(a) \;\propto\; \tau^{|a|}, \qquad \tau \;=\; \tanh\!\left(\frac{1}{4\sigma^2}\right)
$$

where $|a|$ is the Hamming weight of $a$ (just the number of 1s). See [[Kernel Spectral Decomposition]] for the full derivation of this step.

The important property is that each weight is a constant (not a function of $x$ or $y$) and the $x, y$ dependence has been pushed into the two characters $\chi_a(x)$ and $\chi_a(y)$. That is exactly what you need for the next step to work.

### Step 3: turn character averages into Pauli-Z expectations

For any distribution $p$ on bit strings, the average of the parity character $\chi_a$ over $x \sim p$ is *exactly* the Pauli-Z expectation of the observable $Z_a$:

$$
\mathbb{E}_{x \sim p}[\chi_a(x)] \;=\; \mathbb{E}_{x \sim p}[(-1)^{x \cdot a}] \;=\; \langle Z_a \rangle_p
$$

The right-hand side is a standard observable: $Z_a$ is the tensor product of Pauli $Z$ operators on the qubits picked out by $a$, i.e. $Z_a = \bigotimes_{i : a_i = 1} Z_i$, and its expectation value is whatever a measurement of that operator would give on average. For the data distribution it is just the empirical parity average.

Plugging Steps 2 and 3 into Step 1:

$$
\mathbb{E}_{x,x' \sim p}[k(x,x')] \;=\; \sum_a w_k(a)\,\langle Z_a\rangle_p^2
$$

and similarly for the other two terms. The cross term becomes $\sum_a w_k(a)\,\langle Z_a\rangle_p\,\langle Z_a\rangle_q$. Putting it all together and completing the square:

$$
\boxed{\;\mathrm{MMD}^2(p, q) \;=\; \sum_{a \in \{0,1\}^n} w_k(a)\,\big(\langle Z_a\rangle_p - \langle Z_a\rangle_q\big)^2\;}
$$

or equivalently, interpreting the normalized weights as a probability distribution $P_k(a) \propto w_k(a)$ over Z-word masks,

$$
\mathrm{MMD}^2(p, q) \;=\; \mathbb{E}_{a \sim P_k}\!\Big[\big(\langle Z_a\rangle_p - \langle Z_a\rangle_q\big)^2\Big]
$$

That last expression is the *mixture-of-observables form* (Recio-Armengol et al. 2025, Prop. 2, Eq. 16). In plain English: *the MMD² is the average squared error in matching a random Pauli-Z observable between $p$ and $q$, where the randomness over which observable picks out which observable is dictated by the kernel.*

The big conceptual shift in going from Step 1 to this form is that the loss has been re-expressed as an expectation over *observables*, not over *samples*. That is what "turning the MMD into an observable" means. Sampling $q$ is hard, but computing a single Pauli-Z expectation on an IQP circuit is easy, so if you can sample observables you can compute the loss.

### Step 4: classical Pauli-Z expectation for an IQP circuit

The last ingredient is the classical estimator for $\langle Z_a\rangle_{q_\theta}$ when $q_\theta$ is the output of a parameterized IQP circuit. Recio-Armengol et al. 2025 (Prop. 1, derived from den Nest 2010) gives the formula:

$$
\langle Z_a\rangle_{q_\theta} \;=\; \mathbb{E}_{z \sim U(\{0,1\}^n)}\!\Big[\cos\!\big(\Phi(\theta, z, a; G)\big)\Big]
$$

with the phase

$$
\Phi(\theta, z, a; G) \;=\; 2\sum_{j=1}^{m} \theta_j\,(a \cdot g_j \bmod 2)\,(-1)^{z \cdot g_j}
$$

The only thing the formula needs from the IQP circuit is the *generator matrix* $G$: its rows $g_j$ are the bitmasks of which qubits each rotation touches. Everything else is parity arithmetic and a cosine average. See [[IQP Expectation]] for the derivation sketch.

The Monte Carlo estimate just replaces the uniform expectation over $z$ with an empirical mean, which has variance at most $1/|Z|$ because the cosine is bounded. No state vector. No $2^n$ work.

### Putting the pieces together

Combining Steps 3 and 4, the locked form of the Gaussian MMD² in terms of the things the code actually computes is:

$$
\boxed{\;
\mathrm{MMD}^2_\sigma(p, q_\theta; G)
\;=\; \sum_{a \in \{0,1\}^n}\;
\underbrace{\frac{\tau^{|a|}}{(1+\tau)^n}}_{\text{from }\sigma}
\Big(
\underbrace{\langle Z_a\rangle_p}_{\text{data parity}}
-
\underbrace{\mathbb{E}_{z \sim U}\!\big[\cos \Phi(\theta, z, a; G)\big]}_{\text{IQP parity, depends on }G}
\Big)^2
\;}
$$

I split the dependency into three labeled pieces because that's exactly how the loss breaks up in practice. The bandwidth $\sigma$ controls the weights. The generator matrix $G$ controls the IQP side through the phase. The training data controls the data side.

The normalization factor $(1+\tau)^n$ is what turns the weights $\tau^{|a|}$ into a bona fide probability distribution over Z-word masks, which is what the `gaussian_sample_a` sampler draws from.

---

## How bandwidth $\sigma$ enters

The bandwidth enters in exactly one place: the Walsh decay constant

$$
\tau \;=\; \tanh\!\left(\frac{1}{4\sigma^2}\right) \;\in\; (0, 1)
$$

which is the probability ratio at which higher-weight Z-words get suppressed in the mixture. Two limits worth keeping in mind.

**Small $\sigma$ (narrow kernel, $\tau \to 1$).** The mixture $P_k(a) \propto \tau^{|a|}$ becomes flat over all Z-word masks, and the sampled Hamming weight concentrates near $n/2$. The loss is now dominated by many-body parity observables, so the MMD² is effectively probing *high-order correlations* of the data distribution. In Recio-Armengol et al. 2025's language this is the regime where $|a|$ is peaked around $np$ with $p$ near $1/2$.

**Large $\sigma$ (wide kernel, $\tau \to 0$).** The weights collapse onto low-weight masks, and the expected Hamming weight goes to zero. The loss is dominated by single-qubit and two-qubit expectations, so the MMD² is probing *low-order correlations*. This is the regime where the kernel "smooths over" fine distinctions between distributions.

A useful way to see both limits at once: Recio-Armengol et al. 2025 (Prop. 2) writes the sampling distribution as a product of independent Bernoullis, each bit of $a$ being 1 with probability

$$
p_{\sigma} \;=\; \frac{1 - e^{-1/(2\sigma^2)}}{2}
$$

The relation between this Rudolph-2025 Bernoulli parameter and the project's Walsh decay constant is

$$
p_{\sigma} \;=\; \frac{\tau}{1+\tau}, \qquad \tau \;=\; \frac{p_{\sigma}}{1 - p_{\sigma}}
$$

which is just a reparameterization. Both conventions describe the same distribution on Z-word masks. The project chose $\tau$ because the product form $\tau^{|a|}$ lines up cleanly with the Walsh character decomposition, which is what the sampler and the spectral weight functions are actually computing. See [[Gaussian Convention]] for the full dictionary.

The practical takeaway is that *changing $\sigma$ changes which observables the loss is paying attention to*. A run at one bandwidth is asking a different question than a run at another bandwidth. This is why Recio-Armengol et al. 2025 trains against an average over several bandwidths, and why this project treats $\sigma$ as a sweep axis (see [[Multi-Scale Gaussian Kernel]]).

---

## How connectivity $G$ enters

The generator matrix $G$ is hiding inside the phase

$$
\Phi(\theta, z, a; G) \;=\; 2 \sum_{j=1}^{m} \theta_j\,(a \cdot g_j \bmod 2)\,(-1)^{z \cdot g_j}
$$

and shows up in the loss *only* through the IQP expectation $\langle Z_a\rangle_{q_\theta}$. Concretely, $G$ enters through two parity products:

1. $(a \cdot g_j) \bmod 2$ asks "does Z-word $a$ overlap *oddly* with generator $g_j$?" The answer is 1 or 0. If it is 0, generator $j$ contributes nothing to this Z-word's phase, no matter what $\theta_j$ is.
2. $(-1)^{z \cdot g_j}$ is the sign that generator $j$ picks up on a random bitstring $z$.

Put them together and the sum in $\Phi$ only ranges over the generators that actually "see" the observable $Z_a$. The set of those generators is

$$
S_a(G) \;=\; \{j : g_j \cdot a \equiv 1 \pmod 2\}
$$

and the size $|S_a(G)|$ is the number of parameters that contribute to $\langle Z_a\rangle_{q_\theta}$ for this particular mask. When the set is small, only a handful of knobs move that observable. When the set is large, most of the circuit is involved.

This is where the connectivity family ([[Families MOC]]) actually matters for the loss landscape. The same formula runs for every family, but the *distribution* of $|S_a(G)|$ across sampled $a$s is different:

- **Product state family** ($G = I_n$): for a mask of Hamming weight $|a| = k$, $|S_a(G)|$ is exactly $k$. The number of participating parameters tracks the Z-word weight directly.
- **2D lattice family**: for a mask $a$, $|S_a(G)|$ is the number of lattice edges that have an odd overlap with $a$. Geometric masks (connected blobs) see few edges cut. Random masks see many.
- **Complete graph family**: for any $a$ with $|a| \geq 1$, $|S_a(G)|$ is $|a|(n - |a|) + \binom{|a|}{2}$-ish, which is $O(n)$ to $O(n^2)$. Almost every parameter touches almost every observable. This is the family where the variance story gets hard fastest.
- **Erdős–Rényi family**: $|S_a(G)|$ is a random variable whose distribution depends on both the sampled edge set and on $a$.

See [[Generator Matrix]] for the shapes and [[Gradient Derivation#Step 1]] for the parity-gate interpretation in the derivative.

The short version: *bandwidth $\sigma$ controls which observables the loss asks about; connectivity $G$ controls how many parameters each of those observables depends on*. Those two choices are independent in the formula but coupled through the variance, which is what we go to next.

---

## General variance expression

The thing the project actually measures to diagnose barren plateaus is the variance of the MMD² *gradient*, not of the MMD² itself. Both are worth writing down, because they behave differently.

### Variance of the loss

Start from the mixture form. Since $\langle Z_a\rangle_p$ does not depend on $\theta$, the only $\theta$-dependent piece inside the expectation is $\langle Z_a\rangle_{q_\theta}$. So

$$
\mathrm{Var}_\theta[\mathrm{MMD}^2] \;=\; \mathrm{Var}_\theta\!\left[\,\mathbb{E}_{a \sim P_k}\!\big[(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta})^2\big]\,\right]
$$

Expanding the square and pulling the constant-in-$\theta$ term out,

$$
\mathrm{Var}_\theta[\mathrm{MMD}^2] \;=\; \mathrm{Var}_\theta\!\left[\,\mathbb{E}_{a \sim P_k}\!\big[\langle Z_a\rangle_{q_\theta}^2 - 2\langle Z_a\rangle_p\langle Z_a\rangle_{q_\theta}\big]\,\right]
$$

This is a single-number diagnostic that depends on both the kernel (through $P_k$, which is where $\sigma$ enters) and on the circuit family (through $\langle Z_a\rangle_{q_\theta}$, which is where $G$ enters).

### Variance of the gradient (the BP metric)

The per-parameter gradient is derived in [[Gradient Derivation]]:

$$
\partial_{\theta_i}\mathrm{MMD}^2 \;=\; -2\,\mathbb{E}_{a \sim P_k}\!\left[\,\big(\langle Z_a\rangle_p - \langle Z_a\rangle_{q_\theta}\big)\,\partial_{\theta_i}\langle Z_a\rangle_{q_\theta}\,\right]
$$

with the single-observable gradient

$$
\partial_{\theta_i}\langle Z_a\rangle_{q_\theta} \;=\; -2\,(a \cdot g_i \bmod 2)\,\mathbb{E}_{z \sim U}\!\left[\sin(\Phi(\theta, z, a; G))\,(-1)^{z \cdot g_i}\right]
$$

Two structural observations fall out of this immediately.

**Observation 1 — the parity gate.** The factor $(a \cdot g_i \bmod 2)$ is zero unless generator $g_i$ overlaps oddly with the Z-word $a$. In the gradient variance, this means *only the Z-words in $P_k$ for which $i \in S_a(G)$ contribute at all*. Every other sampled $a$ contributes exactly zero to $\partial_{\theta_i}$. So the effective sampling distribution over "useful" Z-words is

$$
P_k(a \mid i) \;\propto\; P_k(a)\,\mathbb{1}[i \in S_a(G)]
$$

and the fraction of $P_k$'s mass that falls on useful masks depends on both the kernel *and* the family through $G$.

**Observation 2 — the cosine collapse.** Under a $\theta$ distribution that is uniform on $[-\pi, \pi]$, the expectations $\mathbb{E}_\theta[\cos(2\theta_j)]$ and $\mathbb{E}_\theta[\sin(2\theta_j)]$ are zero, and $\mathbb{E}_\theta[\cos^2(2\theta_j)] = \mathbb{E}_\theta[\sin^2(2\theta_j)] = 1/2$. For a family where the IQP expectation $\langle Z_a\rangle$ is a *pure product of cosines* over $S_a(G)$ (the "interference-free" case of Recio-Armengol et al. 2025 Eq. 27 with $\Lambda = \emptyset$), this gives a clean closed-form result:

$$
\mathrm{Var}_\theta[\langle Z_a\rangle_{q_\theta}] \;=\; \mathbb{E}_\theta\!\left[\prod_{j \in S_a(G)}\cos^2(2\theta_j)\right] - \left(\mathbb{E}_\theta\!\left[\prod_{j \in S_a(G)}\cos(2\theta_j)\right]\right)^2 \;=\; \left(\frac{1}{2}\right)^{|S_a(G)|}
$$

This is a rewrite of Recio-Armengol et al. 2025 Eq. 49 in our notation. The variance of the observable decays exponentially in the number of generators that see it. When $|S_a(G)|$ grows linearly with $n$ (as it does in the complete-graph family for single-qubit observables), you get the classical barren plateau signature: exponential suppression in $n$.

### Combining them into the gradient variance

Putting Observation 1 and Observation 2 together gives a general structural form for the MMD² gradient variance in the interference-free regime:

$$
\boxed{\;\mathrm{Var}_\theta[\partial_{\theta_i}\mathrm{MMD}^2] \;\sim\; 4\,\mathbb{E}_{a, a' \sim P_k}\!\left[\,\mathbb{1}[i \in S_a]\,\mathbb{1}[i \in S_{a'}]\,C_{a,a'}(G, \theta_{\text{dist}})\,\right]\;}
$$

where $C_{a, a'}$ is a covariance-like term that collects the second-moment contribution of the two sampled observables and depends on the symmetric difference $S_a \triangle S_{a'}$ (which parameters the two Z-words share and which they do not). The key point is the structural dependency:

- Bandwidth $\sigma$ enters through $P_k$, controlling how often each Z-word mask is drawn.
- Connectivity $G$ enters through $S_a$ and $S_{a'}$, controlling how big the participating-parameter sets are and how much they overlap.
- Initialization $\theta_{\text{dist}}$ enters through the cosine moments inside $C_{a, a'}$.

In the specific case where $C_{a, a'}$ reduces to a product of independent cosine moments, each factor contributes $1/2$, so the whole thing scales like $\mathbb{E}_{a, a'}[2^{-|S_a \cup S_{a'}|}]$ up to constants. That is the quantity the project's [[Gradient Variance|gradient variance scalar]] $V(i; \theta_{\text{dist}}, F, K, n)$ is ultimately tracking as $n$ grows.

### What the paper proves vs what the project measures

Recio-Armengol et al. 2025 does not prove a general variance formula for the MMD² loss. They *derive* Eq. 49 for the special case of a single-qubit observable $Z_1$ in an all-to-all two-qubit gate circuit, observe that it decays exponentially under uniform init, and then argue informally that higher-order expectations like $\langle Z_1 Z_2\rangle$ need not decay the same way because the set $\Lambda$ in Eq. 27 becomes exponentially large and the interference terms start mattering. They do not close the loop.

This is exactly the gap the project is set up to close empirically. The scalar $V$ tracked by [[Scaling Runner]] is, morally, the quantity on the right-hand side of the boxed formula above, measured at finite $n$ across families and kernels, without assuming the interference terms cancel.

---

## What "locked" does NOT mean

- **Not mathematically proven forever.** The locked formula is the one the project is comparing against during this study phase. If the derivation is later tightened, the lock moves.
- **Not all kernels.** Gaussian is locked. Laplacian is explicitly stubbed until its decomposition is derived. Multi-scale Gaussian is validated but not fully exact-mixture-checked.
- **Not frozen as a test.** Tests check invariants like "sum of weights equals something sensible" and "MC path agrees with exact on small $n$," not frozen floats.

---

## Remaining lock work

From [[TODO Roadmap|T2]]:

- Laplacian MMD² decomposition derivation + lock
- Multi-scale Gaussian exact mixture validation
- Close the gap between Recio-Armengol et al. 2025 Eq. 49 and a true variance bound for $\partial_{\theta_i}\mathrm{MMD}^2$ in the interference-full regime (currently an open numerics-first question; see [[Gradient Variance]])

---

## Related

- [[MMD Loss]] — the top-level note on the loss
- [[Kernel Spectral Decomposition]] — the Walsh decomposition step
- [[Gaussian Convention]] — the $\tau \leftrightarrow p_\sigma$ dictionary
- [[IQP Expectation]] — Step 4 in full
- [[Gradient Derivation]] — where $\partial_{\theta_i}\mathrm{MMD}^2$ comes from
- [[Gradient Variance]] — the scalar the [[Scaling Runner]] measures
- [[Generator Matrix]] — the $G$ that controls the IQP side
- [[Families MOC]] — how $G$ is built family by family
- [[Barren Plateaus]] — the failure mode the variance expression is diagnosing
- [[Theory-Implementation Parity]]
- [[Glossary#Locked]]
- [[References#Recio-Armengol et al. 2025 — Scaling GQML]] — the source paper for the full derivation (not "Rudolph 2025" — earlier drafts of this vault mislabeled it)
