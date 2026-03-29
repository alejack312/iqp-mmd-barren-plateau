# Glossary

This page collects the project-specific terms that recur across the technical docs, SMART scope,
and code comments. The goal is to keep the vocabulary stable as the implementation catches up to
the locked study design.

## Locked

In this repo, **locked** means "the version currently fixed for the agreed study scope." It does
not mean mathematically proven forever; it means the team has chosen a concrete formulation to
implement, document, and compare against during the current phase.

Examples:
- **locked SMART scope**: the currently approved experiment scope from `docs/SMART-spec.md`
- **locked experiment axes**: the current set of sweep dimensions such as family, kernel, init,
  and system size
- **locked MMD^2 derivation**: the specific analytical form of the loss we intend the code to
  match exactly

<a id="locked-mmd2-derivation"></a>
## Locked MMD^2 derivation

The **locked MMD^2 derivation** is the reference formula for the loss used in the present study.
For the Gaussian-kernel case, that means the Walsh/Fourier decomposition of

`MMD^2_sigma(p, q) = C * sum_S tau^|S| ( <Z_S>_p - <Z_S>_q )^2`

plus the associated sampling distribution over Z-word masks `S`.

Why this matters:
- theory needs one unambiguous formula for proofs and scaling arguments
- implementation needs one unambiguous formula for `sample_a()`, kernel weights, and estimators
- comments like "confirm the exact Gaussian spectral normalization" mean some constants or
  parameterizations are not fully pinned to that reference form yet

At the moment, the project has the structure of this derivation documented, but some exact
normalization details are still marked TODO in code.

## Theory/implementation parity

**Theory/implementation parity** means the formula in the docs and the formula in code are the
same object, not merely qualitatively similar. If the theory says the Gaussian spectrum uses one
normalization or one bandwidth parameterization, the implementation should expose the same choice
explicitly.

<a id="iqp-circuit-family"></a>
## IQP circuit family

A **family** is a rule for constructing the generator matrix `G` as the system size `n` changes.
Each family defines a different connectivity pattern, and therefore a different trainability
regime.

In this project, the primary sweep uses four families:
- product state
- ZZ lattice
- sparse Erdos-Renyi
- complete graph

<a id="generator-matrix"></a>
## Generator matrix `G`

The **generator matrix** `G` is the binary matrix that specifies which qubits each IQP generator
acts on. It has shape `(m, n)`:
- `n`: number of qubits
- `m`: number of generators or parameterized interaction terms

Row `j` is the bitmask `g_j` for the generator `exp(i theta_j X^{g_j})`. In practice, almost all
of the classical formulas reduce to binary arithmetic involving `G`.

<a id="binary-matrix"></a>
## Binary matrix

A **binary matrix** is a matrix whose entries are only `0` or `1`. In this project, `G` is
binary because each entry answers a yes/no question:
- `1`: qubit `i` is included in generator `j`
- `0`: qubit `i` is not included in generator `j`

The point of using a binary matrix is that overlap questions become cheap matrix arithmetic modulo
2.

<a id="generator"></a>
## Generator / hyperedge

A **generator** (often also called a **hyperedge**) is one row of `G`. It identifies the subset
of qubits touched by one IQP interaction term.

More concretely, a generator does three jobs at once:
- it defines one interaction pattern in the circuit
- it identifies which parameter `theta_j` controls that interaction
- it determines which Z-word modes can "see" that interaction through `(a dot g_j mod 2)`

So a generator is not just bookkeeping. It is the basic structural unit that tells the model where
correlations can be created and which Fourier modes each parameter can influence.

Examples:
- weight 1 generator: a single-qubit term
- weight 2 generator: a pairwise `ZZ`-style interaction
- higher-weight generator: a many-body interaction

When people in this project talk about "local generators," "dense generators," or "generator
overlap," they are talking about how these rows of `G` are arranged and how much they intersect.

## Product-state family

The **product-state family** is the no-entanglement baseline. Its generator matrix is the
identity, so each parameter controls only one qubit. If this family showed severe barren plateau
behavior, that would be suspicious, because it is the simplest and most local case.

<a id="zz-lattice-family"></a>
## ZZ lattice family

The **ZZ lattice family** is the intended local-interaction baseline for the locked study. It
means pairwise nearest-neighbor interactions arranged on a lattice, usually a 2D square grid.

Important distinction:
- the current code exposes a generic `lattice` family with 1D and 2D local patches
- some comments refer more specifically to the **exact nearest-neighbor ZZ lattice family** that
  the SMART scope wants as the canonical local family

So when you see "replace generic 2D patch sampler with the exact nearest-neighbor ZZ lattice
family," it means the code's current local-family approximation should be tightened to the
pairwise lattice used by the locked study design.

<a id="sparse-erdos-renyi-family"></a>
## Sparse Erdos-Renyi family

The **sparse Erdos-Renyi family** is the random-connectivity baseline. Each candidate qubit is
included in each generator independently with some small probability, so the resulting interaction
pattern is irregular rather than geometric.

Why "sparse" matters:
- plain Erdos-Renyi can become too dense as `n` grows
- the study usually wants bounded or slowly growing expected degree, so random connectivity stays
  meaningfully comparable to the local and dense baselines

In code, this appears as `erdos_renyi`; comments noting future calibration mean the current
sampler still needs to be tuned to the exact sparse regime intended by the SMART spec.

## Complete-graph family

The **complete-graph family** is the densest pairwise baseline. Every qubit pair interacts, so
the generator matrix contains one weight-2 row for each pair `(i, j)`.

This is the all-to-all connectivity extreme and is often the regime where gradient concentration is
expected to be strongest.

<a id="pairwise-baseline"></a>
## Pairwise baseline

A **pairwise baseline** is a reference circuit family built only from two-qubit interactions.

It is called a baseline because it gives a clean comparison point between:
- local pairwise structure
- random pairwise structure
- all-to-all pairwise structure
- higher-order many-body alternatives

In this project, pairwise baselines matter because they let us compare connectivity effects
without also changing the interaction order at the same time.

## Hamming weight

The **Hamming weight** of a binary mask is the number of `1`s in it. For a generator it is the
number of qubits participating in that interaction; for a Z-word mask it is the number of qubits
whose parity is being checked.

<a id="mask"></a>
## Mask

A **mask** is a binary selector. It tells you which positions in a vector matter for the current
calculation and which ones are ignored.

In this repo, masks usually mean:
- a generator mask `g_j`, selecting which qubits belong to one interaction term
- a Z-word mask `a`, selecting which qubits contribute to a parity observable

You can read a mask as "include the positions where the entry is 1."

<a id="binary-mask"></a>
## Binary mask

A **binary mask** is just a mask written as a `0/1` vector. For example, with `n = 5`, the mask
`[1, 0, 1, 0, 0]` selects qubits 1 and 3 and ignores the rest.

Binary masks are useful because they let subset questions be written as vector arithmetic instead
of as set notation.

<a id="binary-selector"></a>
## Binary selector

A **binary selector** is a `0/1` object that marks which positions are included and which are
ignored.

In this project, masks are binary selectors:
- a generator row selects the qubits touched by one interaction term
- a Z-word mask selects the qubits included in one observable

You can read `1` as "include this position" and `0` as "leave it out."

<a id="mode"></a>
## Mode

A **mode** is one Fourier/Walsh component of the distribution, indexed by a subset of qubits `S`
or equivalently by a binary mask `a`.

In this project, each mode corresponds to one Z-word observable `Z_S`. So a mode is one specific
pattern of parity or correlation that the loss can check.

Examples:
- weight 1 mode: single-qubit structure
- weight 2 mode: pairwise structure
- high-weight mode: global many-qubit structure

<a id="z-word"></a>
## Z-word / Z-string / mask

A **Z-word** (also called a **Z-string** or **mask**) is a subset `S` of qubits, encoded as a
binary vector `a in {0,1}^n`, associated with the Pauli observable `Z_S`.

Its expectation value

`<Z_S> = E_x [(-1)^(S dot x)]`

is just the parity average over the selected qubits. The MMD derivations in this project work by
comparing these Z-word expectations between data and model.

<a id="z-word-weight"></a>
## Z-word weight

The **Z-word weight** is the Hamming weight of the Z-word mask. It is the number of qubits
included in that observable.

Examples:
- weight 1 Z-word: probes a single-qubit marginal
- weight 2 Z-word: probes a pairwise parity or correlation
- high-weight Z-word: probes a more global, many-qubit correlation

When the docs say the Gaussian kernel emphasizes low-weight Z-words, they mean the loss pays more
attention to simple, low-order correlations than to global ones.

<a id="pairwise-parity"></a>
## Pairwise parity

A **pairwise parity** is the parity of two selected bits.

For qubits `i` and `j`, it asks whether `x_i + x_j mod 2` is:
- `0`, meaning the two bits have even parity
- `1`, meaning the two bits have odd parity

Equivalently, pairwise parity tells you whether the two selected bits agree or differ. In the
Z-word language, this is a weight-2 observable.

<a id="parity"></a>
## Parity

**Parity** means whether the number of selected `1` bits is even or odd.

For a bitstring `x` and mask `a`, the quantity `(a dot x mod 2)` is:
- `0` if the selected bits contain an even number of ones
- `1` if the selected bits contain an odd number of ones

The associated sign `(-1)^(a dot x)` turns that into:
- `+1` for even parity
- `-1` for odd parity

That is why parity shows up everywhere in the Z-word formulas: Pauli Z observables on
computational-basis samples reduce exactly to parity checks.

Why parity is important in this project:
- the MMD loss is rewritten as a sum over Z-word expectation mismatches
- each Z-word expectation is a parity average over samples
- the IQP formulas compute those same observables on the model side
- circuit trainability depends on which parities a given family can actually influence

So parity is the common language connecting the dataset, the Fourier decomposition, the kernel,
and the IQP circuit observables.

## Spectral distribution `P_sigma`

For kernels like the Gaussian kernel, the MMD loss can be rewritten as a weighted sum over Z-word
modes. `P_sigma` is the corresponding probability distribution used to sample those modes in the
Monte Carlo estimator.

Intuition:
- low-weight masks correspond to simple, local correlations
- high-weight masks correspond to increasingly global correlations
- the kernel bandwidth controls how much mass `P_sigma` puts on each scale

<a id="spectral-weight"></a>
## Spectral weight

A **spectral weight** is the coefficient assigned to one Fourier or Z-word mode in the kernel
expansion. It tells you how much that mode contributes to the loss.

For the Gaussian-kernel picture in this repo, the contribution of a mode `S` is weighted by a term
proportional to `tau^|S|`, up to normalization. So the spectral weight answers the question:
"How strongly does the kernel care about disagreement on this particular mode?"

Large spectral weight means mismatch on that mode matters a lot. Small spectral weight means the
loss mostly ignores mismatch on that mode.

<a id="spectral-weights-decay"></a>
## Spectral weights decaying

When we say **spectral weights decay**, we mean they get smaller as the mode becomes more complex,
usually as the Z-word weight `|S|` increases.

For example, if the weight is proportional to `tau^|S|` with `0 < tau < 1`, then:
- weight 1 modes get multiplied by `tau`
- weight 2 modes get multiplied by `tau^2`
- weight 10 modes get multiplied by `tau^10`

So higher-weight modes are progressively suppressed. Intuitively, the kernel focuses on low-order
structure first and treats global many-body structure as less important.

This is what "the Gaussian spectrum decays with weight" means in practice.

<a id="interaction-pattern"></a>
## Interaction pattern

An **interaction pattern** is the subset of qubits coupled together by one generator.

Examples:
- a weight-1 generator has a single-site interaction pattern
- a weight-2 generator has a pairwise interaction pattern
- a higher-weight generator has a many-body interaction pattern

Changing the circuit family changes which interaction patterns are available. That in turn changes
which modes and parities the model can represent efficiently.

## Gaussian spectral normalization

The **Gaussian spectral normalization** is the exact constant-and-parameter convention used when
writing the Gaussian kernel in Walsh/Fourier form. This includes things like:
- the global prefactor `C`
- the decay parameter `tau`
- whether `tau` is written as `tanh(1/(2 sigma^2))`, `tanh(1/sigma^2)`, or an equivalent
  reparameterization

This term matters because two formulas can have the same qualitative decay in `|S|` while still
disagreeing on the exact estimator or on how `sigma` maps between theory and code.

## Bandwidth `sigma`

The **bandwidth** `sigma` is the kernel scale parameter. In the Gaussian case it determines how
quickly the spectral weights decay with Z-word weight.

In project language:
- smaller effective bandwidth means the loss pays more attention to finer or higher-order
  structure
- larger effective bandwidth means the loss emphasizes lower-order, smoother structure

Be careful comparing formulas across docs or code: some expressions use equivalent but rescaled
parameterizations of `sigma`.

## Barren plateau

A **barren plateau** is the regime where the gradient variance decays exponentially with system
size `n`, making optimization infeasible at scale. In this project the main question is whether
that decay depends on the combination of:
- circuit family
- kernel family and bandwidth
- initialization scheme

<a id="gradient-concentration"></a>
## Gradient concentration

**Gradient concentration** means gradients from different random parameter draws become tightly
clustered near zero, so the variance shrinks and optimization loses useful directional signal.

In practice, this means:
- most random initializations give almost the same tiny gradient
- changing parameters slightly does not produce a reliably distinguishable update direction
- the optimizer becomes dominated by estimator noise, shot noise, or finite-precision effects

In this project, gradient concentration is the observable symptom we measure when diagnosing a
barren plateau. If concentration strengthens rapidly with system size, training becomes less
scalable.

## Scaling law

A **scaling law** is the fitted dependence of a quantity, usually gradient variance, on system
size `n`. Here the main object is whether `Var(partial_theta L)` decays exponentially,
polynomially, or stays roughly constant as `n` grows.
