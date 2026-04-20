#lang forge
// ===========================================================================
// forge/models/hypergraph.frg
// ===========================================================================
// `#lang forge` MUST be the very first line of this file — Racket's default
// load handler only switches to the Forge reader when `#lang forge` is the
// literal first token on line 1. Header comments therefore live *below*
// the pragma, not above it.
//
// This file is written in Forge (https://forge-fm.org), a relational-logic
// specification language in the Alloy family, running on top of Racket. Think
// of it as "SQL meets propositional logic": you declare sigs (roughly: sets
// of atoms / rows in a table) and relations between them, then state
// predicates that a solver either satisfies (finds a witness) or falsifies.
//
// In this project Forge plays the role of a machine-checkable "theory of
// barren plateaus in IQP circuits". The Python side:
//   1. trains / evaluates concrete circuits,
//   2. records whether anti-concentration (AC) held or not,
//   3. exports each circuit's hypergraph structure + its Experiment metadata
//      as a Forge `inst` block (see src/iqp_bp/forge/export_instances.py),
//   4. asks Forge: "does my structural predicate agree with the observed
//      plateau label on this specific row?" (iff-checked in the template at
//      src/iqp_bp/forge/query_templates.py).
// If the predicate agrees on every labeled row, Forge returns SAT; otherwise
// the solver finds the mismatch and reports UNSAT, which is exactly the
// signal we want: the theory lied about that row.

// Disable the Sterling visualizer. Without this, every SAT result would open
// a browser window, which is fine for interactive work but blocks automated
// batch runs like our 111-row sweep.
option run_sterling off

// forge/models/hypergraph.frg
// Forge model for IQP hypergraph structural analysis
// Requires: Forge (https://forge-fm.org)
//
// Purpose: finite model finding for plateau-inducing overlap patterns
// and structural invariants of IQP generator sets.

// ---------------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------------
// Sigs are the "things that exist" in the model. Each sig becomes a set of
// atoms; Forge picks how many atoms to allocate per sig based on the scope
// declared on the query (e.g. `for 4 Qubit, 6 Generator`). Fields on a sig
// are typed relations: `contains: set Qubit` says every Generator has a
// subset of Qubits as its "contains" field.

// One atom per qubit in the IQP register. Q0, Q1, ..., Q(n-1) get bound
// explicitly in the Python-emitted `inst` block, one Qubit atom per wire.
sig Qubit {}

// Each Generator represents one of the m diagonal Z-word phases in the IQP
// layer exp(i·theta_j · Z_{g_j}). Its `contains` field is the support
// (which qubits it acts on), i.e. the row g_j of the binary generator matrix
// G used all over the Python code.
sig Generator {
  // The support of this generator: a subset of Qubit, matching the 1s in
  // G[j, :] for generator j.
  contains: set Qubit,
  // The overlap graph edge set. Redundant with `contains` (you can recompute
  // it by intersecting supports), but F1 export writes it explicitly so
  // Forge does not have to synthesize an `overlaps` field during template
  // search — the synthesized one would be arbitrary, breaking reproducibility.
  overlaps: set Generator
}

// A single `Params` atom carries the integer thresholds used by the
// structural predicates (max allowed Hamming weight, max allowed pairwise
// overlap, max per-qubit degree). `one sig` means "exactly one atom";
// the scope cannot allocate zero or more than one Params.
one sig Params {
  max_weight: one Int,        // single integer, the k in bounded_degree[k]
  overlap_threshold: one Int, // single integer, the t in high_overlap[t]
  max_qubit_degree: one Int   // max times any single qubit appears across Gs
}

// ---------------------------------------------------------------------------
// Loss / Kernel / Init / Outcome — the Experiment axes
// ---------------------------------------------------------------------------
// An Experiment atom is one training configuration (loss, kernel, init) plus
// the observed plateau label. The Python side binds exactly one Experiment
// (`Exp0`) per row when we run a plateau-agreement query. The `abstract`
// sigs below are Forge's way of declaring a closed enum: a Loss must be one
// of its concrete sub-sigs, nothing else.

// Loss family. Only MMD^2 is implemented in Python; the kernel choice
// parameterizes it. Keep a single-atom Loss sig so Experiment has a bindable
// field even though loss is not a sweep axis.
abstract sig Loss {}
// `MMD` is the only concrete Loss — MMD^2 with a positive-definite kernel.
one sig MMD extends Loss {}

// Kernel families. Must match the keys of KERNEL_SAMPLERS in
// src/iqp_bp/mmd/kernel.py (gaussian, laplacian, multi_scale_gaussian,
// polynomial, linear).
abstract sig Kernel {}
// Five concrete kernels; Python emits exactly one of them per Experiment inst.
one sig Gaussian, Laplacian, MultiScaleGaussian, Polynomial, Linear extends Kernel {}

// Init schemes. Must match the cases in run_training.py::_sample_initial_theta
// (uniform, small_angle, data_dependent).
abstract sig Init {}
// UniformInit      — theta ~ U[-pi, pi]
// SmallAngleInit   — theta ~ N(0, sigma^2) with sigma small (0.01 in F3)
// DataDependentInit — theta warm-started from data expectations
one sig UniformInit, SmallAngleInit, DataDependentInit extends Init {}

// Binary plateau label: the ground-truth outcome of the anti-concentration
// check for a row. Every labeled row is either "plateau observed" or
// "plateau absent"; the predicates below aim to predict which.
abstract sig Outcome {}
one sig PlateauObserved, PlateauAbsent extends Outcome {}

// An Experiment bundles the four Experiment-axis fields plus the empirical
// plateau label. `lone Outcome` = "at most one" — rows without a label will
// simply leave the field unbound (not used in F3/F4 but reserved for later).
sig Experiment {
  loss: one Loss,
  kernel: one Kernel,
  init: one Init,
  plateau_observed: lone Outcome
}

// ---------------------------------------------------------------------------
// Derived relations (functions)
// ---------------------------------------------------------------------------
// `fun` declares a function; these are pure expressions Forge rewrites at
// check time. They give us a readable surface for writing predicates.

// Overlap between two generators: number of shared qubits.
// `&` is set intersection; `#` is cardinality. So this is |supp(g1) ∩ supp(g2)|.
fun overlap[g1, g2: Generator]: Int {
  #(g1.contains & g2.contains)
}

// Hamming weight of a generator g: the number of qubits it acts on.
// Equivalent to |supp(g)|, i.e. the number of 1s in row g of G.
fun weight[g: Generator]: Int {
  #g.contains
}

// Boolean "these two generators share at least one qubit in their support".
// `some <set>` means "the set is non-empty"; so this returns True iff the
// intersection of supports is non-empty, i.e. the pair is overlapping.
pred overlapping[g1, g2: Generator] {
  some g1.contains & g2.contains
}

// ---------------------------------------------------------------------------
// Structural predicates
// ---------------------------------------------------------------------------

// The `overlaps` field is redundant with `contains` (derivable via intersect),
// but F1 exports it explicitly so it carries the overlap graph on exported
// instances. When F2 does template search (structure unpinned), Forge would
// otherwise bind `overlaps` arbitrarily. Include this pred as a conjunct in
// any query template that wants synthesized witnesses to have sensible
// `overlaps` bindings. Self-loops are excluded by convention.
// The `iff` below reads: "g1 and g2 are marked as overlapping iff their
// supports actually intersect". Adding this as a conjunct of every query
// keeps the `overlaps` field honest.
pred overlaps_consistent {
  // No generator overlaps itself in the explicit edge relation.
  no g: Generator | g -> g in overlaps
  // For every distinct pair, the explicit edge agrees with the computed
  // support intersection. `disj` means the quantifier's variables are
  // required to be different atoms.
  all disj g1, g2: Generator |
    g1 -> g2 in overlaps iff some (g1.contains & g2.contains)
}

// All generators have Hamming weight ≤ k (the "k-local" condition used in
// most barren-plateau existence proofs).
pred bounded_degree[k: Int] {
  // Universal quantifier: true iff every single generator's weight is ≤ k.
  all g: Generator | weight[g] <= k
}

// No two distinct generators share any qubit — i.e. a commuting family with
// pairwise-disjoint supports. Not used in F3/F4 but kept as a reference shape.
pred pairwise_disjoint {
  all disj g1, g2: Generator | no g1.contains & g2.contains
}

// "Dense" family: every generator acts on at least half the qubits. Used
// historically in template-search runs; `multiply` and `#Qubit` are Forge
// primitives for integer arithmetic and sig-cardinality respectively.
pred dense_family {
  all g: Generator | multiply[2, weight[g]] >= #Qubit
}

// High-overlap condition: some distinct pair shares > `threshold` qubits.
// `some disj g1, g2` = exists at least one pair of distinct generators
// whose overlap count strictly exceeds the threshold.
pred high_overlap[threshold: Int] {
  some disj g1, g2: Generator | overlap[g1, g2] > threshold
}

// Structural predictor of plateau manifestation. Same shape as F2's
// plateau_inducing_bounded template: small Hamming weight plus at least
// one high-overlap pair. This is the F2/F3 predicate; F3 showed it misses
// every observed plateau (0/88 recall), which is what motivates F4 below.
pred plateau_structurally_predicted[k, t: Int] {
  bounded_degree[k]
  high_overlap[t]
}

// Community structure: a partition of qubits into two blocks where every
// generator is mostly intra-block.
//
// COMMENTED OUT: requires higher-order quantification (`some block: set
// Qubit`), which the default `#lang forge` reader rejects. Preserving the
// semantically-correct form here as reference for future work — if we
// later enable Forge's higher-order mode or port to `#lang forge/core`,
// uncomment and re-test. The first-order demotion (`some block: Qubit`)
// parses but means `block` is a single qubit, not a subset, which does
// not express community structure.
//
// pred community_structure[ratio: Int] {
//   some block: set Qubit | {
//     some block
//     some Qubit - block
//     all g: Generator | {
//       let intra = #(g.contains & block) |
//       let total = weight[g] |
//       multiply[ratio, intra] >= total
//     }
//   }
// }

// ---------------------------------------------------------------------------
// F4 predicates: conditioning on (Init, n, structure) jointly
// ---------------------------------------------------------------------------
// F3 used a purely-structural predicate and got 0/88 recall on observed
// plateaus. The F4 hypothesis is that the right predictor has to look at
// the init scheme and qubit count too, with structure providing an "escape
// hatch" for configurations known to avoid plateaus.

// Structural match for the Python `complete_graph` family
// (src/iqp_bp/hypergraph/families.py): every distinct qubit pair is covered
// by a weight-2 generator, with exactly n(n-1)/2 generators. Used by
// plateau_manifests as an escape hatch for uniform-init rows: in F3 the
// only uniform-init sub-sweep that passed anti-concentration at every n
// was the literal complete_graph family.
pred complete_graph_like {
  // Clause 1: every generator has weight exactly 2 (single two-qubit edge).
  all g: Generator | weight[g] = 2
  // Clause 2: for every distinct pair of qubits, some generator's support is
  // the set that contains both. Equivalent to "the generator-qubit
  // containment relation covers every pair".
  all disj q1, q2: Qubit | some g: Generator | q1 in g.contains and q2 in g.contains
  // Clause 3: counting constraint. Exactly n(n-1)/2 generators, so the
  // family has no redundant edges. `multiply` / `subtract` are Forge's
  // integer-arithmetic primitives; equality on Int is literal.
  multiply[2, #Generator] = multiply[#Qubit, subtract[#Qubit, 1]]
}

// F4 joint-conditioning plateau predicate.
//   plateau_manifests[e]  iff  e.init = SmallAngleInit
//                       or   (e.init = UniformInit and #Qubit >= 6 and not complete_graph_like)
// Encodes the F3-closeout hypothesis that a predictive plateau predicate has
// to branch on (Init, n, structure) jointly — the F2/F3 structure-only
// predicate was blind to init and missed 88/88 observed plateaus.
pred plateau_manifests[e: Experiment] {
  // Branch A: small-angle init. F3 found every small_angle row (56/56 at
  // sigma=0.01) hit observed plateau. Short-circuits regardless of structure.
  e.init = SmallAngleInit or
  // Branch B: uniform init AND enough qubits AND structure is not the
  // complete-graph escape hatch. #Qubit is the sig-cardinality of Qubit,
  // so >= 6 picks up n=6 and n=8 rows in our sweep.
  (e.init = UniformInit and #Qubit >= 6 and not complete_graph_like)
}

// Ablation variant of plateau_manifests with the complete-graph escape hatch
// removed. Running this side-by-side with plateau_manifests on the same 111
// rows isolates the structure clause's contribution: the accuracy delta is
// exactly the number of rows that the `not complete_graph_like` exception
// changed the prediction for. If the delta is near zero, the F4 predicate is
// really just an (init, n) classifier and the structure axis can be dropped
// in F4.1; if the delta is meaningful, the escape hatch is pulling weight.
pred plateau_manifests_no_escape[e: Experiment] {
  // Identical to plateau_manifests except the `and not complete_graph_like`
  // conjunct has been dropped from Branch B.
  e.init = SmallAngleInit or
  (e.init = UniformInit and #Qubit >= 6)
}
