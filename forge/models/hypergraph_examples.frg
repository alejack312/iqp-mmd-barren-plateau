#lang forge
// ===========================================================================
// forge/models/hypergraph_examples.frg
// ===========================================================================
// `#lang forge` MUST be the very first line (same constraint as
// hypergraph.frg). Header comments therefore live below the pragma.
//
// Developer smoke tests for the sigs and predicates defined in hypergraph.frg.
// These are *existence* checks — each example asserts "there exists a
// hypergraph at this scope satisfying these constraints". They are not part
// of the 111-row evaluation pipeline; they just confirm the model compiles,
// typechecks, and that the structural predicates aren't trivially
// unsatisfiable at small scopes.
//
// Run manually:
//   racket forge/models/hypergraph_examples.frg
// Each `test expect { ... is sat }` should print "Test passed:"; if any one
// prints "Failed test", the corresponding predicate is over-constrained.

// Pull in every sig/function/predicate defined in the main model file.
// After this, every name declared in hypergraph.frg is available unqualified.
open "hypergraph.frg"

// `test expect { ... }` is Forge's unit-test harness. Each named block is a
// separate assertion: Forge tries to find an instance matching the body at
// the given scope; `is sat` passes iff at least one exists.
test expect {
  // --- Example 1: minimal_all_overlapping --------------------------------
  // "Can I build 4 generators over 6 qubits where every pair overlaps and
  // every generator still has Hamming weight ≤ 3?" If this is unsat the
  // predicate bounded_degree and overlapping disagree in a way that blocks
  // witnesses even at this small scope.
  minimal_all_overlapping: {
    all disj g1, g2: Generator | overlapping[g1, g2]
    bounded_degree[3]
  } for exactly 4 Generator, 6 Qubit, 0 Experiment is sat

  // --- Example 2: plateau_inducing_bounded -------------------------------
  // Matches the shape of the F2 template-search query: bounded_degree plus
  // high_overlap. Solver should find some 6-generator / 8-qubit hypergraph
  // that satisfies both. Confirms that F2's core conjunction is not
  // over-constrained at a practical scope.
  plateau_inducing_bounded: {
    bounded_degree[3]
    high_overlap[2]
  } for exactly 6 Generator, 8 Qubit, 0 Experiment is sat

  // --- Example 3: bounded_does_not_imply_disjoint ------------------------
  // Sanity check that bounded_degree and pairwise_disjoint are *independent*
  // properties. There should exist a bounded-degree-3 hypergraph that is
  // NOT pairwise disjoint (i.e. some pair overlaps). If this were unsat the
  // two predicates would be semantically linked, which would surprise us.
  bounded_does_not_imply_disjoint: {
    bounded_degree[3]
    not pairwise_disjoint
  } for 4 Generator, 6 Qubit, 0 Experiment is sat
}
