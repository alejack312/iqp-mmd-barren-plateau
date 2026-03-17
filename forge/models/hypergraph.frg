// forge/models/hypergraph.frg
// Forge model for IQP hypergraph structural analysis
// Requires: Forge (https://forge-fm.org)
//
// Purpose: finite model finding for plateau-inducing overlap patterns
// and structural invariants of IQP generator sets.

#lang forge

// ---------------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------------

sig Qubit {}

sig Generator {
  contains: set Qubit       // support of this generator
}

// ---------------------------------------------------------------------------
// Derived relations
// ---------------------------------------------------------------------------

// Overlap between two generators: number of shared qubits
fun overlap[g1, g2: Generator]: Int {
  #(g1.contains & g2.contains)
}

// Hamming weight of a generator
fun weight[g: Generator]: Int {
  #g.contains
}

// Overlap graph: two generators are connected if their support intersects
pred overlapping[g1, g2: Generator] {
  some g1.contains & g2.contains
}

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

// All generators have bounded weight (k-local)
pred bounded_degree[k: Int] {
  all g: Generator | weight[g] <= k
}

// Every pair of generators has disjoint support (commuting family)
pred pairwise_disjoint {
  all disj g1, g2: Generator | no g1.contains & g2.contains
}

// Dense family: every generator acts on at least half the qubits
pred dense_family {
  all g: Generator | mul[2, weight[g]] >= #Qubit
}

// High-overlap condition: some pair shares >threshold qubits
pred high_overlap[threshold: Int] {
  some disj g1, g2: Generator | overlap[g1, g2] > threshold
}

// Community structure: exists a partition of qubits into two blocks
// where all generators are mostly intra-block
pred community_structure[ratio: Int] {
  some block: set Qubit | {
    some block
    some Qubit - block
    all g: Generator | {
      let intra = #(g.contains & block) |
      let total = weight[g] |
      mul[ratio, intra] >= total
    }
  }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

// Find minimal hypergraph (by qubit count) where all generators overlap
example minimal_all_overlapping is {
  all disj g1, g2: Generator | overlapping[g1, g2]
  bounded_degree[3]
} for exactly 4 Generator, 6..8 Qubit

// Find instance where high overlap exists despite bounded degree
example plateau_inducing_bounded is {
  bounded_degree[3]
  high_overlap[2]
} for exactly 6 Generator, 8 Qubit

// Check: does bounded-degree imply pairwise disjoint? (expected: NO)
check bounded_does_not_imply_disjoint {
  bounded_degree[3] => pairwise_disjoint
} for 4 Generator, 6 Qubit expect 0

// Find community structure in dense family
example dense_with_community is {
  dense_family
  community_structure[2]
} for exactly 4 Generator, 8 Qubit
