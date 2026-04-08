---
title: Families MOC
tags:
  - moc
  - families
---

# IQP Circuit Families — Map of Content

The project studies how trainability depends on **connectivity family**. Each family is a rule for building the [[Generator Matrix|generator matrix G]] as $n$ grows.

## The SMART Four (Primary Sweep)

| Family | Note | Page |
|---|---|---|
| **Product state** | Single-qubit Z rotations; no entanglement; baseline | [[Product State Family]] |
| **2D ZZ lattice** | Open-boundary nearest-neighbor pairs on $L\times L$ grid | [[ZZ Lattice Family]] |
| **Sparse Erdős–Rényi** | Random pairwise graph with bounded expected degree | [[Erdos-Renyi Family]] |
| **Complete graph** | All-to-all pairwise interactions | [[Complete Graph Family]] |

All four are **pairwise baselines** — every generator has Hamming weight 1 or 2. The comparison is about connectivity, not interaction order. See [[Implementation Choices#Why Erdos-Renyi was changed this way]].

## Legacy Families (Kept for Reference)

Not in the primary sweep but still in the code for historical comparison:

- **`bounded_degree`** — generic k-local with degree cap
- **`dense`** — Bernoulli(0.5) on all entries of each generator
- **`community`** — block-structured
- **`symmetric`** — global bitflip-symmetric (even/odd parity rows)

## Expected Outcomes by Family

From [[Scope Lock]]:

| Family | BP expectation (uniform init) |
|---|---|
| Product state | No BP — too simple |
| 2D Lattice | No BP expected; gradient decays at most polynomially |
| ER sparse (`p ~ 2/n`) | No BP |
| ER dense (`p ~ 0.5`) | BP expected |
| Complete graph | BP expected for all kernels |
| Community | Partial BP; block-dependent |
| Symmetric | Unknown; symmetry may redistribute gradient mass |

## Hypothesis Conjecture Space

Connectivity family controls:

- **Commuting structure** — which generators commute sets which modes can mix
- **Generator overlap patterns** — how much the mod-2 parity vectors intersect
- **Degree statistics** — how many generators each qubit appears in
- **Spectral support** — which Z-word modes are "reachable" by any single parameter

The hypothesis is that locality (lattice, bounded-degree) suppresses plateau behavior while density (complete graph, dense) enhances it.

## Related

- [[Hypergraph Families]] — the code
- [[Generator Matrix]]
- [[Research Questions]] — Q2
- [[Scope Lock]] — section 4
