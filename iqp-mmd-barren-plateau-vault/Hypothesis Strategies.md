---
title: Hypothesis Strategies
tags:
  - code
  - iqp_bp
  - testing
---

# `iqp_bp.hypergraph.hypothesis_strategies`

Property-based test strategies for the SMART family layer. Used by the test suite and the internal validation harness.

**File:** [`src/iqp_bp/hypergraph/hypothesis_strategies.py`](../src/iqp_bp/hypergraph/hypothesis_strategies.py)

## Purpose

[Hypothesis](https://hypothesis.readthedocs.io/) is a property-based testing library: instead of writing a single example and checking its output, you define a **strategy** that generates many examples, and assert invariants that should hold on all of them.

For this project, Hypothesis is used to verify that the SMART family layer actually produces what the SMART scope requires — not just that it returns *some* matrix.

## Key Strategies

- `smart_family_config()` — randomly sample a family name + parameters from the SMART primary four
- `smart_family_instance()` — build an actual `G` from a sampled config
- `mmd_instance()` — build a full (G, theta, data) tuple for end-to-end invariant checks

These were narrowed from an earlier, more generic layer that sampled arbitrary binary matrices. See [[Implementation Choices#Why the Hypothesis layer was narrowed to SMART families]].

## Invariants Tested

From the test suite (see [[Tests]]):

### Common

- `G` is binary
- `G` has the expected shape
- No row is all zero

### Product state

- $G$ is the identity
- Every row has weight 1
- `m = n`

### Lattice (2D)

- Every row has weight 2
- Row count is exactly $2L(L-1)$
- Every row is a horizontal or vertical nearest-neighbor edge
- No duplicate rows
- Deterministic across RNG seeds

### Erdős–Rényi (sparse)

- Every row has weight 2
- No duplicate rows
- Reproducible for a fixed seed
- Expected degree over repeated draws close to target $c$

### Complete graph

- Every pair of qubits appears exactly once
- Every row has weight 2
- Row count is exactly $n(n-1)/2$

## Philosophy

From [[Implementation Choices]]:

> "The test suite focuses on invariants, not frozen numeric outputs. That style of testing is deliberate. The implementation still moves, but the mathematics should not."

## Related

- [[Hypergraph Families]]
- [[Tests]]
- [[Implementation Choices]]
