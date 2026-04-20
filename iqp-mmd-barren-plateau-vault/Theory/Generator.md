---
title: Generator
aliases:
  - Hyperedge
tags:
  - theory
  - glossary
---

# Generator / Hyperedge

A **generator** (also called a **hyperedge**) is one row of the [[Generator Matrix|generator matrix G]]. It identifies the subset of qubits touched by one IQP interaction term.

## Three Roles at Once

A single row of `G` does three things:

1. **Defines one interaction pattern** in the circuit — the gate $\exp(i \theta_j X^{g_j})$
2. **Identifies which parameter** $\theta_j$ controls that interaction
3. **Determines which Z-word modes** can "see" that interaction through the parity overlap $(a \cdot g_j \bmod 2)$

A generator is not just bookkeeping. It is the basic structural unit that says **where correlations can be created** and **which Fourier modes each parameter can influence**.

## Generator Weights

- **Weight 1** — single-qubit term; typical of the [[Product State Family|product state]] baseline
- **Weight 2** — pairwise $ZZ$-style interaction; the SMART four families are all weight-2 except product state
- **Weight $\ge 3$** — many-body interaction; used by the legacy `bounded_degree`, `dense`, `symmetric` families

## In the Code

A generator is a row of the `uint8` `G` array. Finding the support of a generator:

```python
support = np.flatnonzero(generator)  # indices of qubits it touches
weight = int(generator.sum())        # Hamming weight
```

These are used throughout [[IQP Model|`IQPModel`]] and [[Qiskit Circuit Builder]].

## Talking About Generators

When the project talks about:

- **"Local generators"** — small-support generators, typically weight $\le k$
- **"Dense generators"** — high-support generators, touching many qubits
- **"Generator overlap"** — how much two generators share qubits
- **"Generator family"** — the rule for sampling the full set of generators

All of those are statements about the rows of `G`.

## Related

- [[Generator Matrix]]
- [[Hypergraph Families]]
- [[Families MOC]]
- [[Parity Algebra]]
