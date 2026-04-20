---
title: Generator Matrix
aliases:
  - G matrix
  - "G"
tags:
  - theory
  - core
---

# Generator Matrix `G`

The **generator matrix** `G` is the binary matrix that specifies which qubits each IQP generator acts on. Almost every formula in this project reduces to cheap binary arithmetic on `G`.

## Shape and Semantics

- Shape: `(m, n)`
- `n`: number of qubits
- `m`: number of generators (parameterized interaction terms)
- Entries are `0` or `1`

Row $j$ is the bitmask $g_j$ for the generator $\exp(i \theta_j X^{g_j})$:

- $G_{j,i} = 1$ → qubit $i$ participates in generator $j$
- $G_{j,i} = 0$ → qubit $i$ is not touched

Each row is also called a [[Generator|generator or hyperedge]].

## The Two Operations That Matter

Almost every calculation uses one of these two mod-2 matrix-vector products:

```python
(G @ a) % 2   # shape (m,): which generators have odd overlap with Z-word a
(G @ z) % 2   # shape (m,): which generators have odd overlap with random bitstring z
```

These feed directly into the phase formula of [[IQP Expectation]]:

$$
\Phi(\theta, z, a) = 2 \sum_j \theta_j \cdot \underbrace{(a \cdot g_j \bmod 2)}_{(G a)_j \bmod 2} \cdot (-1)^{z \cdot g_j}
$$

## Families Build G Differently

Each [[Families MOC|connectivity family]] is nothing but a rule for constructing `G` as $n$ grows:

| Family | Row pattern | `m` policy |
|---|---|---|
| [[Product State Family]] | identity — one 1 per row | `m = n` |
| [[ZZ Lattice Family]] | open-boundary nearest-neighbor pairs on $L\times L$ grid | `m = 2L(L-1)`, intrinsic |
| [[Erdos-Renyi Family]] | one pairwise row per sampled edge | intrinsic to sample |
| [[Complete Graph Family]] | all $\binom{n}{2}$ pairs | `m = n(n-1)/2` |

Callers **cannot trust** the requested `m` for the intrinsic families. Downstream code must use `G.shape[0]` — see [[Implementation Choices]].

## Code Paths That Operate on G

- [[Hypergraph Families|`iqp_bp.hypergraph.families`]] — builds `G` for each family
- [[IQP Expectation|`iqp_bp.iqp.expectation`]] — consumes `G` in the phase formula
- [[IQP Model|`iqp_bp.iqp.model.IQPModel`]] — wraps `G` + `θ`
- [[Qiskit Circuit Builder|`iqp_bp.qiskit.circuit_builder.build_iqp_circuit`]] — translates `G` to a Qiskit circuit
- [[Anti-Concentration]] — uses `G` to compute the exact probability vector via FWHT

## Why Binary

Because overlap questions — "does generator $j$ see qubit $i$?", "does Z-word $a$ have odd parity with generator $j$?" — are exactly the `0/1` arithmetic that drives IQP observables. The entire phase machinery lives in mod-2 land, so `G` being a [[Binary Matrix|binary matrix]] is not incidental.

## Related

- [[Hypergraph Families]]
- [[IQP Circuits]]
- [[Generator]] (single row = hyperedge)
- [[Parity Algebra]]
- [[Glossary]]
