---
title: Complete Graph Family
tags:
  - family
  - iqp
  - dense
---

# Complete Graph Family

The **densest pairwise baseline**: every pair of qubits gets one weight-2 generator.

## G Matrix

One row per pair $(i,j)$ with $i < j$:

- Row count: $m = \binom{n}{2} = \frac{n(n-1)}{2}$
- Every row has Hamming weight 2
- Deterministic — no RNG consumed
- Caller-provided `m` is ignored

## Circuit Semantics

Every pair of qubits is coupled:

$$
|\psi(\theta)\rangle = \exp\!\left(i \sum_{i<j} \theta_{(i,j)} X_i X_j\right) |+\rangle^{\otimes n}
$$

The number of parameters grows quadratically in $n$, so experimental cost is higher than for the sparse families. The scaling runner still handles it fine up to $n \sim 100$ because the actual work is Monte Carlo over $z$ and $a$, not enumeration.

## Why It Matters

- **All-to-all extreme.** The expected regime for barren plateaus — every parameter is in the "loudest" connectivity pattern.
- **Comparison point.** The gap between lattice and complete graph is the natural axis for "does locality matter?"
- **Sampling hardness.** The densest IQP regime is also the regime most closely associated with Bremner–Jozsa–Shepherd sampling hardness results.

## Expected Gradient Variance

**BP expected for all kernels.** The question is not whether it exhibits exponential decay — it's whether the decay rate $\alpha$ depends on kernel choice, and whether any initialization scheme dampens it.

## In the Code

```python
def complete_graph(n, **kwargs):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    G = np.zeros((len(pairs), n), dtype=np.uint8)
    for k, (i, j) in enumerate(pairs):
        G[k, i] = 1; G[k, j] = 1
    return G
```

## Hypothesis Invariants

- Every pair of qubits appears exactly once
- Every row has weight exactly 2
- Row count is exactly $n(n-1)/2$

From the test suite: "If it does not contain every pair exactly once, it is not the family its name claims."

## Related

- [[Families MOC]]
- [[Hypergraph Families]]
- [[IQP Classical Sampling]]
