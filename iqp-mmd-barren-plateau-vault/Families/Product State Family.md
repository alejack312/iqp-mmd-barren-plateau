---
title: Product State Family
tags:
  - family
  - iqp
---

# Product State Family

The **no-entanglement baseline**. Every generator is a single-qubit Z rotation.

## G Matrix

$$
G = I_n \quad\text{(the identity)}
$$

- Shape: `(n, n)`
- Every row has Hamming weight 1
- $m = n$ (caller-provided `m` is ignored)

## Circuit Semantics

Each parameter $\theta_i$ controls exactly one qubit's phase:

$$
|\psi(\theta)\rangle = \prod_{i=1}^n e^{i \theta_i X_i} \, |+\rangle^{\otimes n}
$$

No entanglement at all. The output distribution factorizes.

## Why It Matters

- **Simplest possible case.** If this family shows barren plateau behavior, something is wrong with the whole measurement pipeline.
- **Disentangles circuit structure from kernel/data.** Any BP signal on `product_state` comes purely from the loss side, not from entanglement.
- **Sanity anchor** for all families that build on top of it.

## Expected Gradient Variance

Should remain **constant or weakly decreasing** in $n$. Any exponential decay here would be an alarm.

## In the Code

```python
def product_state(n: int, **kwargs) -> np.ndarray:
    return np.eye(n, dtype=np.uint8)
```

See [[Hypergraph Families]].

## Hypothesis Invariants

From [`hypothesis_strategies.py`](../src/iqp_bp/hypergraph/hypothesis_strategies.py) and `tests/test_hypergraph_families.py`:

- `G.shape == (n, n)`
- `G` is the identity
- Every row has weight exactly 1
- No row is all zero
- Deterministic (no RNG state consumed)

## Related

- [[Families MOC]]
- [[Hypergraph Families]]
- [[Generator Matrix]]
