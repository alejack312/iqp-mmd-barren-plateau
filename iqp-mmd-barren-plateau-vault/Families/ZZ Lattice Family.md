---
title: ZZ Lattice Family
aliases:
  - 2D Lattice Family
tags:
  - family
  - iqp
  - lattice
---

# 2D ZZ Lattice Family

The **local-interaction baseline**: nearest-neighbor pairwise generators on an $L \times L$ square grid.

## G Matrix

For $n = L^2$ qubits arranged on an $L \times L$ open-boundary grid:

- Row count: $m = 2L(L-1)$ (all horizontal + all vertical neighbor pairs)
- Every row has Hamming weight 2
- Deterministic — no RNG consumed
- Rejects non-square $n$ and any `range_ != 1`

## Circuit Semantics

Each pair $(i, j)$ of nearest-neighbor qubits contributes $\exp(i \theta_{ij} X_i X_j)$. In the Z-basis, after the Hadamard sandwich, this is a nearest-neighbor $ZZ$ coupling. Hence the name.

## Why It Matters

- **Physical locality** — the closest thing in the sweep to a 2D quantum Ising model.
- **Commuting structure** — all $X_i X_j$ gates commute because they only share ±1 qubits, and the gates are already in a single Pauli family.
- **Polynomial gradient variance hypothesis** — local interactions are the canonical escape hatch from barren plateaus (Cerezo et al. 2021).

## Expected Gradient Variance

Should scale **at most polynomially** in $n$. If it doesn't, the project's "local structure avoids BP" hypothesis for MMD loss fails on the easiest test.

## In the Code

From `lattice(n, m, dimension=2, range_=1, rng)`:

```python
side = int(np.sqrt(n))
if side * side != n:
    raise ValueError(f"n must be a perfect square for 2D lattice, got {n}")
if range_ != 1:
    raise ValueError(f"range_ must be 1 for 2D lattice, got {range_}")
n_edges = 2 * side * (side - 1)
G = np.zeros((n_edges, n), dtype=np.uint8)
row = 0
for i in range(side):
    for j in range(side):
        q = i * side + j
        if j + 1 < side:
            G[row, q] = 1; G[row, q + 1] = 1
            row += 1
        if i + 1 < side:
            G[row, q] = 1; G[row, q + side] = 1
            row += 1
```

See [[Hypergraph Families#2D Lattice Implementation]].

## Why Square-Only

> [!warning] No rectangular fallbacks
> The SMART scope requires the exact 2D nearest-neighbor family. There is no honest way to represent $n = 24$ or $n = 48$ as a square grid without changing the geometry or sneaking in a different family. The code **fails fast** instead of pretending.

This is why `n_qubits` in `base.yaml` is a list of perfect squares: `[4, 9, 16, 25, 36, 49, ...]`.

## Hypothesis Invariants

- Every row has weight exactly 2
- Row count is exactly $2L(L-1)$
- Every row is a horizontal or vertical nearest-neighbor edge
- No duplicate rows
- Deterministic across RNG seeds

See `tests/test_hypergraph_families.py` for the exact assertions.

## Related

- [[Families MOC]]
- [[Hypergraph Families]]
- [[Implementation Choices#Why the lattice is square-only]]
- [[Generator Matrix]]
