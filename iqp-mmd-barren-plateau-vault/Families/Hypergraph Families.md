---
title: Hypergraph Families
tags:
  - code
  - families
  - iqp_bp
---

# `iqp_bp.hypergraph.families`

The module that builds the [[Generator Matrix|generator matrix G]] for each circuit family. The per-family pages are under [[Families MOC]].

**File:** [`src/iqp_bp/hypergraph/families.py`](../src/iqp_bp/hypergraph/families.py)

## Public API

| Function | Returns `G` of shape | Notes |
|---|---|---|
| `product_state(n, **kwargs)` | `(n, n)` identity | ignores `m` |
| `lattice(n, m, dimension, range_, rng)` | `(2L(L-1), n)` for 2D | 2D: perfect-square `n`, `range_=1`; 1D: `(m, n)` |
| `erdos_renyi(n, m, p_edge, rng)` | `(|E|, n)` | intrinsic row count |
| `complete_graph(n, **kwargs)` | `(n(n-1)/2, n)` | ignores `m` |
| `bounded_degree(n, m, max_weight, max_degree, rng)` | `(m, n)` | legacy |
| `dense(n, m, expected_weight, rng)` | `(m, n)` | legacy |
| `community(n, m, n_blocks, p_intra, p_inter, rng)` | `(m, n)` | legacy |
| `symmetric(n, m, parity, rng)` | `(m, n)` | legacy |
| `make_hypergraph(family, n, m, rng, **kwargs)` | dispatcher | |

## The SMART Four

Four primary families for the locked study (see [[Scope Lock]]):

1. **`product_state`** — see [[Product State Family]]
2. **`lattice`** (2D, range=1) — see [[ZZ Lattice Family]]
3. **`erdos_renyi`** (sparse pairwise) — see [[Erdos-Renyi Family]]
4. **`complete_graph`** — see [[Complete Graph Family]]

The legacy families (`bounded_degree`, `dense`, `community`, `symmetric`) are kept in the code but not used in the primary sweep.

## Intrinsic vs Requested `m`

> [!warning] Trust `G.shape[0]`
> Several primary families determine their own row count. Callers should never assume the requested `m` survived — always read `actual_m = G.shape[0]` after calling `make_hypergraph`.

- `product_state`: `m = n` (ignores input)
- `lattice` (2D): `m = 2L(L-1)` (ignores input, validates perfect-square `n`, `range_=1`)
- `erdos_renyi`: `m = |sampled edges|` (varies by draw)
- `complete_graph`: `m = n(n-1)/2` (ignores input)

See [[Implementation Choices#Why the runners now trust G.shape[0]]].

## 2D Lattice Implementation

Deterministic open-boundary nearest-neighbor grid:

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

Every row has Hamming weight 2. Every edge is horizontal or vertical nearest-neighbor. No duplicates. Deterministic across RNG seeds. See [[Tests#Lattice invariants]].

## Sparse Erdős–Rényi Implementation

```python
p = min(1.0, p_edge / n)  # p_edge = target average degree c
sampled_edges = []
for i in range(n):
    for j in range(i + 1, n):
        if rng.random() < p:
            sampled_edges.append((i, j))
G = np.zeros((len(sampled_edges), n), dtype=np.uint8)
for row, (i, j) in enumerate(sampled_edges):
    G[row, i] = 1; G[row, j] = 1
```

- `p_edge` is **interpreted as the target average degree constant `c`**, not a literal edge probability
- Every row has weight 2
- Row count varies per draw
- Bounded expected degree as $n$ grows — this is what "sparse" means in the SMART regime

## Dispatcher

```python
FAMILIES = {
    "product_state": product_state,
    "lattice": lattice,
    "erdos_renyi": erdos_renyi,
    "complete_graph": complete_graph,
    # legacy
    "bounded_degree": bounded_degree,
    "dense": dense,
    "community": community,
    "symmetric": symmetric,
}

def make_hypergraph(family, n, m, rng=None, **kwargs):
    return FAMILIES[family](n=n, m=m, rng=rng, **kwargs)
```

## Open Work

- `D4.1` — enforce the primary four-family sweep + comparable parameter-count policies centrally in the runners, instead of distributing the logic between `families.py` and the runners.

## Related

- [[Generator Matrix]]
- [[Families MOC]]
- [[Implementation Choices]]
- [[Hypothesis Strategies]]
- [[Tests]]
