---
title: Erdos-Renyi Family
aliases:
  - Sparse Erdős–Rényi Family
  - Random Graph Family
tags:
  - family
  - iqp
  - random
---

# Sparse Erdős–Rényi Family

The **random pairwise baseline**: one weight-2 generator per sampled graph edge. Calibrated to bounded expected degree so it stays meaningfully sparse as $n$ grows.

## G Matrix

Sample an undirected Erdős–Rényi graph $G_{ER}(n, p)$ with edge probability

$$
p = \min\!\left(1, \frac{c}{n}\right)
$$

then use one weight-2 generator row per sampled edge.

- `p_edge` in the config is interpreted as the target average degree **constant** $c$, not a literal edge probability
- Expected number of edges: $\binom{n}{2} \cdot p \approx \frac{c(n-1)}{2}$
- Expected degree per qubit: $(n-1) p \approx c$ (bounded as $n \to \infty$)
- Row count is intrinsic to the draw; **`m` is ignored**

## Why the `c / n` Calibration

> [!tip] Sparse means bounded degree, not "low p"
> Plain Erdős–Rényi with fixed `p_edge = 0.1` becomes denser and denser as $n$ grows: expected degree scales as $\Theta(n)$, and by $n = 100$ every generator covers ~10 qubits. The SMART regime wants a genuine **sparse baseline** where the graph stays qualitatively similar at $n = 16$ and $n = 1024$.

With $p = c/n$, the expected degree is the constant $c$ regardless of $n$. At $c = 2$ the graph is in the supercritical connected regime with a giant component; at $c < 1$ it's subcritical.

See [[Implementation Choices#Why Erdos-Renyi was changed this way]].

## Circuit Semantics

Each edge $(i,j)$ contributes $\exp(i \theta_{(i,j)} X_i X_j)$. Randomly placed ZZ interactions with bounded expected degree — a geometric interpolation between [[ZZ Lattice Family|lattice]] and [[Complete Graph Family|complete graph]].

## Expected Gradient Variance

- At $c \sim 2$: no BP expected — similar to bounded-degree
- At $c \sim \log n$: borderline — open question
- At $c \sim n/2$ (dense): BP expected, matches the complete graph limit

## In the Code

```python
def erdos_renyi(n, m, p_edge=0.1, rng=None):
    p = min(1.0, p_edge / n)
    sampled_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                sampled_edges.append((i, j))
    G = np.zeros((len(sampled_edges), n), dtype=np.uint8)
    for row, (i, j) in enumerate(sampled_edges):
        G[row, i] = 1; G[row, j] = 1
    return G
```

See [[Hypergraph Families#Sparse Erdős–Rényi Implementation]].

## Hypothesis Invariants

- Every row has weight exactly 2
- No duplicate rows (no double edges)
- Reproducible for a fixed RNG seed
- Over repeated draws, expected degree stays close to target constant $c$

## Runner Behavior

The scaling runner reads `circuit.erdos_renyi.p_edge` as a list (e.g. `[1.0, 2.0, 5.0]`) and sweeps over each value as a separate sub-setting, like `bandwidth` for Gaussian.

## Related

- [[Families MOC]]
- [[Hypergraph Families]]
- [[Implementation Choices#Why Erdos-Renyi was changed this way]]
- [[Scaling Runner]]
