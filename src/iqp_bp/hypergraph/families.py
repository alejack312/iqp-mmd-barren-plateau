"""IQP hypergraph family generators.

Each function returns a generator matrix G of shape (m, n) with binary
entries. Row j is the bitmask g_j in {0,1}^n for the j-th IQP generator
exp(i theta_j X^{g_j}).

Glossary:
  - generator matrix: docs/technical/glossary.md#generator-matrix
  - generator / hyperedge: docs/technical/glossary.md#generator
  - interaction pattern: docs/technical/glossary.md#interaction-pattern
"""

from __future__ import annotations

import numpy as np


def bounded_degree(
    n: int,
    m: int,
    max_weight: int = 3,
    max_degree: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """k-local bounded-degree hypergraph.

    Each generator g_j has Hamming weight <= max_weight.
    Each qubit appears in at most max_degree generators.

    Returns:
        G: ndarray of shape (m, n), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()
    G = np.zeros((m, n), dtype=np.uint8)
    qubit_degree = np.zeros(n, dtype=int)
    for j in range(m):
        weight = rng.integers(1, max_weight + 1)
        available = np.where(qubit_degree < max_degree)[0]
        if len(available) < weight:
            weight = len(available)
        qubits = rng.choice(available, size=weight, replace=False)
        G[j, qubits] = 1
        qubit_degree[qubits] += 1
    return G


def erdos_renyi(
    n: int,
    m: int,
    p_edge: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sparse pairwise Erdős-Rényi family in the SMART regime.

    The config value ``p_edge`` is interpreted as the target average degree
    constant ``c``. A graph edge is sampled with probability ``p = min(1, c / n)``,
    which keeps the expected degree bounded as ``n`` grows.

    Each sampled graph edge becomes one weight-2 generator row. The returned
    matrix therefore has one row per sampled edge and ignores the requested ``m``.

    Returns:
        G: ndarray of shape (num_sampled_edges, n), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()

    p = min(1.0, p_edge / n)
    sampled_edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                sampled_edges.append((i, j))

    G = np.zeros((len(sampled_edges), n), dtype=np.uint8)
    for row, (i, j) in enumerate(sampled_edges):
        G[row, i] = 1
        G[row, j] = 1
    return G


def lattice(
    n: int,
    m: int,
    dimension: int = 1,
    range_: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Lattice-local hypergraph.

    1D: generators on consecutive qubit windows of size (range_+1).
    2D: exact nearest-neighbor ZZ interactions on a square grid.

    Returns:
        G: ndarray of shape (m, n), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()
    G = np.zeros((m, n), dtype=np.uint8)
    if dimension == 1:
        for j in range(m):
            start = rng.integers(0, n)
            for d in range(range_ + 1):
                G[j, (start + d) % n] = 1
    elif dimension == 2:
        # The SMART 2D lattice family is the fixed set of open-boundary
        # horizontal and vertical nearest-neighbor ZZ pairs on an L x L grid.
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
                    G[row, q] = 1
                    G[row, q + 1] = 1
                    row += 1
                if i + 1 < side:
                    G[row, q] = 1
                    G[row, q + side] = 1
                    row += 1
        return G
    else:
        raise ValueError(f"Unsupported lattice dimension {dimension!r}; choose 1 or 2")
    return G


def dense(
    n: int,
    m: int,
    expected_weight: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dense hypergraph with expected weight E[|g_j|] = expected_weight * n.

    Returns:
        G: ndarray of shape (m, n), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random((m, n)) < expected_weight).astype(np.uint8)


def product_state(
    n: int,
    **kwargs,
) -> np.ndarray:
    """Product state IQP: single-qubit Z rotations only.

    Generator matrix G = I_n: one weight-1 generator per qubit.
    No entanglement; baseline for connectivity comparisons.
    m is always n (one gate per qubit); any passed m is ignored.

    Returns:
        G: ndarray of shape (n, n), dtype uint8
    """
    return np.eye(n, dtype=np.uint8)


def complete_graph(
    n: int,
    **kwargs,
) -> np.ndarray:
    """Complete-graph IQP: all-to-all ZZ interactions.

    Generator matrix has C(n,2) rows, one for each qubit pair (i,j).
    m is always n*(n-1)//2; any passed m is ignored.

    Returns:
        G: ndarray of shape (n*(n-1)//2, n), dtype uint8
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    G = np.zeros((len(pairs), n), dtype=np.uint8)
    for k, (i, j) in enumerate(pairs):
        G[k, i] = 1
        G[k, j] = 1
    return G


def community(
    n: int,
    m: int,
    n_blocks: int = 4,
    p_intra: float = 0.8,
    p_inter: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Community-structured hypergraph.

    Qubits are divided into n_blocks blocks. Generators are mostly intra-block.

    Returns:
        G: ndarray of shape (m, n), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()
    block_size = n // n_blocks
    block_id = np.array(
        [i // block_size if i // block_size < n_blocks else n_blocks - 1 for i in range(n)]
    )
    G = np.zeros((m, n), dtype=np.uint8)
    for j in range(m):
        focus_block = rng.integers(0, n_blocks)
        for qi in range(n):
            p = p_intra if block_id[qi] == focus_block else p_inter
            if rng.random() < p:
                G[j, qi] = 1
        if G[j].sum() == 0:
            G[j, rng.integers(0, n)] = 1
    return G


def symmetric(
    n: int,
    m: int,
    parity: str = "even",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Global bitflip-symmetric hypergraph.

    Generators have even (or odd) Hamming weight to be symmetric under X^⊗n.

    Returns:
        G: ndarray of shape (m, n), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()
    G = np.zeros((m, n), dtype=np.uint8)
    target_parity = 0 if parity == "even" else 1
    for j in range(m):
        g = rng.integers(0, 2, size=n, dtype=np.uint8)
        if g.sum() % 2 != target_parity:
            flip = rng.integers(0, n)
            g[flip] ^= 1
        G[j] = g
    return G


FAMILIES: dict[str, callable] = {
    # Primary study families (supervisor scope, Mar 2026)
    "product_state": product_state,
    "lattice": lattice,  # use dimension=2, range_=1 for 2D lattice
    "erdos_renyi": erdos_renyi,
    "complete_graph": complete_graph,
    # Legacy / auxiliary families (kept for reference, not in primary sweep)
    "bounded_degree": bounded_degree,
    "dense": dense,
    "community": community,
    "symmetric": symmetric,
}


def make_hypergraph(
    family: str,
    n: int,
    m: int,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    """Dispatch to family generator by name."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}. Choose from {list(FAMILIES)}")
    # TODO: Weeks 3-4 (D4.1) enforce the primary four-family sweep and comparable
    # parameter-count policies centrally instead of distributing that logic in runners.
    return FAMILIES[family](n=n, m=m, rng=rng, **kwargs)
