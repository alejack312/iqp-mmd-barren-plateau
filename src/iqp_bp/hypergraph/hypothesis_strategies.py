"""Hypothesis strategies for property-based testing of hypergraph families.

Provides Hypothesis `st.composite` strategies that generate valid hypergraphs
for structured fuzz testing and property-based experiment sweeps.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def hypergraph_matrix(
    draw,
    n_min: int = 4,
    n_max: int = 16,
    m_min: int = 4,
    m_max: int = 32,
) -> tuple[np.ndarray, int, int]:
    """Draw a random binary hypergraph matrix G of shape (m, n).

    Returns:
        (G, n, m) where G is uint8 ndarray with no all-zero rows.
    """
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    m = draw(st.integers(min_value=m_min, max_value=m_max))
    G = draw(
        arrays(
            dtype=np.uint8,
            shape=(m, n),
            elements=st.integers(min_value=0, max_value=1),
        )
    )
    # Ensure no all-zero generators
    empty_rows = G.sum(axis=1) == 0
    G[empty_rows, 0] = 1
    return G, n, m


@st.composite
def bounded_degree_hypergraph(
    draw,
    n_min: int = 4,
    n_max: int = 16,
    max_weight: int = 3,
) -> tuple[np.ndarray, int]:
    """Draw a k-local bounded-degree hypergraph."""
    from iqp_bp.hypergraph.families import bounded_degree
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    m = draw(st.integers(min_value=n, max_value=2 * n))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    G = bounded_degree(n=n, m=m, max_weight=max_weight, rng=rng)
    return G, n


@st.composite
def iqp_parameters(
    draw,
    n_params: int,
    scheme: str = "uniform",
    std: float = 0.1,
) -> np.ndarray:
    """Draw IQP parameter vector θ under given initialization scheme."""
    if scheme == "uniform":
        return draw(
            arrays(
                dtype=np.float64,
                shape=(n_params,),
                elements=st.floats(min_value=-np.pi, max_value=np.pi),
            )
        )
    elif scheme == "small_angle":
        raw = draw(
            arrays(
                dtype=np.float64,
                shape=(n_params,),
                elements=st.floats(min_value=-4 * std, max_value=4 * std),
            )
        )
        return raw * std
    else:
        raise ValueError(f"Unknown scheme {scheme!r}")
