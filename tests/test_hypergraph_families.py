import numpy as np
import pytest

from iqp_bp.hypergraph.families import erdos_renyi, lattice


def _expected_square_lattice_edges(side: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(side):
        for j in range(side):
            q = i * side + j
            if j + 1 < side:
                edges.add((q, q + 1))
            if i + 1 < side:
                edges.add((q, q + side))
    return edges


def _rows_to_edge_set(G: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for row in G:
        qubits = np.flatnonzero(row)
        assert len(qubits) == 2
        edges.add(tuple(int(q) for q in qubits))
    return edges


def test_lattice_2d_returns_exact_square_grid_edges():
    n = 16
    side = 4
    G = lattice(n=n, m=n, dimension=2, range_=1, rng=np.random.default_rng(0))

    assert G.dtype == np.uint8
    assert G.shape == (2 * side * (side - 1), n)
    assert np.all((G == 0) | (G == 1))
    assert np.all(G.sum(axis=1) == 2)
    assert _rows_to_edge_set(G) == _expected_square_lattice_edges(side)


def test_lattice_2d_is_deterministic_across_rng_seeds():
    G0 = lattice(n=9, m=9, dimension=2, range_=1, rng=np.random.default_rng(0))
    G1 = lattice(n=9, m=9, dimension=2, range_=1, rng=np.random.default_rng(999))
    assert np.array_equal(G0, G1)


def test_lattice_2d_rejects_non_square_n():
    with pytest.raises(ValueError, match="perfect square"):
        lattice(n=6, m=6, dimension=2, range_=1, rng=np.random.default_rng(0))


def test_lattice_2d_rejects_non_nearest_neighbor_range():
    with pytest.raises(ValueError, match="range_ must be 1"):
        lattice(n=16, m=16, dimension=2, range_=2, rng=np.random.default_rng(0))


def test_erdos_renyi_returns_pairwise_binary_edges():
    n = 32
    G = erdos_renyi(n=n, m=n, p_edge=2.0, rng=np.random.default_rng(0))

    assert G.dtype == np.uint8
    assert G.shape[1] == n
    assert np.all((G == 0) | (G == 1))
    assert np.all(G.sum(axis=1) == 2)
    assert len(_rows_to_edge_set(G)) == G.shape[0]


def test_erdos_renyi_is_reproducible_for_fixed_seed():
    G0 = erdos_renyi(n=32, m=32, p_edge=2.0, rng=np.random.default_rng(7))
    G1 = erdos_renyi(n=32, m=32, p_edge=2.0, rng=np.random.default_rng(7))
    assert np.array_equal(G0, G1)


def test_erdos_renyi_expected_degree_matches_sparse_calibration():
    n = 128
    target_degree = 2.0
    mean_degrees = []
    for seed in range(40):
        G = erdos_renyi(n=n, m=n, p_edge=target_degree, rng=np.random.default_rng(seed))
        mean_degrees.append(2.0 * G.shape[0] / n)

    empirical_mean_degree = float(np.mean(mean_degrees))
    assert abs(empirical_mean_degree - target_degree) < 0.4
