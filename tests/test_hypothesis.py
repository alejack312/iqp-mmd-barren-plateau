import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from iqp_bp.hypergraph.hypothesis_strategies import (
    family_config,
    init_config,
    iqp_parameters,
    kernel_config,
    mmd_instance,
    smart_family_instance,
)
from iqp_bp.mmd.gradients import (
    estimate_gradient_variance,
    grad_mmd2_analytic,
    grad_mmd2_finite_diff,
)
from iqp_bp.mmd.loss import mmd2


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
        edges.add(tuple(int(q) for q in qubits))
    return edges


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=mmd_instance())
def test_mmd2_is_finite(instance):
    value = mmd2(
        theta=instance["theta"],
        G=instance["G"],
        data=instance["data"],
        kernel=instance["kernel"],
        rng=np.random.default_rng(0),
        **instance["kernel_kwargs"],
    )
    assert np.isfinite(value), "MMD is not finite"


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=mmd_instance())
def test_gradient_is_finite_for_sampled_combinations(instance):
    grad = grad_mmd2_analytic(
        theta=instance["theta"],
        G=instance["G"],
        data=instance["data"],
        param_idx=0,
        kernel=instance["kernel"],
        num_a_samples=64,
        num_z_samples=256,
        rng=np.random.default_rng(1),
        **instance["kernel_kwargs"],
    )
    assert np.isfinite(grad), "Gradient is not finite"


@st.composite
def fixed_family_smart_instance(draw, family_name: str):
    return draw(smart_family_instance(family_name=family_name))


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=fixed_family_smart_instance("product_state"))
def test_smart_product_state_family_smoke(instance):
    G = instance["G"]
    n = instance["n"]
    assert G.dtype == np.uint8
    assert G.shape == (n, n)
    assert np.all((G == 0) | (G == 1))
    assert np.all(G.sum(axis=1) == 1)
    assert np.array_equal(G, np.eye(n, dtype=np.uint8))


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=fixed_family_smart_instance("lattice"))
def test_smart_lattice_family_smoke(instance):
    G = instance["G"]
    n = instance["n"]
    side = int(np.sqrt(n))
    assert side * side == n
    assert G.dtype == np.uint8
    assert np.all((G == 0) | (G == 1))
    assert np.all(G.sum(axis=1) == 2)
    assert G.shape == (2 * side * (side - 1), n)
    assert _rows_to_edge_set(G) == _expected_square_lattice_edges(side)


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=fixed_family_smart_instance("erdos_renyi"))
def test_smart_erdos_renyi_family_smoke(instance):
    G = instance["G"]
    n = instance["n"]
    assert G.dtype == np.uint8
    assert G.shape[1] == n
    assert np.all((G == 0) | (G == 1))
    assert G.shape[0] >= 1
    assert np.all(G.sum(axis=1) == 2)
    assert len(_rows_to_edge_set(G)) == G.shape[0]


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=fixed_family_smart_instance("complete_graph"))
def test_smart_complete_graph_family_smoke(instance):
    G = instance["G"]
    n = instance["n"]
    expected_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    assert G.dtype == np.uint8
    assert G.shape == (n * (n - 1) // 2, n)
    assert np.all((G == 0) | (G == 1))
    assert np.all(G.sum(axis=1) == 2)
    assert _rows_to_edge_set(G) == expected_pairs


@st.composite
def small_n_instance(draw):
    family_name, _ = draw(family_config())
    if family_name == "bounded_degree":
        from iqp_bp.hypergraph.hypothesis_strategies import bounded_degree_hypergraph

        G, n = draw(bounded_degree_hypergraph(n_min=4, n_max=7, max_weight=3))
    elif family_name == "hypergraph_matrix":
        from iqp_bp.hypergraph.hypothesis_strategies import hypergraph_matrix

        G, n, _ = draw(hypergraph_matrix(n_min=4, n_max=7, m_min=4, m_max=14))
    else:
        raise ValueError(f"Unknown family {family_name!r}")

    kernel, kernel_kwargs = draw(kernel_config())
    theta = draw(iqp_parameters(n_params=G.shape[0], scheme="small_angle", std=0.1))
    data = draw(
        st.builds(
            lambda arr: arr.astype(np.uint8),
            st.lists(
                st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n),
                min_size=96,
                max_size=96,
            ).map(np.array),
        )
    )
    return {
        "G": G,
        "theta": theta,
        "data": data,
        "kernel": kernel,
        "kernel_kwargs": kernel_kwargs,
    }


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=mmd_instance())
def test_small_angle_and_uniform_inits_have_distinct_variance_profiles(instance):
    G = instance["G"]
    data = instance["data"]
    kernel = instance["kernel"]
    kernel_kwargs = instance["kernel_kwargs"]
    m = G.shape[0]

    uniform_thetas = [
        np.random.default_rng(seed).uniform(-np.pi, np.pi, size=m)
        for seed in range(5)
    ]
    small_angle_thetas = [
        np.random.default_rng(seed + 100).normal(0.0, 0.1, size=m)
        for seed in range(5)
    ]

    uniform_stats = estimate_gradient_variance(
        G=G,
        data=data,
        param_idx=0,
        theta_seeds=uniform_thetas,
        kernel=kernel,
        num_a_samples=64,
        num_z_samples=256,
        rng=np.random.default_rng(10),
        **kernel_kwargs,
    )
    small_stats = estimate_gradient_variance(
        G=G,
        data=data,
        param_idx=0,
        theta_seeds=small_angle_thetas,
        kernel=kernel,
        num_a_samples=64,
        num_z_samples=256,
        rng=np.random.default_rng(11),
        **kernel_kwargs,
    )

    assert np.isfinite(uniform_stats["var"]), "Uniform-init variance is not finite"
    assert np.isfinite(small_stats["var"]), "Small-angle variance is not finite"
    assert abs(uniform_stats["var"] - small_stats["var"]) > 1e-12, (
        f"variance profiles are indistinguishable: uniform={uniform_stats['var']}, "
        f"small_angle={small_stats['var']}"
    )


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(cfg=init_config())
def test_init_config_small_angle_uses_locked_sweep(cfg):
    scheme, kwargs = cfg
    if scheme == "small_angle":
        assert kwargs["std"] in {0.01, 0.1, 0.3}
    elif scheme == "data_dependent":
        assert kwargs == {}


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instance=mmd_instance())
def test_sampled_theta_matches_generator_shape_and_is_finite(instance):
    theta = instance["theta"]
    G = instance["G"]
    assert theta.shape == (G.shape[0],)
    assert np.all(np.isfinite(theta))
    assert instance["init_scheme"] in {"uniform", "small_angle", "data_dependent"}
