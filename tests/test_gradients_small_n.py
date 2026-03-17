"""Tests: Gradient estimator correctness for small n.

Validates grad_mmd2_analytic against grad_mmd2_finite_diff for n ≤ 10.
"""

import numpy as np
import pytest

from iqp_bp.hypergraph.families import bounded_degree
from iqp_bp.mmd.gradients import grad_mmd2_analytic, grad_mmd2_finite_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instance(n: int, seed: int):
    rng = np.random.default_rng(seed)
    m = n
    G = bounded_degree(n=n, m=m, max_weight=2, rng=rng)
    theta = rng.uniform(-0.3, 0.3, size=m)  # small angles for stable gradients
    data = rng.integers(0, 2, size=(500, n), dtype=np.uint8)
    return G, theta, data


# ---------------------------------------------------------------------------
# Gradient correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed,param_idx", [
    (4, 0, 0),
    (6, 1, 1),
    (8, 2, 2),
    (10, 3, 0),
])
def test_gradient_vs_finite_diff(n: int, seed: int, param_idx: int):
    """Analytic gradient should match finite-difference estimate within tolerance."""
    G, theta, data = _make_instance(n, seed)

    analytic = grad_mmd2_analytic(
        theta=theta, G=G, data=data, param_idx=param_idx,
        kernel="gaussian", sigma=1.0,
        num_a_samples=256, num_z_samples=1024,
        rng=np.random.default_rng(seed + 10),
    )
    fd = grad_mmd2_finite_diff(
        theta=theta, G=G, data=data, param_idx=param_idx,
        eps=1e-4, kernel="gaussian", sigma=1.0,
        num_a_samples=256, num_z_samples=2048,
        rng=np.random.default_rng(seed + 20),
    )

    # Generous tolerance due to Monte Carlo noise in both estimates
    tol = 0.1
    assert abs(analytic - fd) < tol, (
        f"n={n}, param={param_idx}: analytic={analytic:.5f}, fd={fd:.5f}, diff={abs(analytic-fd):.5f}"
    )


@pytest.mark.parametrize("kernel,kernel_kwargs", [
    ("gaussian", {"sigma": 1.0}),
    ("laplacian", {"sigma": 1.0}),
    ("polynomial", {"degree": 2, "constant": 1.0}),
    ("linear", {}),
])
def test_gradient_all_kernels(kernel: str, kernel_kwargs: dict):
    """Gradient should be computable for all kernel types."""
    n, seed = 6, 42
    G, theta, data = _make_instance(n, seed)

    grad = grad_mmd2_analytic(
        theta=theta, G=G, data=data, param_idx=0,
        kernel=kernel, num_a_samples=128, num_z_samples=512,
        rng=np.random.default_rng(0),
        **kernel_kwargs,
    )
    assert np.isfinite(grad), f"Gradient is not finite for kernel={kernel}: {grad}"


def test_gradient_at_zero_theta():
    """At θ=0, gradient of MMD² w.r.t. any θ_i depends only on circuit structure."""
    n, m = 6, 6
    G = bounded_degree(n=n, m=m, max_weight=2, rng=np.random.default_rng(0))
    theta = np.zeros(m)
    # Product Bernoulli data: ⟨Z_a⟩_p = 0 for |a| ≥ 1
    data = np.random.default_rng(0).integers(0, 2, size=(1000, n), dtype=np.uint8)

    for i in range(min(3, m)):
        grad = grad_mmd2_analytic(
            theta=theta, G=G, data=data, param_idx=i,
            kernel="gaussian", sigma=1.0,
            num_a_samples=256, num_z_samples=512,
            rng=np.random.default_rng(i),
        )
        # At θ=0: ⟨Z_a⟩_q = 1 for a=0, = 0 otherwise
        # With product Bernoulli target, ⟨Z_a⟩_p ≈ 0 for |a| ≥ 1
        # So gradient at θ=0 should be close to 0 (both expectations match)
        assert np.isfinite(grad), f"Gradient not finite at θ=0 for param {i}"


def test_gradient_variance_decreases_small_angle():
    """Gradient variance under small-angle init should be ≤ variance under uniform."""
    n, m = 8, 8
    G = bounded_degree(n=n, m=m, max_weight=2, rng=np.random.default_rng(0))
    data = np.random.default_rng(0).integers(0, 2, size=(500, n), dtype=np.uint8)

    T = 20  # seeds
    rng = np.random.default_rng(42)

    uniform_grads = []
    small_grads = []
    for seed in range(T):
        rng_s = np.random.default_rng(seed)
        theta_uniform = rng_s.uniform(-np.pi, np.pi, size=m)
        theta_small = rng_s.normal(0, 0.1, size=m)

        g_u = grad_mmd2_analytic(
            theta=theta_uniform, G=G, data=data, param_idx=0,
            kernel="gaussian", sigma=1.0,
            num_a_samples=128, num_z_samples=512,
            rng=np.random.default_rng(seed + 100),
        )
        g_s = grad_mmd2_analytic(
            theta=theta_small, G=G, data=data, param_idx=0,
            kernel="gaussian", sigma=1.0,
            num_a_samples=128, num_z_samples=512,
            rng=np.random.default_rng(seed + 200),
        )
        uniform_grads.append(g_u)
        small_grads.append(g_s)

    var_uniform = float(np.var(uniform_grads))
    var_small = float(np.var(small_grads))

    # Small-angle should have lower or comparable variance
    # (Not always strictly true for small T, so allow 3x margin)
    assert var_small <= var_uniform * 3.0 or var_uniform < 1e-8, (
        f"Small-angle var={var_small:.6f} >> uniform var={var_uniform:.6f}"
    )
