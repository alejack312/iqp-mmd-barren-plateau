"""Tests: IQP expectation estimator correctness for small n.

Validates iqp_expectation against iqp_expectation_exact (brute-force)
for n ≤ 12.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iqp_bp.hypergraph.families import bounded_degree, lattice
from iqp_bp.iqp.expectation import iqp_expectation, iqp_expectation_exact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_instance(n: int, m: int, seed: int):
    rng = np.random.default_rng(seed)
    G = bounded_degree(n=n, m=m, max_weight=2, rng=rng)
    theta = rng.uniform(-np.pi, np.pi, size=m)
    a = rng.integers(0, 2, size=n, dtype=np.uint8)
    a[0] = 1  # ensure a is nonzero
    return G, theta, a


# ---------------------------------------------------------------------------
# Deterministic tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(4, 0), (6, 1), (8, 2), (10, 3), (12, 4)])
def test_expectation_vs_exact(n: int, seed: int):
    """Monte Carlo estimate should be within 3 sigma of exact value."""
    G, theta, a = _random_instance(n, m=n, seed=seed)

    exact = iqp_expectation_exact(theta, G, a)
    estimate, stderr = iqp_expectation(
        theta, G, a, num_z_samples=4096, rng=np.random.default_rng(seed + 100)
    )

    # 3-sigma tolerance
    tol = max(3 * stderr, 0.05)
    assert abs(estimate - exact) < tol, (
        f"n={n}: estimate={estimate:.4f}, exact={exact:.4f}, stderr={stderr:.4f}, tol={tol:.4f}"
    )


def test_expectation_trivial_theta():
    """At θ=0, ⟨Z_a⟩ = 1 for a=0 and depends on circuit for a≠0."""
    n, m = 6, 6
    G = bounded_degree(n=n, m=m, max_weight=2, rng=np.random.default_rng(0))
    theta = np.zeros(m)

    # For Z_a with a=0 (identity), exact value should be 1.0
    a_zero = np.zeros(n, dtype=np.uint8)
    val = iqp_expectation_exact(theta, G, a_zero)
    assert abs(val - 1.0) < 1e-10, f"Expected 1.0 for identity observable, got {val}"


def test_expectation_range():
    """⟨Z_a⟩ must always be in [-1, 1]."""
    n, m = 8, 8
    G, theta, a = _random_instance(n, m, seed=42)
    val = iqp_expectation_exact(theta, G, a)
    assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10, f"Out of range: {val}"


def test_estimator_convergence():
    """Variance of estimator should decrease with more samples."""
    n, m = 6, 6
    G, theta, a = _random_instance(n, m, seed=7)
    exact = iqp_expectation_exact(theta, G, a)

    errors = []
    for nz in [64, 256, 1024, 4096]:
        est, _ = iqp_expectation(theta, G, a, num_z_samples=nz, rng=np.random.default_rng(0))
        errors.append(abs(est - exact))

    # Errors should generally decrease (not strict monotone due to randomness)
    assert errors[-1] < errors[0] + 0.1, "Estimator did not converge with more samples"


# ---------------------------------------------------------------------------
# Property-based tests via Hypothesis
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=20, deadline=5000)
def test_expectation_in_range_hypothesis(n: int, seed: int):
    """Property: ⟨Z_a⟩ ∈ [-1, 1] for any (n, G, θ, a)."""
    G, theta, a = _random_instance(n, m=n, seed=seed)
    val = iqp_expectation_exact(theta, G, a)
    assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9


@given(
    n=st.integers(min_value=3, max_value=10),
    seed=st.integers(min_value=0, max_value=500),
)
@settings(max_examples=10, deadline=10000)
def test_mc_estimate_close_to_exact_hypothesis(n: int, seed: int):
    """Property: MC estimate with large budget is close to exact (n ≤ 10)."""
    G, theta, a = _random_instance(n, m=n, seed=seed)
    exact = iqp_expectation_exact(theta, G, a)
    est, stderr = iqp_expectation(
        theta, G, a, num_z_samples=4096, rng=np.random.default_rng(seed + 999)
    )
    tol = max(5 * stderr, 0.1)
    assert abs(est - exact) < tol, f"n={n}: |est-exact|={abs(est-exact):.4f} > tol={tol:.4f}"
