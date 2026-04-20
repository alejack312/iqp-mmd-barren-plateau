"""Tests for the exact small-n MMD² path.

Validates mmd2_exact_small_n() against:
  - large-sample Monte Carlo estimates (convergence)
  - hand-constructed toy distributions with known analytical values
  - structural invariants (determinism, nonnegativity, weights sum to 1)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iqp_bp.iqp.model import IQPModel
from iqp_bp.mmd.loss import mmd2, mmd2_exact_small_n


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _random_instance(n: int, m: int, seed: int):
    """Return (G, theta, data) for a random small instance."""
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
    theta = rng.uniform(-np.pi, np.pi, size=m)
    data = rng.integers(0, 2, size=(60, n), dtype=np.uint8)
    return G, theta, data


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_exact_is_deterministic():
    """Exact path has no randomness — identical inputs produce identical outputs."""
    G, theta, data = _random_instance(n=4, m=6, seed=0)
    r1 = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0)
    r2 = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0)
    assert r1 == r2


def test_exact_mmd2_nonneg():
    G, theta, data = _random_instance(n=4, m=6, seed=1)
    result = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0)
    assert result >= 0.0


def test_exact_contributions_nonneg():
    G, theta, data = _random_instance(n=4, m=6, seed=2)
    d = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0, return_details=True)
    assert np.all(d["contributions"] >= 0.0)


# ---------------------------------------------------------------------------
# return_details structure
# ---------------------------------------------------------------------------


def test_exact_details_structure():
    n = 4
    G, theta, data = _random_instance(n=n, m=6, seed=3)
    d = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0, return_details=True)

    assert set(d.keys()) == {"mmd2", "a_samples", "exp_p", "exp_q", "contributions", "weights"}
    assert d["a_samples"].shape == (2**n, n)
    assert d["exp_p"].shape == (2**n,)
    assert d["exp_q"].shape == (2**n,)
    assert d["contributions"].shape == (2**n,)
    assert d["weights"].shape == (2**n,)

    # weights sum to 1
    assert pytest.approx(d["weights"].sum(), abs=1e-12) == 1.0

    # mmd2 reconstructed from weighted contributions
    assert pytest.approx(d["mmd2"], abs=1e-12) == float(np.dot(d["weights"], d["contributions"]))

    # contributions are squared differences
    np.testing.assert_allclose(
        d["contributions"], (d["exp_p"] - d["exp_q"]) ** 2, atol=1e-14
    )


# ---------------------------------------------------------------------------
# Toy analytical case: n=2, point-mass data, theta=0
# ---------------------------------------------------------------------------


def test_exact_toy_n1_known_value():
    """n=1 Gaussian toy with a closed-form analytical answer.

    Setup:
      data = all [0] (point mass at 0)  →  ⟨Z_a⟩_p = 1 for both a ∈ {[0],[1]}
      G = [[1]], theta = [π/4]

    Model expectations:
      ⟨Z_{[0]}⟩_q = 1  (Z_0 is identity)
      ⟨Z_{[1]}⟩_q = mean_z cos(2θ·(-1)^z)
                   = (cos(π/2) + cos(-π/2)) / 2 = 0

    Gaussian spectral weights (n=1, sigma=1.0, tau=tanh(1/4)):
      w([0]) = 1 / (1 + tau)
      w([1]) = tau / (1 + tau)

    Contributions:
      a=[0]: (1 - 1)^2 = 0
      a=[1]: (1 - 0)^2 = 1

    Expected MMD² = 0·w([0]) + 1·w([1]) = tau / (1 + tau)
    """
    sigma = 1.0
    tau = float(np.tanh(1.0 / (4.0 * sigma**2)))
    expected_mmd2 = tau / (1.0 + tau)

    n, m = 1, 1
    G = np.ones((m, n), dtype=np.uint8)       # single generator g=[1]
    theta = np.array([np.pi / 4])
    data = np.zeros((20, n), dtype=np.uint8)  # point mass at [0]

    result = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=sigma)
    assert pytest.approx(result, abs=1e-12) == expected_mmd2


# ---------------------------------------------------------------------------
# Zero MMD² when p = q
# ---------------------------------------------------------------------------


def test_exact_zero_when_p_equals_q():
    """When data is sampled from q_θ itself, exact MMD² ≈ 0."""
    n, m = 4, 6
    rng = np.random.default_rng(42)
    G = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
    theta = rng.uniform(-np.pi, np.pi, size=m)

    # Sample data from the IQP model's exact probability distribution
    model = IQPModel(G, theta)
    probs = model.probability_vector_exact()
    data_indices = rng.choice(2**n, size=2000, p=probs)
    data = ((data_indices[:, None] >> np.arange(n)) & 1).astype(np.uint8)

    result = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0)
    # With 2000 samples the empirical ⟨Z_a⟩_p fluctuation drops materially, but
    # the weighted observable sum still has finite-sample noise; keep a modest
    # tolerance instead of expecting the empirical MMD² to vanish exactly.
    assert result < 0.08


# ---------------------------------------------------------------------------
# Convergence: exact path ≈ large-sample MC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,seed", [(4, 10), (6, 11), (8, 12)])
def test_exact_matches_mc_large_sample(n, seed):
    """Exact MMD² and large-sample MC agree within 5 standard errors."""
    m = n + 2
    G, theta, data = _random_instance(n=n, m=m, seed=seed)

    exact = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0)
    details = mmd2(
        theta, G, data,
        kernel="gaussian", sigma=1.0,
        num_a_samples=4000,
        num_z_samples=2000,
        rng=np.random.default_rng(seed + 100),
        return_details=True,
    )
    mc_est = details["mmd2"]
    stderr = details["mc_diagnostics"]["stderr"]

    assert abs(exact - mc_est) < 5 * stderr + 1e-3


# ---------------------------------------------------------------------------
# Kernel sweep: exact path runs for all supported kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel,extra", [
    ("gaussian", {"sigma": 1.0}),
    ("linear", {}),
    ("multi_scale_gaussian", {"sigmas": [0.5, 1.5], "weights": [0.4, 0.6]}),
])
def test_exact_kernel_sweep(kernel, extra):
    """Exact path should run without error for each supported kernel."""
    G, theta, data = _random_instance(n=4, m=6, seed=20)
    result = mmd2_exact_small_n(theta, G, data, kernel=kernel, **extra)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# Hypothesis: result always in [0, 4]
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=30, deadline=10_000)
def test_exact_result_in_valid_range(n, seed):
    """Exact MMD² must lie in [0, 4] for any inputs (exp differences in [-2,2])."""
    rng = np.random.default_rng(seed)
    m = n + 2
    G = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
    theta = rng.uniform(-np.pi, np.pi, size=m)
    data = rng.integers(0, 2, size=(20, n), dtype=np.uint8)
    result = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0)
    assert 0.0 <= result <= 4.0 + 1e-12
