"""Tests for the detailed MMD² diagnostic path (return_details=True)."""

from __future__ import annotations

import numpy as np
import pytest

from iqp_bp.mmd.loss import mmd2


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def small_instance():
    rng = np.random.default_rng(42)
    n, m = 4, 6
    G = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
    theta = rng.uniform(-np.pi, np.pi, size=m)
    data = rng.integers(0, 2, size=(50, n), dtype=np.uint8)
    return {"G": G, "theta": theta, "data": data, "n": n, "m": m}


# ---------------------------------------------------------------------------
# Shape and key contract
# ---------------------------------------------------------------------------


def test_details_keys(small_instance):
    inst = small_instance
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
        rng=np.random.default_rng(0),
        return_details=True,
    )
    assert set(details.keys()) == {"mmd2", "a_samples", "exp_p", "exp_q", "contributions", "mc_diagnostics"}


def test_details_shape(small_instance):
    inst = small_instance
    B = 32
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=B,
        num_z_samples=64,
        rng=np.random.default_rng(0),
        return_details=True,
    )
    assert details["a_samples"].shape == (B, inst["n"])
    assert details["exp_p"].shape == (B,)
    assert details["exp_q"].shape == (B,)
    assert details["contributions"].shape == (B,)


def test_mc_diagnostics_keys(small_instance):
    inst = small_instance
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
        rng=np.random.default_rng(0),
        return_details=True,
    )
    assert set(details["mc_diagnostics"].keys()) == {
        "point_estimate", "sample_std", "stderr", "num_samples"
    }


def test_mc_diagnostics_num_samples(small_instance):
    inst = small_instance
    B = 48
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=B,
        num_z_samples=64,
        rng=np.random.default_rng(0),
        return_details=True,
    )
    assert details["mc_diagnostics"]["num_samples"] == B


# ---------------------------------------------------------------------------
# Scalar reconstruction from per-observable contributions
# ---------------------------------------------------------------------------


def test_scalar_reconstructed_from_contributions(small_instance):
    """mmd2 == mean(contributions) to floating-point precision."""
    inst = small_instance
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=64,
        num_z_samples=128,
        rng=np.random.default_rng(1),
        return_details=True,
    )
    reconstructed = float(np.mean(details["contributions"]))
    assert reconstructed == pytest.approx(details["mmd2"], rel=1e-12)
    assert reconstructed == pytest.approx(details["mc_diagnostics"]["point_estimate"], rel=1e-12)


def test_contributions_equal_squared_diffs(small_instance):
    """contributions[i] == (exp_p[i] - exp_q[i])^2 exactly."""
    inst = small_instance
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
        rng=np.random.default_rng(2),
        return_details=True,
    )
    expected = (details["exp_p"] - details["exp_q"]) ** 2
    np.testing.assert_array_equal(details["contributions"], expected)


# ---------------------------------------------------------------------------
# Estimator uncertainty consistency
# ---------------------------------------------------------------------------


def test_contributions_are_nonneg(small_instance):
    inst = small_instance
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
        rng=np.random.default_rng(3),
        return_details=True,
    )
    assert np.all(details["contributions"] >= 0), "Per-observable contributions must be non-negative"


def test_stderr_equals_std_over_sqrt_n(small_instance):
    inst = small_instance
    B = 64
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=B,
        num_z_samples=64,
        rng=np.random.default_rng(4),
        return_details=True,
    )
    diag = details["mc_diagnostics"]
    expected_stderr = diag["sample_std"] / np.sqrt(B)
    assert diag["stderr"] == pytest.approx(expected_stderr, rel=1e-12)


def test_sample_std_matches_contributions(small_instance):
    inst = small_instance
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=64,
        num_z_samples=64,
        rng=np.random.default_rng(5),
        return_details=True,
    )
    diag = details["mc_diagnostics"]
    assert diag["sample_std"] == pytest.approx(float(np.std(details["contributions"])), rel=1e-12)


# ---------------------------------------------------------------------------
# Deterministic / seeded outputs
# ---------------------------------------------------------------------------


def test_seeded_determinism(small_instance):
    """Two calls with the same seed must produce identical outputs."""
    inst = small_instance
    kwargs = dict(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
        return_details=True,
    )
    d1 = mmd2(**kwargs, rng=np.random.default_rng(42))
    d2 = mmd2(**kwargs, rng=np.random.default_rng(42))

    assert d1["mmd2"] == d2["mmd2"]
    np.testing.assert_array_equal(d1["a_samples"], d2["a_samples"])
    np.testing.assert_array_equal(d1["exp_p"], d2["exp_p"])
    np.testing.assert_array_equal(d1["exp_q"], d2["exp_q"])
    np.testing.assert_array_equal(d1["contributions"], d2["contributions"])
    assert d1["mc_diagnostics"] == d2["mc_diagnostics"]


def test_different_seeds_differ(small_instance):
    """Two calls with different seeds should (almost certainly) differ."""
    inst = small_instance
    kwargs = dict(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
        return_details=True,
    )
    d1 = mmd2(**kwargs, rng=np.random.default_rng(0))
    d2 = mmd2(**kwargs, rng=np.random.default_rng(99))
    # a_samples depend on the kernel RNG stream; different seeds → different words
    assert not np.array_equal(d1["a_samples"], d2["a_samples"])


# ---------------------------------------------------------------------------
# Backward-compatibility: scalar contract unchanged
# ---------------------------------------------------------------------------


def test_default_returns_float(small_instance):
    inst = small_instance
    result = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
        rng=np.random.default_rng(6),
    )
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_scalar_equals_details_mmd2(small_instance):
    """Scalar path and detailed path agree when given the same seed."""
    inst = small_instance
    common_kwargs = dict(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=32,
        num_z_samples=64,
    )
    scalar = mmd2(**common_kwargs, rng=np.random.default_rng(7))
    details = mmd2(**common_kwargs, rng=np.random.default_rng(7), return_details=True)
    assert scalar == pytest.approx(details["mmd2"], rel=1e-12)


def test_mc_details_vs_exact_small_n(small_instance):
    """MC path with large samples should be within 5 stderr of the exact value."""
    from iqp_bp.mmd.loss import mmd2_exact_small_n

    inst = small_instance  # n=4
    exact = mmd2_exact_small_n(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
    )
    details = mmd2(
        theta=inst["theta"],
        G=inst["G"],
        data=inst["data"],
        kernel="gaussian",
        sigma=1.0,
        num_a_samples=2000,
        num_z_samples=2000,
        rng=np.random.default_rng(99),
        return_details=True,
    )
    stderr = details["mc_diagnostics"]["stderr"]
    assert abs(details["mmd2"] - exact) < 5 * stderr + 1e-4
