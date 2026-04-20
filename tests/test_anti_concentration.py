"""Tests for the exact anti-concentration checker."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from iqp_bp.experiments import check_anti_concentration
from iqp_bp.hypergraph.hypothesis_strategies import anti_concentration_probability_vector


def _threshold_map(result: dict) -> dict[float, dict]:
    """Index threshold checks by alpha for cleaner assertions."""
    return {entry["alpha"]: entry for entry in result["threshold_checks"]}


def test_uniform_distribution_metrics_match_exact_values():
    """Uniform p should reproduce the exact baseline statistics."""
    p = np.full(8, 1.0 / 8.0, dtype=np.float64)

    result = check_anti_concentration(
        p,
        alphas=(0.5, 1.0, 2.0),
        primary_alpha=1.0,
        beta_min=0.25,
        second_moment_threshold=1.0,
    )
    threshold_checks = _threshold_map(result)

    assert result["n"] == 3
    assert math.isclose(result["collision_probability"], 1.0 / 8.0)
    assert math.isclose(result["scaled_second_moment"], 1.0)
    assert result["passes_second_moment_threshold"] is True
    assert math.isclose(result["max_probability"], 1.0 / 8.0)
    assert math.isclose(result["max_probability_scaled"], 1.0)
    assert math.isclose(result["effective_support"], 8.0)
    assert math.isclose(result["effective_support_ratio"], 1.0)
    assert math.isclose(threshold_checks[0.5]["beta_hat"], 1.0)
    assert math.isclose(threshold_checks[1.0]["beta_hat"], 1.0)
    assert math.isclose(threshold_checks[2.0]["beta_hat"], 0.0)
    assert result["passes_primary_threshold"] is True


def test_delta_distribution_fails_threshold_diagnostic_but_has_large_second_moment():
    """A concentrated distribution should be obvious in beta_hat(alpha)."""
    p = np.zeros(8, dtype=np.float64)
    p[0] = 1.0

    result = check_anti_concentration(
        p,
        alphas=(0.5, 1.0, 2.0),
        primary_alpha=1.0,
        beta_min=0.25,
        second_moment_threshold=1.0,
    )
    threshold_checks = _threshold_map(result)

    assert math.isclose(result["scaled_second_moment"], 8.0)
    assert result["passes_second_moment_threshold"] is True
    assert math.isclose(result["max_probability_scaled"], 8.0)
    assert math.isclose(result["effective_support"], 1.0)
    assert math.isclose(threshold_checks[0.5]["beta_hat"], 1.0 / 8.0)
    assert math.isclose(threshold_checks[1.0]["beta_hat"], 1.0 / 8.0)
    assert math.isclose(threshold_checks[2.0]["beta_hat"], 1.0 / 8.0)
    assert result["passes_primary_threshold"] is False


def test_checker_requires_power_of_two_probability_vector():
    """The exact small-n contract requires length 2**n."""
    with pytest.raises(ValueError, match="power of two"):
        check_anti_concentration([0.5, 0.25, 0.25])


def test_checker_requires_normalized_probability_vector():
    """The exact checker should reject malformed probability vectors."""
    with pytest.raises(ValueError, match="sum to 1"):
        check_anti_concentration([0.2, 0.2, 0.2, 0.2])


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(probabilities=anti_concentration_probability_vector())
def test_checker_reports_bounded_consistent_diagnostics(probabilities: np.ndarray):
    """Any valid probability vector should satisfy the deterministic identities."""
    result = check_anti_concentration(
        probabilities,
        alphas=(0.5, 1.0, 2.0, 4.0),
        primary_alpha=1.0,
        beta_min=0.25,
        second_moment_threshold=1.0,
    )
    num_outcomes = len(probabilities)
    collision_probability = float(np.sum(probabilities**2))

    assert result["n"] == int(np.log2(num_outcomes))
    assert result["num_outcomes"] == num_outcomes
    assert math.isclose(result["uniform_probability"], 1.0 / num_outcomes)
    assert math.isclose(result["collision_probability"], collision_probability)
    assert math.isclose(result["scaled_second_moment"], num_outcomes * collision_probability)
    assert math.isclose(result["effective_support"], 1.0 / collision_probability)
    assert math.isclose(result["effective_support_ratio"], result["effective_support"] / num_outcomes)
    assert 0.0 <= result["primary_beta_hat"] <= 1.0
    assert 0.0 <= result["max_probability"] <= 1.0
    assert 1.0 <= result["max_probability_scaled"] <= num_outcomes
    assert 1.0 <= result["effective_support"] <= num_outcomes
    assert (1.0 / num_outcomes) <= result["effective_support_ratio"] <= 1.0


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    probabilities=anti_concentration_probability_vector(),
    primary_alpha=st.sampled_from([0.5, 1.0, 1.5, 2.0, 4.0]),
    beta_min=st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]),
)
def test_threshold_checks_are_monotone_and_primary_consistent(
    probabilities: np.ndarray,
    primary_alpha: float,
    beta_min: float,
):
    """`beta_hat(alpha)` should shrink with stricter thresholds."""
    result = check_anti_concentration(
        probabilities,
        alphas=(0.5, 1.0, 2.0),
        primary_alpha=primary_alpha,
        beta_min=beta_min,
        second_moment_threshold=1.0,
    )
    threshold_checks = result["threshold_checks"]
    threshold_map = _threshold_map(result)

    assert float(primary_alpha) in threshold_map
    assert math.isclose(
        result["primary_beta_hat"],
        threshold_map[float(primary_alpha)]["beta_hat"],
    )
    assert (
        result["passes_primary_threshold"]
        == threshold_map[float(primary_alpha)]["passes_beta_threshold"]
    )

    beta_hats = [entry["beta_hat"] for entry in threshold_checks]
    for previous, current in zip(beta_hats, beta_hats[1:]):
        assert previous >= current - 1e-12

    for entry in threshold_checks:
        assert 0.0 <= entry["beta_hat"] <= 1.0
        assert math.isclose(
            entry["threshold_probability"],
            entry["alpha"] * result["uniform_probability"],
        )


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(probabilities=anti_concentration_probability_vector())
def test_checker_is_invariant_under_outcome_permutation(probabilities: np.ndarray):
    """Relabeling bitstrings should not change any anti-concentration diagnostic."""
    baseline = check_anti_concentration(
        probabilities,
        alphas=(0.5, 1.0, 1.5, 2.0),
        primary_alpha=1.5,
        beta_min=0.25,
        second_moment_threshold=1.0,
    )
    permuted = check_anti_concentration(
        probabilities[::-1],
        alphas=(0.5, 1.0, 1.5, 2.0),
        primary_alpha=1.5,
        beta_min=0.25,
        second_moment_threshold=1.0,
    )

    scalar_fields = [
        "n",
        "num_outcomes",
        "uniform_probability",
        "collision_probability",
        "scaled_second_moment",
        "second_moment_threshold",
        "passes_second_moment_threshold",
        "max_probability",
        "max_probability_scaled",
        "effective_support",
        "effective_support_ratio",
        "beta_min",
        "primary_alpha",
        "primary_beta_hat",
        "passes_primary_threshold",
    ]
    for field in scalar_fields:
        baseline_value = baseline[field]
        permuted_value = permuted[field]
        if isinstance(baseline_value, float):
            assert math.isclose(baseline_value, permuted_value)
        else:
            assert baseline_value == permuted_value

    assert len(baseline["threshold_checks"]) == len(permuted["threshold_checks"])
    for baseline_entry, permuted_entry in zip(
        baseline["threshold_checks"],
        permuted["threshold_checks"],
    ):
        assert math.isclose(baseline_entry["alpha"], permuted_entry["alpha"])
        assert math.isclose(
            baseline_entry["threshold_probability"],
            permuted_entry["threshold_probability"],
        )
        assert math.isclose(baseline_entry["beta_hat"], permuted_entry["beta_hat"])
        assert (
            baseline_entry["passes_beta_threshold"]
            == permuted_entry["passes_beta_threshold"]
        )
