"""Tests for the exact anti-concentration checker."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iqp_bp.experiments import check_anti_concentration


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
