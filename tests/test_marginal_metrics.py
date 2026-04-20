from __future__ import annotations

import numpy as np

from iqp_bp.distributions import marginal_chi_square, marginal_tv, summarize_by_order
from iqp_bp.distributions.marginals import exact_fourier_coefficients
from iqp_bp.mmd.kernel import spectral_weights_exact


def _all_masks(n: int) -> np.ndarray:
    indices = np.arange(2**n, dtype=np.uint64)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)


def test_identical_distributions_have_zero_marginal_metrics():
    probabilities = np.array([0.125] * 8, dtype=float)
    summary = summarize_by_order(probabilities, probabilities, n=3, sigma=2.0)

    assert marginal_tv(probabilities[:4], probabilities[:4]) == 0.0
    assert marginal_chi_square(probabilities[:4], probabilities[:4]) == 0.0
    assert summary["weighted_mmd2_total"] == 0.0
    assert all(order["mean_tv"] == 0.0 for order in summary["orders"])


def test_delta_vs_uniform_shows_larger_high_order_mismatch():
    uniform = np.array([0.125] * 8, dtype=float)
    delta = np.zeros(8, dtype=float)
    delta[0] = 1.0
    summary = summarize_by_order(uniform, delta, n=3, sigma=2.0)

    order_to_tv = {entry["order"]: entry["mean_tv"] for entry in summary["orders"]}
    assert order_to_tv[1] < order_to_tv[2] < order_to_tv[3]


def test_order_summary_reconstructs_gaussian_weighted_mmd_total():
    p = np.array([0.125] * 8, dtype=float)
    q = np.array([0.40, 0.15, 0.10, 0.05, 0.10, 0.08, 0.07, 0.05], dtype=float)
    sigma = 1.7
    summary = summarize_by_order(p, q, n=3, sigma=sigma)

    masks = _all_masks(3)
    coeff_p = exact_fourier_coefficients(p, masks)
    coeff_q = exact_fourier_coefficients(q, masks)
    weights = spectral_weights_exact("gaussian", 3, sigma=sigma)
    expected = float(np.dot(weights, (coeff_p - coeff_q) ** 2))

    assert np.isclose(summary["weighted_mmd2_total"], expected)
