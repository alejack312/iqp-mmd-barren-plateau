from __future__ import annotations

import numpy as np

from iqp_bp.distributions import (
    draw_samples_from_probability_vector,
    exact_fourier_coefficients,
    exact_marginal,
    sample_fourier_coefficient,
    sample_marginal,
)


def _all_masks(n: int) -> np.ndarray:
    indices = np.arange(2**n, dtype=np.uint64)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)


def test_exact_marginal_has_expected_shape_and_normalization():
    probabilities = np.array([0.05, 0.10, 0.10, 0.15, 0.10, 0.15, 0.20, 0.15], dtype=float)
    marginal = exact_marginal(probabilities, subset=[0, 2])

    assert marginal.shape == (4,)
    assert np.isclose(marginal.sum(), 1.0)
    assert np.all(marginal >= 0.0)


def test_sample_paths_converge_to_exact_marginal_and_fourier_values():
    probabilities = np.array([0.05, 0.10, 0.10, 0.15, 0.10, 0.15, 0.20, 0.15], dtype=float)
    rng = np.random.default_rng(7)
    samples = draw_samples_from_probability_vector(probabilities, 20000, rng=rng)

    exact = exact_marginal(probabilities, subset=[0, 2])
    empirical = sample_marginal(samples, subset=[0, 2])
    assert np.allclose(empirical, exact, atol=0.02)

    mask = np.array([1, 0, 1], dtype=np.uint8)
    exact_coeff = exact_fourier_coefficients(probabilities, mask)[0]
    sample_coeff = sample_fourier_coefficient(samples, mask)
    assert np.isclose(sample_coeff, exact_coeff, atol=0.03)


def test_fourier_coefficients_reconstruct_probability_vector():
    probabilities = np.array([0.08, 0.12, 0.14, 0.16, 0.10, 0.18, 0.12, 0.10], dtype=float)
    n = 3
    masks = _all_masks(n)
    basis = _all_masks(n)
    coeffs = exact_fourier_coefficients(probabilities, masks)
    signs = 1.0 - 2.0 * ((basis.astype(np.int64) @ masks.T.astype(np.int64)) % 2).astype(float)
    reconstructed = (signs @ coeffs) / (2**n)

    assert np.allclose(reconstructed, probabilities)
