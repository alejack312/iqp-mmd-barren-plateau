from math import comb

import numpy as np

from iqp_bp.mmd.kernel import (
    gaussian_kernel,
    gaussian_sample_a,
    gaussian_spectral_weights,
    multi_scale_gaussian_kernel,
)


def test_gaussian_kernel_uses_locked_half_sigma_squared_normalization():
    x = np.array([0, 1, 0, 1], dtype=np.uint8)
    y = np.array([1, 1, 0, 0], dtype=np.uint8)  # Hamming distance 2
    sigma = 1.7

    expected = np.exp(-2 / (2 * sigma**2))
    assert np.isclose(gaussian_kernel(x, y, sigma=sigma), expected)


def test_gaussian_spectral_weights_use_locked_tau():
    n = 5
    sigma = 1.3
    tau = np.tanh(1.0 / (4.0 * sigma**2))

    expected = tau ** np.arange(n + 1)
    actual = gaussian_spectral_weights(n=n, sigma=sigma)
    assert np.allclose(actual, expected)


def test_gaussian_sample_a_matches_locked_weight_distribution():
    n = 6
    sigma = 1.1
    num_a_samples = 20000
    tau = np.tanh(1.0 / (4.0 * sigma**2))

    rng = np.random.default_rng(7)
    samples = gaussian_sample_a(n=n, num_a_samples=num_a_samples, sigma=sigma, rng=rng)
    weights = samples.sum(axis=1)
    empirical = np.bincount(weights, minlength=n + 1) / num_a_samples

    unnormalized = np.array([comb(n, w) * (tau ** w) for w in range(n + 1)], dtype=float)
    expected = unnormalized / unnormalized.sum()

    assert np.allclose(empirical, expected, atol=0.02)


def test_multi_scale_gaussian_kernel_matches_weighted_component_average():
    x = np.array([0, 1, 1, 0], dtype=np.uint8)
    y = np.array([1, 1, 0, 0], dtype=np.uint8)
    sigmas = [0.7, 1.4, 2.1]
    weights = [0.2, 0.3, 0.5]

    expected = sum(
        weight * gaussian_kernel(x, y, sigma=sigma)
        for weight, sigma in zip(weights, sigmas)
    )
    actual = multi_scale_gaussian_kernel(x, y, sigmas=sigmas, weights=weights)
    assert np.isclose(actual, expected)
