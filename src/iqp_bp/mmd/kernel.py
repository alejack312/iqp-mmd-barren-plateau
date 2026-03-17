"""Kernel functions and spectral weight samplers for MMD².

For each kernel k : {±1}^n × {±1}^n → R, we provide:
  - k(x, y): kernel evaluation
  - spectral_weights(n, **params): weight array w_k(a) for all a ∈ {0,1}^n
  - sample_a(n, num_a_samples, **params): sample Z-word indices ~ P_k

All kernels use ±1 encoding convention: x_i ∈ {±1}.
For binary x ∈ {0,1}^n, convert via x_pm = 1 - 2*x.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """k(x,y) = exp(-H(x,y) / σ²) where H is Hamming distance."""
    hamming = np.sum(x != y)
    return float(np.exp(-hamming / sigma**2))


def gaussian_spectral_weights(n: int, sigma: float) -> np.ndarray:
    """Spectral weights w_G(a; σ) ∝ tanh(1/σ²)^|a| for all a by weight.

    Returns:
        weights: shape (n+1,) where weights[w] is the weight for all a with |a|=w.
        (Weights are equal for all a with the same Hamming weight.)
    """
    tau = np.tanh(1.0 / sigma**2)
    return tau ** np.arange(n + 1)


def gaussian_sample_a(
    n: int,
    num_a_samples: int,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Z-word bitmasks a ~ P_G(a; σ).

    First samples Hamming weight w ~ P(w) ∝ C(n,w) * tau^w,
    then samples uniform a of that weight.

    Returns:
        a_samples: shape (num_a_samples, n), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()
    tau = np.tanh(1.0 / sigma**2)
    weights = np.arange(n + 1)
    log_probs = weights * np.log(tau + 1e-300) + np.array(
        [_log_binom(n, w) for w in weights]
    )
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()

    w_samples = rng.choice(n + 1, size=num_a_samples, p=probs)
    result = np.zeros((num_a_samples, n), dtype=np.uint8)
    for i, w in enumerate(w_samples):
        if w > 0:
            result[i, rng.choice(n, size=int(w), replace=False)] = 1
    return result


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def laplacian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """k(x,y) = exp(-sqrt(H(x,y)) / σ)."""
    hamming = float(np.sum(x != y))
    return float(np.exp(-np.sqrt(hamming) / sigma))


def laplacian_sample_a(
    n: int,
    num_a_samples: int,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Z-word bitmasks a ~ P_L(a; σ) via Walsh–Hadamard transform."""
    if rng is None:
        rng = np.random.default_rng()
    weights = np.array([
        _laplacian_spectral_weight(n, w, sigma) * np.exp(_log_binom(n, w))
        for w in range(n + 1)
    ])
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    w_samples = rng.choice(n + 1, size=num_a_samples, p=weights)
    result = np.zeros((num_a_samples, n), dtype=np.uint8)
    for i, w in enumerate(w_samples):
        if w > 0:
            result[i, rng.choice(n, size=int(w), replace=False)] = 1
    return result


def _laplacian_spectral_weight(n: int, w: int, sigma: float) -> float:
    """Approximate spectral weight for Laplacian kernel at Hamming weight w."""
    # Via inclusion-exclusion / Krawtchouk expansion (approximate)
    total = 0.0
    for h in range(n + 1):
        total += _krawtchouk(w, h, n) * np.exp(-np.sqrt(h) / sigma)
    return total / (2**n)


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------

def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int, constant: float = 1.0) -> float:
    """k(x,y) = (x·y/n + c)^d with ±1 encoding."""
    inner = float(np.dot(x, y)) / len(x)
    return float((inner + constant) ** degree)


def polynomial_sample_a(
    n: int,
    num_a_samples: int,
    degree: int,
    constant: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Z-word bitmasks a ~ P_P(a; d, c).

    Polynomial kernel has support only on |a| ≤ degree.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Weights are nonzero only for |a| ≤ degree
    max_w = min(degree, n)
    log_weights = np.array([
        _log_binom(n, w) + np.log(abs(_poly_coeff(w, degree, constant)) + 1e-300)
        for w in range(max_w + 1)
    ])
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    w_samples = rng.choice(max_w + 1, size=num_a_samples, p=weights)
    result = np.zeros((num_a_samples, n), dtype=np.uint8)
    for i, w in enumerate(w_samples):
        if w > 0:
            result[i, rng.choice(n, size=int(w), replace=False)] = 1
    return result


def _poly_coeff(w: int, degree: int, constant: float) -> float:
    """Coefficient of the x^w term in (x + c)^degree."""
    from math import comb
    return comb(degree, w) * (constant ** (degree - w)) / 1.0


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """k(x,y) = x·y/n with ±1 encoding."""
    return float(np.dot(x, y)) / len(x)


def linear_sample_a(
    n: int,
    num_a_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Z-word bitmasks a ~ P_lin(a) = Uniform over weight-1 bitmasks."""
    if rng is None:
        rng = np.random.default_rng()
    result = np.zeros((num_a_samples, n), dtype=np.uint8)
    qubits = rng.integers(0, n, size=num_a_samples)
    result[np.arange(num_a_samples), qubits] = 1
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

KERNEL_SAMPLERS = {
    "gaussian": gaussian_sample_a,
    "laplacian": laplacian_sample_a,
    "polynomial": polynomial_sample_a,
    "linear": linear_sample_a,
}


def sample_a(
    kernel: str,
    n: int,
    num_a_samples: int,
    rng: np.random.Generator | None = None,
    **kernel_params,
) -> np.ndarray:
    """Dispatch to the correct kernel's Z-word sampler."""
    if kernel not in KERNEL_SAMPLERS:
        raise ValueError(f"Unknown kernel {kernel!r}. Choose from {list(KERNEL_SAMPLERS)}")
    return KERNEL_SAMPLERS[kernel](n=n, num_a_samples=num_a_samples, rng=rng, **kernel_params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_binom(n: int, k: int) -> float:
    from math import lgamma
    if k < 0 or k > n:
        return -np.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _krawtchouk(k: int, x: int, n: int) -> float:
    """Krawtchouk polynomial K_k(x; n) = Σ_j (-1)^j C(x,j) C(n-x,k-j)."""
    from math import comb
    total = 0
    for j in range(k + 1):
        if j <= x and (k - j) <= (n - x):
            total += ((-1) ** j) * comb(x, j) * comb(n - x, k - j)
    return float(total)
