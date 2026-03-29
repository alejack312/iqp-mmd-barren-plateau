"""Classical IQP expectation estimator.

Implements ⟨Z_a⟩_{q_θ} via uniform-z Monte Carlo:

    ⟨Z_a⟩_{q_θ} = E_{z ~ U({0,1}^n)} [cos(Φ(θ, z, a))]

where the phase is:

    Φ(θ, z, a) = 2 · Σ_j θ_j · (a · g_j mod 2) · (-1)^{z · g_j}

This is efficiently computable for large n (no 2^n state vector required).

Glossary:
  - generator matrix: docs/technical/glossary.md#generator-matrix
  - mask: docs/technical/glossary.md#mask
  - parity: docs/technical/glossary.md#parity
"""

from __future__ import annotations

import numpy as np


def iqp_phase(
    theta: np.ndarray,
    G: np.ndarray,
    z: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Compute phase Φ(θ, z, a) for a batch of z samples.

    Args:
        theta: Parameter vector, shape (m,)
        G: Generator matrix, shape (m, n), dtype uint8
        z: Bitstring samples, shape (B, n), dtype uint8
        a: Observable mask, shape (n,), dtype uint8

    Returns:
        phases: shape (B,), float64
    """
    # (a · g_j mod 2) for each generator j: shape (m,)
    # This is a parity overlap test; see docs/technical/glossary.md#parity.
    a_dot_g = (G @ a) % 2  # shape (m,)

    # (-1)^{z · g_j} for each (sample, generator): shape (B, m)
    z_dot_G = z @ G.T  # (B, m), values 0..n
    sign = 1 - 2 * (z_dot_G % 2)  # (-1)^{z·g_j}, shape (B, m)

    # Φ = 2 · Σ_j θ_j · (a·g_j mod 2) · (-1)^{z·g_j}
    weighted = theta * a_dot_g  # (m,)
    return 2.0 * (sign @ weighted)  # (B,)


def iqp_expectation(
    theta: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate ⟨Z_a⟩_{q_θ} via Monte Carlo over z ~ U({0,1}^n).

    Args:
        theta: Parameter vector, shape (m,)
        G: Generator matrix, shape (m, n)
        a: Observable bitmask, shape (n,)
        num_z_samples: Number of Monte Carlo samples B
        rng: NumPy random generator (seeded)

    Returns:
        (estimate, stderr): point estimate and standard error
    """
    if rng is None:
        rng = np.random.default_rng()
    # TODO: Week 1 (D1.3) add stable batching / streaming over z samples so the
    # expectation engine can scale without memory blowups while still reporting CI data.
    n = G.shape[1]
    z = rng.integers(0, 2, size=(num_z_samples, n), dtype=np.uint8)
    phases = iqp_phase(theta, G, z, a)
    cos_vals = np.cos(phases)
    estimate = float(cos_vals.mean())
    stderr = float(cos_vals.std() / np.sqrt(num_z_samples))
    return estimate, stderr


def iqp_expectation_exact(
    theta: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
) -> float:
    """Exact ⟨Z_a⟩_{q_θ} via full sum over all 2^n bitstrings.

    Only feasible for n ≤ 20. Used for correctness validation.

    Args:
        theta: shape (m,)
        G: shape (m, n)
        a: shape (n,)

    Returns:
        exact expectation value
    """
    # TODO: Week 2 (D2.1/D2.3) wire this exact path into automated MC-vs-exact
    # regression plots for n <= 12, including estimator error and runtime curves.
    n = G.shape[1]
    if n > 20:
        raise ValueError(f"Exact computation infeasible for n={n} > 20")
    all_z = np.array(
        [[int(b) for b in format(i, f"0{n}b")] for i in range(2**n)],
        dtype=np.uint8,
    )
    phases = iqp_phase(theta, G, all_z, a)
    return float(np.cos(phases).mean())
