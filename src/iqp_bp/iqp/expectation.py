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
    sign = 1.0 - 2.0 * ((z_dot_G % 2).astype(np.float64))  # (-1)^{z·g_j}, shape (B, m)

    # Φ = 2 · Σ_j θ_j · (a·g_j mod 2) · (-1)^{z·g_j}
    weighted = theta * a_dot_g  # (m,)
    return 2.0 * (sign @ weighted)  # (B,)


def iqp_expectation(
    theta: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
    batch_size: int | None = None,
) -> tuple[float, float]:
    """Estimate ⟨Z_a⟩_{q_θ} via Monte Carlo over z ~ U({0,1}^n).

    Args:
        theta: Parameter vector, shape (m,)
        G: Generator matrix, shape (m, n)
        a: Observable bitmask, shape (n,)
        num_z_samples: Number of Monte Carlo samples B
        rng: NumPy random generator (seeded)
        batch_size: If given, process z-samples in chunks of this size so peak
            memory stays O(batch_size · n) rather than O(num_z_samples · n).
            Mean and variance are accumulated online via the parallel Welford
            algorithm; the return value is identical in structure to the
            unbatched path.

    Returns:
        (estimate, stderr): point estimate and standard error
    """
    if rng is None:
        rng = np.random.default_rng()
    if num_z_samples <= 0:
        raise ValueError("num_z_samples must be positive")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided")
    n = G.shape[1]

    if batch_size is None:
        # Single-batch path — materialize all z-samples at once.
        z = rng.integers(0, 2, size=(num_z_samples, n), dtype=np.uint8)
        phases = iqp_phase(theta, G, z, a)
        cos_vals = np.cos(phases)
        estimate = float(cos_vals.mean())
        stderr = float(cos_vals.std() / np.sqrt(num_z_samples))
        return estimate, stderr

    # Streaming batched path — accumulate mean and M2 (sum of squared deviations)
    # using Chan's parallel Welford algorithm so no full cosine array is held in
    # memory.  The update rule for combining group A (count, mean, M2) with a new
    # batch B (count_b, mean_b, M2_b) is:
    #   delta      = mean_b - mean_A
    #   new_count  = count_A + count_b
    #   new_mean   = mean_A + delta * count_b / new_count
    #   new_M2     = M2_A + M2_b + delta^2 * count_A * count_b / new_count
    count = 0
    mean = 0.0
    M2 = 0.0
    remaining = num_z_samples
    while remaining > 0:
        chunk = min(batch_size, remaining)
        z_chunk = rng.integers(0, 2, size=(chunk, n), dtype=np.uint8)
        cos_chunk = np.cos(iqp_phase(theta, G, z_chunk, a))

        count_b = chunk
        mean_b = float(cos_chunk.mean())
        M2_b = float(np.sum((cos_chunk - mean_b) ** 2))

        delta = mean_b - mean
        new_count = count + count_b
        mean = mean + delta * count_b / new_count
        M2 = M2 + M2_b + delta ** 2 * count * count_b / new_count
        count = new_count
        remaining -= chunk

    # Population variance → std → stderr  (matches the unbatched formula)
    stderr = float(np.sqrt(M2 / count ** 2))
    return float(mean), stderr


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
    # Read first: pytest parametrize https://docs.pytest.org/en/7.1.x/how-to/parametrize.html ;
    # itertools.product https://docs.python.org/3/library/itertools.html#itertools.product
    n = G.shape[1]
    if n > 20:
        raise ValueError(f"Exact computation infeasible for n={n} > 20")
    all_z = np.array(
        [[int(b) for b in format(i, f"0{n}b")] for i in range(2**n)],
        dtype=np.uint8,
    )
    phases = iqp_phase(theta, G, all_z, a)
    return float(np.cos(phases).mean())
