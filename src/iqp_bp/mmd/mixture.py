"""MMD² mixture decomposition.

MMD²(p, q_θ) = E_{a ~ P_k}[(⟨Z_a⟩_p - ⟨Z_a⟩_{q_θ})²]

This module handles the estimation of ⟨Z_a⟩_p from dataset samples.
"""

from __future__ import annotations

import numpy as np


def dataset_expectation(data: np.ndarray, a: np.ndarray) -> float:
    """Estimate ⟨Z_a⟩_p from dataset samples.

    For binary data x ∈ {0,1}^n: Z_a = (-1)^{a·x}

    Args:
        data: shape (N, n), dtype int/uint8 with values in {0,1}
        a: observable bitmask, shape (n,)

    Returns:
        Estimate of ⟨Z_a⟩_p = (1/N) Σ_x (-1)^{a·x}
    """
    parities = (data @ a) % 2  # shape (N,), values {0,1}
    signs = 1.0 - 2.0 * parities.astype(np.float64)
    return float(signs.mean())


def dataset_expectations_batch(data: np.ndarray, a_batch: np.ndarray) -> np.ndarray:
    """Vectorized ⟨Z_a⟩_p for a batch of observables.

    Args:
        data: shape (N, n)
        a_batch: shape (B, n)

    Returns:
        expectations: shape (B,)
    """
    # TODO: Weeks 3-4 (D4.1) add cached parity statistics and structured target-data
    # helpers for the Ising-like and binary-mixture datasets in the SMART plan.
    # Read first: pathlib.Path https://docs.python.org/3/library/pathlib.html#pathlib.Path ;
    # json https://docs.python.org/3/library/json.html
    parities = (data @ a_batch.T) % 2  # (N, B)
    signs = 1.0 - 2.0 * parities.astype(np.float64)
    return signs.mean(axis=0)  # (B,)
