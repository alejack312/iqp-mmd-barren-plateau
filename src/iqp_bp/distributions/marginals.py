"""Marginal-distribution and Walsh/Fourier helpers."""

from __future__ import annotations

from itertools import combinations
from math import comb, log2
from typing import Iterable, Sequence

import numpy as np


def _infer_num_qubits_from_probabilities(probabilities: np.ndarray) -> int:
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array")
    n = int(round(log2(probabilities.size)))
    if 2**n != probabilities.size:
        raise ValueError(
            "probability vector length must be a power of two; "
            f"got {probabilities.size}"
        )
    return n


def _validate_subset(subset: Sequence[int], n: int) -> tuple[int, ...]:
    normalized = tuple(int(index) for index in subset)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"subset must not repeat qubit indices; got {subset!r}")
    if any(index < 0 or index >= n for index in normalized):
        raise ValueError(f"subset indices must lie in [0, {n}); got {subset!r}")
    return normalized


def _basis_bits(n: int) -> np.ndarray:
    indices = np.arange(2**n, dtype=np.uint64)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)


def _normalize_probability_vector(probabilities: np.ndarray) -> np.ndarray:
    if np.any(probabilities < -1e-12):
        raise ValueError("probability vector cannot contain negative entries")
    normalized = np.asarray(probabilities, dtype=np.float64)
    normalized = np.where(np.abs(normalized) < 1e-15, 0.0, normalized)
    total = float(normalized.sum())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"probability vector must sum to 1, got {total:.12g}")
    return normalized


def exact_marginal(probabilities: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    """Return the exact marginal ``p(x_S)`` for one subset of qubits."""
    p = _normalize_probability_vector(np.asarray(probabilities, dtype=np.float64))
    n = _infer_num_qubits_from_probabilities(p)
    subset_tuple = _validate_subset(subset, n)
    k = len(subset_tuple)
    if k == 0:
        return np.array([1.0], dtype=np.float64)

    selected = _basis_bits(n)[:, subset_tuple]
    outcome_weights = 2 ** np.arange(k - 1, -1, -1, dtype=np.uint64)
    outcome_indices = selected.astype(np.uint64) @ outcome_weights
    marginal = np.zeros(2**k, dtype=np.float64)
    np.add.at(marginal, outcome_indices, p)
    return marginal


def sample_marginal(samples: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    """Return the empirical marginal histogram on one subset of columns."""
    sample_array = np.asarray(samples, dtype=np.uint8)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        raise ValueError("samples must be a non-empty 2D binary array")
    if not np.all((sample_array == 0) | (sample_array == 1)):
        raise ValueError("samples must contain only binary values")

    subset_tuple = _validate_subset(subset, sample_array.shape[1])
    k = len(subset_tuple)
    if k == 0:
        return np.array([1.0], dtype=np.float64)

    selected = sample_array[:, subset_tuple]
    outcome_weights = 2 ** np.arange(k - 1, -1, -1, dtype=np.uint64)
    outcome_indices = selected.astype(np.uint64) @ outcome_weights
    counts = np.bincount(outcome_indices, minlength=2**k).astype(np.float64)
    return counts / counts.sum()


def exact_fourier_coefficients(probabilities: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return exact Walsh/Fourier coefficients ``<Z_a>`` for one or more masks."""
    p = _normalize_probability_vector(np.asarray(probabilities, dtype=np.float64))
    n = _infer_num_qubits_from_probabilities(p)
    mask_array = np.asarray(masks, dtype=np.uint8)
    if mask_array.ndim == 1:
        mask_array = mask_array[None, :]
    if mask_array.ndim != 2 or mask_array.shape[1] != n:
        raise ValueError(f"masks must have shape (B, {n})")

    basis = _basis_bits(n).astype(np.uint8)
    parities = (basis.astype(np.int64) @ mask_array.T.astype(np.int64)) % 2
    signs = 1.0 - 2.0 * parities.astype(np.float64)
    return p @ signs


def exact_fourier_coefficient(probabilities: np.ndarray, a: np.ndarray) -> float:
    """Scalar wrapper around :func:`exact_fourier_coefficients`."""
    return float(exact_fourier_coefficients(probabilities, np.asarray(a, dtype=np.uint8))[0])


def sample_fourier_coefficients(samples: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return empirical Walsh/Fourier coefficients ``<Z_a>`` for one or more masks."""
    sample_array = np.asarray(samples, dtype=np.uint8)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        raise ValueError("samples must be a non-empty 2D binary array")
    if not np.all((sample_array == 0) | (sample_array == 1)):
        raise ValueError("samples must contain only binary values")

    n = sample_array.shape[1]
    mask_array = np.asarray(masks, dtype=np.uint8)
    if mask_array.ndim == 1:
        mask_array = mask_array[None, :]
    if mask_array.ndim != 2 or mask_array.shape[1] != n:
        raise ValueError(f"masks must have shape (B, {n})")

    parities = (sample_array.astype(np.int64) @ mask_array.T.astype(np.int64)) % 2
    signs = 1.0 - 2.0 * parities.astype(np.float64)
    return signs.mean(axis=0)


def sample_fourier_coefficient(samples: np.ndarray, a: np.ndarray) -> float:
    """Scalar wrapper around :func:`sample_fourier_coefficients`."""
    return float(sample_fourier_coefficients(samples, np.asarray(a, dtype=np.uint8))[0])


def _unrank_combination(n: int, k: int, rank: int) -> tuple[int, ...]:
    """Map one lexicographic rank to a ``k``-subset of ``range(n)``."""
    subset: list[int] = []
    next_value = 0
    remaining_rank = int(rank)
    remaining_slots = k
    while remaining_slots > 0:
        for candidate in range(next_value, n):
            tail_count = comb(n - candidate - 1, remaining_slots - 1)
            if remaining_rank < tail_count:
                subset.append(candidate)
                next_value = candidate + 1
                remaining_slots -= 1
                break
            remaining_rank -= tail_count
    return tuple(subset)


def enumerate_subsets(
    n: int,
    order: int,
    *,
    max_subsets: int | None = None,
    rng: np.random.Generator | None = None,
) -> Iterable[tuple[int, ...]]:
    """Enumerate or uniformly subsample subsets of one fixed order."""
    if order < 0 or order > n:
        raise ValueError(f"order must lie in [0, {n}], got {order}")
    total = comb(n, order)
    if max_subsets is None or max_subsets >= total:
        return combinations(range(n), order)

    if max_subsets <= 0:
        raise ValueError("max_subsets must be positive when provided")
    if rng is None:
        rng = np.random.default_rng()

    sampled_ranks = np.sort(rng.choice(total, size=max_subsets, replace=False))
    return [_unrank_combination(n, order, int(rank)) for rank in sampled_ranks]


def draw_samples_from_probability_vector(
    probabilities: np.ndarray,
    num_samples: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw bitstring samples from an exact probability vector."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    p = _normalize_probability_vector(np.asarray(probabilities, dtype=np.float64))
    n = _infer_num_qubits_from_probabilities(p)
    if rng is None:
        rng = np.random.default_rng()

    outcome_indices = rng.choice(len(p), size=num_samples, p=p)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((outcome_indices[:, None].astype(np.uint64) >> bit_positions[None, :]) & 1).astype(
        np.uint8
    )
