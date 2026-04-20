"""Metrics for comparing learned and target marginals by subset order."""

from __future__ import annotations

from math import comb
from typing import Any, Sequence

import numpy as np

from iqp_bp.distributions.marginals import (
    enumerate_subsets,
    exact_fourier_coefficient,
    exact_marginal,
    sample_fourier_coefficient,
    sample_marginal,
)


def marginal_tv(p_S: np.ndarray, q_S: np.ndarray) -> float:
    """Total variation distance between two marginal histograms."""
    p = np.asarray(p_S, dtype=np.float64)
    q = np.asarray(q_S, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"marginals must share one shape, got {p.shape} vs {q.shape}")
    return float(0.5 * np.abs(p - q).sum())


def marginal_chi_square(p_S: np.ndarray, q_S: np.ndarray, *, epsilon: float = 1e-12) -> float:
    """Chi-square style mismatch with safe regularization of zero bins."""
    p = np.asarray(p_S, dtype=np.float64)
    q = np.asarray(q_S, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"marginals must share one shape, got {p.shape} vs {q.shape}")
    denom = np.maximum(q, epsilon)
    return float(np.sum((p - q) ** 2 / denom))


def fourier_squared_error(
    p_source: np.ndarray,
    q_source: np.ndarray,
    a: np.ndarray,
) -> float:
    """Squared error of the one Walsh/Fourier coefficient indexed by ``a``."""
    p_coeff = _fourier_from_source(p_source, a)
    q_coeff = _fourier_from_source(q_source, a)
    return float((p_coeff - q_coeff) ** 2)


def summarize_by_order(
    p_full: np.ndarray,
    q_full: np.ndarray,
    n: int,
    orders: Sequence[int] | None = None,
    *,
    max_subsets_per_order: int | None = None,
    rng: np.random.Generator | None = None,
    sigma: float | None = None,
) -> dict[str, Any]:
    """Summarize marginal mismatch metrics stratified by subset order."""
    if orders is None:
        orders = tuple(range(1, n + 1))
    if rng is None:
        rng = np.random.default_rng()

    tau = None if sigma is None else float(np.tanh(1.0 / (4.0 * float(sigma) ** 2)))
    normalizer = None if tau is None else float((1.0 + tau) ** n)
    order_summaries: list[dict[str, Any]] = []

    for order in orders:
        subsets = list(
            enumerate_subsets(
                n=n,
                order=int(order),
                max_subsets=max_subsets_per_order,
                rng=rng,
            )
        )
        if not subsets:
            continue

        tv_values: list[float] = []
        chi_values: list[float] = []
        fourier_errors: list[float] = []
        for subset in subsets:
            p_marginal = _marginal_from_source(p_full, subset)
            q_marginal = _marginal_from_source(q_full, subset)
            tv_values.append(marginal_tv(p_marginal, q_marginal))
            chi_values.append(marginal_chi_square(p_marginal, q_marginal))

            mask = np.zeros(n, dtype=np.uint8)
            mask[list(subset)] = 1
            fourier_errors.append(fourier_squared_error(p_full, q_full, mask))

        tv_array = np.asarray(tv_values, dtype=np.float64)
        chi_array = np.asarray(chi_values, dtype=np.float64)
        fourier_array = np.asarray(fourier_errors, dtype=np.float64)
        total_subsets = comb(n, int(order))
        used_all_subsets = len(subsets) == total_subsets

        tau_power = None if tau is None else float(tau**int(order))
        order_weight = None
        weighted_contribution = None
        if tau is not None and normalizer is not None:
            order_weight = float(total_subsets * tau_power / normalizer)
            weighted_contribution = float(order_weight * fourier_array.mean())

        order_summaries.append(
            {
                "order": int(order),
                "subset_count": len(subsets),
                "total_subsets": total_subsets,
                "used_all_subsets": used_all_subsets,
                "mean_tv": float(tv_array.mean()),
                "median_tv": float(np.median(tv_array)),
                "max_tv": float(tv_array.max()),
                "mean_chi_square": float(chi_array.mean()),
                "median_chi_square": float(np.median(chi_array)),
                "max_chi_square": float(chi_array.max()),
                "mean_fourier_squared_error": float(fourier_array.mean()),
                "median_fourier_squared_error": float(np.median(fourier_array)),
                "max_fourier_squared_error": float(fourier_array.max()),
                "tau_power": tau_power,
                "order_weight": order_weight,
                "weighted_mmd2_contribution": weighted_contribution,
            }
        )

    return {
        "n": int(n),
        "sigma": None if sigma is None else float(sigma),
        "tau": tau,
        "orders": order_summaries,
        "weighted_mmd2_total": (
            None
            if tau is None
            else float(
                sum(
                    summary["weighted_mmd2_contribution"] or 0.0
                    for summary in order_summaries
                )
            )
        ),
    }


def _marginal_from_source(source: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    source_array = np.asarray(source)
    if source_array.ndim == 1:
        return exact_marginal(source_array, subset)
    if source_array.ndim == 2:
        return sample_marginal(source_array, subset)
    raise ValueError(f"unsupported distribution source shape {source_array.shape}")


def _fourier_from_source(source: np.ndarray, mask: np.ndarray) -> float:
    source_array = np.asarray(source)
    if source_array.ndim == 1:
        return exact_fourier_coefficient(source_array, mask)
    if source_array.ndim == 2:
        return sample_fourier_coefficient(source_array, mask)
    raise ValueError(f"unsupported distribution source shape {source_array.shape}")
