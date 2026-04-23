"""Parseval-MC scaled-second-moment estimator and low-k Pauli-inversion marginal mismatch.

This module implements two Monte Carlo estimators built on top of
``iqpopt.IqpSimulator.op_expval``:

1. :func:`pauli_ac_estimator` -- estimates the scaled second moment
   ``2**n * sum_x q(x)**2`` via Parseval Monte Carlo over random Pauli-Z
   supports. The identity Pauli's contribution is added analytically.

2. :func:`pauli_marginal_mismatch` -- estimates the low-order marginal
   mismatch between ``q`` and an empirical target via Pauli inversion on
   random ``k``-subsets: enumerate all ``2**k`` Pauli-Z supports on the
   subset, estimate ``<Z_a>`` for each, then invert via the unnormalised
   Walsh-Hadamard transform to recover the marginal ``q_S``.

Bit-ordering convention
-----------------------
Whenever a length-``2**k`` array is indexed by an integer ``idx``, bit
``j`` of ``idx`` (LSB-first: ``(idx >> j) & 1``) corresponds to the
support state for the ``j``-th column of the subset ``S`` (i.e. qubit
``S[j]``). Both the Pauli-support enumeration and the empirical histogram
use this same LSB-first convention, so the FWHT inversion is consistent.
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from iqpopt import IqpSimulator

from iqp_bp.iqp.model import IQPModel  # for _fwht_inplace static method


@dataclass
class ACResult:
    scaled_ss_hat: float
    sigma: float                         # plug-in standard error, = 2**n * sqrt(var(Y_m) / M)
    by_weight: dict[int, dict[str, float]]  # k -> {contribution_to_total, contribution_sigma, count}
    effective_m: int                     # count of m with |Y_m| > 1e-12
    max_squared_expval: float            # max_m Y_m
    max_support: list[int]               # bitstring `a` that produced max_squared_expval
    raw_Y_samples: np.ndarray            # shape (M,), float64, per-sample Y_m
    metadata: dict[str, Any]

    """
    by_weight[k] = {
        "contribution_to_total": float,   # (2**n) * sum(Y_m for m with weight==k) / M
                                          #   -- additive contribution of weight-k Paulis to
                                          #   scaled_ss_hat (excluding the analytic identity
                                          #   term +1). sum over k of contribution_to_total
                                          #   == scaled_ss_hat - 1.
        "contribution_sigma": float,      # plug-in SE of the above contribution:
                                          #   (2**n) * sqrt(var(Y_m[weights==k], ddof=1)
                                          #                 * count / M**2)
        "count": int,                     # number of drawn a's with hamming weight k
                                          #   (sum over k == M).
    }
    """


@dataclass
class MarginalResult:
    per_k: dict[int, dict[str, Any]]     # k -> {mean_tv, max_tv, sigma, num_subsets, per_subset_tvs}
    metadata: dict[str, Any]


def pauli_ac_estimator(
    simulator: IqpSimulator,
    theta: np.ndarray,
    num_pauli_samples: int = 2000,
    n_expval_samples: int = 2000,
    seed: int = 666,
    max_batch_ops: int | None = None,
    max_batch_samples: int | None = None,
) -> ACResult:
    """Parseval-MC estimator of the scaled second moment ``2**n * sum_x q(x)**2``.

    Uses the Parseval identity

        ``2**n * sum_x q(x)**2 = (1 / 2**n) * sum_a <Z_a>**2``,

    which we split into an analytic identity-Pauli term and a Monte Carlo
    tail over non-identity supports:

        ``scaled_ss = 1 + (2**n / M) * sum_m Y_m``

    where the leading ``1`` is the analytic identity-Pauli contribution
    ``2**n * (1/2**n)**2`` and ``Y_m = <Z_{a_m}>**2`` for ``M`` draws
    ``a_m ~ Bernoulli(1/2)**n``, rejecting ``a_m = 0`` (the identity is
    already accounted for analytically).

    Args:
        simulator: fully-constructed ``iqpopt.IqpSimulator``; the caller
            handles gates / spin_sym.
        theta: parameter vector of length ``len(simulator.gates)``; cast
            to ``jnp`` inside.
        num_pauli_samples: ``M``, the outer MC budget over Pauli supports.
        n_expval_samples: inner sample count passed to
            ``simulator.op_expval``.
        seed: root jax PRNG seed. Each ``op_expval`` call uses a split
            sub-key.
        max_batch_ops: pass-through to ``op_expval`` op-axis batching.
        max_batch_samples: pass-through to ``op_expval`` sample-axis batching.

    Returns:
        :class:`ACResult`. ``raw_Y_samples`` has shape ``(M,)``.
    """
    start_time = time.perf_counter()

    n = int(simulator.n_qubits)
    gates = simulator.gates
    assert len(theta) == len(gates), (
        f"theta has {len(theta)} entries but {len(gates)} gates "
        f"(n={n}). Hyperparameters likely mismatch."
    )

    M = int(num_pauli_samples)
    assert M >= 2, f"num_pauli_samples must be >= 2 (got {M}); plug-in variance needs at least 2 draws."

    # Draw non-identity Pauli supports a ~ Bernoulli(1/2)^n with rejection of a=0.
    rng = np.random.default_rng(seed)
    slack = max(16, M // 100)
    rows_list: list[np.ndarray] = []
    collected = 0
    # Loop with extra draws until we have at least M non-zero rows.
    while collected < M:
        candidate = rng.integers(0, 2, size=(M + slack, n), dtype=np.uint8)
        nonzero_mask = candidate.sum(axis=1) > 0
        filtered = candidate[nonzero_mask]
        rows_list.append(filtered)
        collected += filtered.shape[0]
    all_rows = np.concatenate(rows_list, axis=0)[:M]
    assert all_rows.shape == (M, n), f"expected ({M}, {n}), got {all_rows.shape}"

    ops_np = all_rows.astype(np.uint8, copy=False)
    ops = jnp.asarray(ops_np)

    theta_j = jnp.asarray(np.asarray(theta, dtype=np.float64))
    key = jax.random.PRNGKey(int(seed))

    means_j, _stderrs_j = simulator.op_expval(
        theta_j,
        ops,
        n_samples=int(n_expval_samples),
        key=key,
        max_batch_ops=max_batch_ops,
        max_batch_samples=max_batch_samples,
    )
    means = np.asarray(means_j, dtype=np.float64)

    Y_m = means ** 2  # shape (M,)

    # Overall estimate and plug-in standard error.
    scaled_ss_hat = 1.0 + (2.0 ** n) * float(Y_m.mean())
    sigma = (2.0 ** n) * float(np.sqrt(np.var(Y_m, ddof=1) / M))

    # Per-weight breakdown.
    weights = ops_np.sum(axis=1).astype(np.int64)
    by_weight: dict[int, dict[str, float]] = {}
    for k in np.unique(weights):
        mask = (weights == k)
        count_k = int(mask.sum())
        Y_k = Y_m[mask]
        contribution_to_total = (2.0 ** n) * float(Y_k.sum()) / M
        if count_k > 1:
            contribution_sigma = (2.0 ** n) * float(
                np.sqrt(np.var(Y_k, ddof=1) * count_k / (M ** 2))
            )
        else:
            contribution_sigma = 0.0
        by_weight[int(k)] = {
            "contribution_to_total": contribution_to_total,
            "contribution_sigma": contribution_sigma,
            "count": count_k,
        }

    effective_m = int((np.abs(Y_m) > 1e-12).sum())
    max_idx = int(np.argmax(Y_m))
    max_squared_expval = float(Y_m[max_idx])
    max_support = [int(b) for b in ops_np[max_idx].tolist()]

    elapsed = time.perf_counter() - start_time
    metadata = {
        "M": M,
        "n_expval_samples": int(n_expval_samples),
        "seed": int(seed),
        "wall_time_sec": float(elapsed),
        "n_qubits": n,
        "num_gates": len(gates),
        "spin_sym": bool(simulator.spin_sym),
    }

    return ACResult(
        scaled_ss_hat=scaled_ss_hat,
        sigma=sigma,
        by_weight=by_weight,
        effective_m=effective_m,
        max_squared_expval=max_squared_expval,
        max_support=max_support,
        raw_Y_samples=Y_m.astype(np.float64, copy=False),
        metadata=metadata,
    )


def pauli_marginal_mismatch(
    simulator: IqpSimulator,
    theta: np.ndarray,
    target_empirical: np.ndarray,
    k_values: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    num_subsets: int = 128,
    n_expval_samples: int = 2000,
    seed: int = 666,
    max_batch_ops: int | None = None,
    max_batch_samples: int | None = None,
) -> MarginalResult:
    """Low-k marginal mismatch via Pauli inversion on random k-subsets.

    For each random k-subset ``S`` of ``[n]``, enumerate all ``2**k``
    Pauli-Z supports ``a`` with support contained in ``S``, estimate
    ``<Z_a>`` via ``simulator.op_expval``, then invert with the
    unnormalised Walsh-Hadamard transform on the k-bit subspace to
    recover ``q_S``. Compute ``TV(q_S, p_S)`` where ``p_S`` is the
    empirical marginal of ``target_empirical`` restricted to ``S``.

    Bit convention (LSB-first): the length-``2**k`` arrays are indexed
    by an integer ``idx`` where bit ``j`` of ``idx`` corresponds to
    column ``S[j]`` of the full n-wide bitstring / Pauli support.

    Args:
        simulator: ``iqpopt.IqpSimulator``.
        theta: parameter vector (same contract as
            :func:`pauli_ac_estimator`).
        target_empirical: shape ``(num_train, n)``, values in
            ``{0, 1}``. Empirical training data; marginals are
            histogrammed on the fly.
        k_values: marginal orders to estimate. Default matches
            REQUIREMENTS.
        num_subsets: random k-subsets per order; if
            ``C(n, k) <= num_subsets``, enumerate all.
        n_expval_samples: inner sample count passed to ``op_expval``.
        seed: root jax/numpy seed.
        max_batch_ops: pass-through to ``op_expval``.
        max_batch_samples: pass-through to ``op_expval``.

    Returns:
        :class:`MarginalResult`. ``per_k[k]`` has keys ``mean_tv``,
        ``max_tv``, ``sigma``, ``num_subsets``, ``per_subset_tvs``.
    """
    start_time = time.perf_counter()

    n = int(simulator.n_qubits)
    assert len(theta) == len(simulator.gates), (
        f"theta has {len(theta)} entries but {len(simulator.gates)} gates "
        f"(n={n})."
    )

    target_empirical = np.asarray(target_empirical)
    assert target_empirical.ndim == 2 and target_empirical.shape[1] == n, (
        f"target_empirical must have shape (num_train, {n}); got "
        f"{target_empirical.shape}"
    )
    assert target_empirical.dtype.kind in ("u", "i", "b"), (
        f"target_empirical must be integer/bool dtype; got {target_empirical.dtype}"
    )
    unique_vals = np.unique(target_empirical)
    assert set(int(v) for v in unique_vals).issubset({0, 1}), (
        f"target_empirical values must be in {{0, 1}}; got {unique_vals}"
    )
    num_train = int(target_empirical.shape[0])
    assert num_train > 0, "target_empirical must have at least one row"

    rng = np.random.default_rng(seed)
    root_key = jax.random.PRNGKey(int(seed))
    theta_j = jnp.asarray(np.asarray(theta, dtype=np.float64))

    per_k: dict[int, dict[str, Any]] = {}

    # Global subset counter so fold_in keys are unique across k values.
    subset_counter = 0

    for k in k_values:
        k = int(k)
        if k < 1 or k > n:
            # Skip orders that are not meaningful for this n.
            per_k[k] = {
                "mean_tv": float("nan"),
                "max_tv": float("nan"),
                "sigma": float("nan"),
                "num_subsets": 0,
                "per_subset_tvs": [],
            }
            continue

        total_possible = math.comb(n, k)
        if total_possible <= num_subsets:
            subsets: list[tuple[int, ...]] = [
                tuple(s) for s in itertools.combinations(range(n), k)
            ]
        else:
            seen: set[tuple[int, ...]] = set()
            max_draws = 10 * num_subsets
            draws = 0
            while len(seen) < num_subsets and draws < max_draws:
                pick = tuple(sorted(int(x) for x in rng.choice(n, size=k, replace=False)))
                seen.add(pick)
                draws += 1
            subsets = list(seen)

        # Precompute the 2**k x k binary grid (LSB-first: bit j of idx -> column j).
        size = 1 << k
        grid = np.zeros((size, k), dtype=np.uint8)
        for j in range(k):
            # Row idx's column j is bit j of idx (LSB-first).
            grid[:, j] = (np.arange(size, dtype=np.int64) >> j) & 1

        per_subset_tvs: list[float] = []

        for S in subsets:
            S_list = list(S)
            # Build ops of shape (2**k, n): scatter grid columns into full n-wide.
            ops_np = np.zeros((size, n), dtype=np.uint8)
            for j, col in enumerate(S_list):
                ops_np[:, col] = grid[:, j]

            try:
                key_sub = jax.random.fold_in(root_key, subset_counter)
                subset_counter += 1
                means_j, _ = simulator.op_expval(
                    theta_j,
                    jnp.asarray(ops_np),
                    n_samples=int(n_expval_samples),
                    key=key_sub,
                    max_batch_ops=max_batch_ops,
                    max_batch_samples=max_batch_samples,
                )
                means = np.asarray(means_j, dtype=np.float64).copy()
            except Exception as err:  # noqa: BLE001
                print(
                    f"[pauli_marginal_mismatch] subset {S} failed: {err!r}",
                    file=sys.stderr,
                    flush=True,
                )
                per_subset_tvs.append(float("nan"))
                continue

            # Invert to marginal q_S via unnormalised FWHT; then divide by 2**k.
            values = means.astype(np.float64, copy=True)
            IQPModel._fwht_inplace(values)
            q_S = values / float(size)

            # Defensive clip + renormalise against MC noise.
            q_S = np.clip(q_S, 0.0, None)
            s = q_S.sum()
            if s > 0.0:
                q_S = q_S / s
            else:
                # Extreme MC failure; fall back to uniform.
                q_S = np.full(size, 1.0 / size, dtype=np.float64)

            # Empirical marginal p_S on the same LSB-first index convention.
            cols = target_empirical[:, S_list].astype(np.int64, copy=False)
            weights_lsb = (1 << np.arange(k, dtype=np.int64))  # LSB-first weights
            idx = (cols * weights_lsb[None, :]).sum(axis=1)
            p_S = np.bincount(idx, minlength=size).astype(np.float64) / float(num_train)

            tv = 0.5 * float(np.abs(q_S - p_S).sum())
            per_subset_tvs.append(tv)

        arr = np.asarray(per_subset_tvs, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            mean_tv = float("nan")
            max_tv = float("nan")
            se = float("nan")
        else:
            mean_tv = float(np.nanmean(arr))
            max_tv = float(np.nanmax(arr))
            finite = arr[np.isfinite(arr)]
            if finite.size >= 2:
                se = float(np.nanstd(arr, ddof=1) / np.sqrt(finite.size))
            else:
                se = 0.0

        per_k[k] = {
            "mean_tv": mean_tv,
            "max_tv": max_tv,
            "sigma": se,
            "num_subsets": len(subsets),
            "per_subset_tvs": [float(x) for x in per_subset_tvs],
        }

    elapsed = time.perf_counter() - start_time
    metadata = {
        "k_values": [int(k) for k in k_values],
        "num_subsets": int(num_subsets),
        "n_expval_samples": int(n_expval_samples),
        "seed": int(seed),
        "wall_time_sec": float(elapsed),
        "n_qubits": n,
        "spin_sym": bool(simulator.spin_sym),
    }

    return MarginalResult(per_k=per_k, metadata=metadata)
