"""Parseval-MC scaled-second-moment estimator and low-k Pauli-inversion marginal mismatch.

This module implements Monte Carlo estimators built on top of
``iqpopt.IqpSimulator.op_expval``. Task 1 ships
:func:`pauli_ac_estimator`; :func:`pauli_marginal_mismatch` is added in
Task 2.

1. :func:`pauli_ac_estimator` -- estimates the scaled second moment
   ``2**n * sum_x q(x)**2`` via Parseval Monte Carlo over random Pauli-Z
   supports. The identity Pauli's contribution is added analytically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from iqpopt import IqpSimulator

from iqp_bp.iqp.model import IQPModel  # noqa: F401  (reserved for Task 2 FWHT inversion)


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
