"""Unit tests for iqp_bp.estimators.pauli on a 4-qubit toy.

Ground truth is routed entirely through ``sim.probs(jnp.asarray(theta))``,
which is valid because every fixture here pins ``spin_sym=False``. No
``IQPModel`` is constructed in these tests -- its gate representation
(``G`` as a ``(m, n)`` uint8 matrix) is incompatible with the
``list[list[list[int]]]`` layout returned by ``iqpopt.utils.local_gates``.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from iqpopt import IqpSimulator
from iqpopt.utils import local_gates

from iqp_bp.estimators import pauli_ac_estimator, pauli_marginal_mismatch
from iqp_bp.experiments import check_anti_concentration


def _toy_4q(seed: int = 0) -> tuple[IqpSimulator, np.ndarray]:
    """Build a 4-qubit toy simulator + random theta.

    Pins ``spin_sym=False`` so ``sim.probs(theta)`` is a valid
    ground-truth probability vector. No ``IQPModel`` is constructed --
    it would require a different gate representation (``G``:
    ``(m, n)`` uint8 matrix) than ``iqpopt.utils.local_gates`` returns.
    """
    n, max_weight = 4, 2
    gates = local_gates(n_qubits=n, max_weight=max_weight)
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.5, size=len(gates))
    sim = IqpSimulator(n_qubits=n, gates=gates, sparse=False, spin_sym=False)
    return sim, theta


def test_pauli_ac_estimator_matches_exact_scaled_second_moment_4q():
    sim, theta = _toy_4q()

    # Exact ground-truth probability vector via iqpopt (valid under spin_sym=False).
    q_exact = np.asarray(sim.probs(jnp.asarray(theta)))
    assert q_exact.shape == (2 ** 4,), q_exact.shape
    assert abs(q_exact.sum() - 1.0) < 1e-6, q_exact.sum()

    truth = float(check_anti_concentration(q_exact)["scaled_second_moment"])

    result = pauli_ac_estimator(
        sim, theta,
        num_pauli_samples=5000,
        n_expval_samples=5000,
        seed=666,
    )

    assert abs(result.scaled_ss_hat - truth) <= 3.0 * result.sigma, (
        f"|scaled_ss_hat - truth| = |{result.scaled_ss_hat} - {truth}| "
        f"= {abs(result.scaled_ss_hat - truth)} "
        f"exceeds 3*sigma = {3.0 * result.sigma}"
    )


def test_pauli_ac_estimator_returns_populated_metadata_and_dataclass_fields():
    sim, theta = _toy_4q()

    M = 200
    result = pauli_ac_estimator(
        sim, theta,
        num_pauli_samples=M,
        n_expval_samples=500,
        seed=666,
    )

    assert result.raw_Y_samples.shape == (M,), result.raw_Y_samples.shape
    assert result.raw_Y_samples.dtype == np.float64, result.raw_Y_samples.dtype

    assert 0 < result.effective_m <= M, result.effective_m

    expected_meta_keys = {
        "M", "n_expval_samples", "seed", "wall_time_sec",
        "n_qubits", "num_gates", "spin_sym",
    }
    assert set(result.metadata.keys()) >= expected_meta_keys, (
        f"missing meta keys: {expected_meta_keys - set(result.metadata.keys())}"
    )
    assert result.metadata["wall_time_sec"] >= 0.0, result.metadata["wall_time_sec"]

    # by_weight shape + key contract.
    assert isinstance(result.by_weight, dict)
    expected_inner = {"contribution_to_total", "contribution_sigma", "count"}
    total_count = 0
    for k, entry in result.by_weight.items():
        assert isinstance(k, int), type(k)
        assert set(entry.keys()) == expected_inner, (k, entry.keys())
        total_count += entry["count"]
    assert total_count == M, (total_count, M)

    # per-weight contributions sum to scaled_ss_hat - 1 (analytic identity term).
    contrib_sum = sum(v["contribution_to_total"] for v in result.by_weight.values())
    assert abs(contrib_sum - (result.scaled_ss_hat - 1.0)) < 1e-9, (
        contrib_sum, result.scaled_ss_hat
    )

    assert len(result.max_support) == 4, result.max_support


def test_pauli_marginal_mismatch_matches_exact_marginals_4q():
    sim, theta = _toy_4q()

    q_exact = np.asarray(sim.probs(jnp.asarray(theta)))
    q_exact = q_exact / q_exact.sum()

    rng = np.random.default_rng(0)
    idx = rng.choice(2 ** 4, size=4000, p=q_exact)
    # sim.probs() / op_expval use an MSB-first basis: row-position j of the
    # Pauli-support row (and of target_empirical) corresponds to bit (n-1-j)
    # of the computational-basis integer idx. We construct target_empirical
    # column-by-column to match that so the estimator's q_S (via FWHT) and
    # the empirical p_S agree on which qubit column j refers to.
    bit_positions = np.arange(4 - 1, -1, -1, dtype=np.int64)  # MSB-first
    target_empirical = ((idx[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)

    result = pauli_marginal_mismatch(
        sim, theta, target_empirical,
        k_values=(1, 2),
        num_subsets=6,
        n_expval_samples=5000,
        seed=666,
    )

    for k in (1, 2):
        entry = result.per_k[k]
        assert entry["mean_tv"] < 0.10, (k, entry["mean_tv"])
        assert entry["num_subsets"] == math.comb(4, k), (
            k, entry["num_subsets"], math.comb(4, k),
        )


def test_pauli_marginal_mismatch_handles_zero_variance_targets():
    sim, theta = _toy_4q()

    target_empirical = np.zeros((100, 4), dtype=np.uint8)

    result = pauli_marginal_mismatch(
        sim, theta, target_empirical,
        k_values=(1, 2),
        num_subsets=3,
        n_expval_samples=1000,
        seed=666,
    )

    for k in (1, 2):
        mean_tv = result.per_k[k]["mean_tv"]
        assert math.isfinite(mean_tv), (k, mean_tv)
