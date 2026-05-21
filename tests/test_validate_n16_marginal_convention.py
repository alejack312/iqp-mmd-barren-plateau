"""n=4 toy unit test: estimator mean-TV parity with exact reference.

Guards against the MSB/LSB marginal-convention mismatch flagged in
02-RESEARCH Sec 5.1:

- ``iqp_bp.estimators.pauli.pauli_marginal_mismatch`` uses LSB-first
  indexing internally for both ``q_S`` (FWHT inversion) and ``p_S``
  (empirical histogram).
- ``scripts/investigate_iqp_mmd_ac.py::per_order_marginal_mismatch`` uses
  MSB-first indexing internally for both ``q_S`` (marginal_from_full) and
  ``p_S`` (empirical_marginal_from_samples).

TV is invariant to the bit convention as long as BOTH sides are
internally consistent. This test exercises that on a 4-qubit fixed-theta
toy: estimator mean_tv for k=2 should agree with exact mean_tv for k=2
within 3 sigma. If either side silently mixed MSB/LSB the TVs would
diverge, and this test would catch the regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from iqpopt import IqpSimulator
from iqpopt.utils import local_gates

from iqp_bp.estimators import pauli_marginal_mismatch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from investigate_iqp_mmd_ac import per_order_marginal_mismatch  # noqa: E402


def _toy_4q(seed: int = 0) -> tuple[IqpSimulator, np.ndarray]:
    """Minimal 4-qubit IqpSimulator with a fixed-seed random theta.

    Mirrors the existing pattern in tests/test_pauli_estimator.py::_toy_4q:
    spin_sym=False so sim.probs(theta) is a valid ground-truth vector.
    """
    n, max_weight = 4, 2
    gates = local_gates(n_qubits=n, max_weight=max_weight)
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.5, size=len(gates))
    sim = IqpSimulator(n_qubits=n, gates=gates, sparse=False, spin_sym=False)
    return sim, theta


def _synthetic_x_target(n: int, num_samples: int, seed: int) -> np.ndarray:
    """Synthetic target {0,1} samples at n=4. Biased so marginals are non-trivial.

    Uses Bernoulli(0.3) per qubit -- gives a single mode at 0000-ish, so TVs
    against a random-theta IQP output distribution are well above zero (sanity
    check below).
    """
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.3, size=(num_samples, n)).astype(np.uint8)


def test_marginal_convention_parity_n4_k2():
    """Estimator and exact reference must agree on mean-TV at k=2, within 3 sigma."""
    sim, theta = _toy_4q(seed=0)
    n = 4
    k = 2
    num_subsets = 6  # = C(4,2); enumerate all

    X_target = _synthetic_x_target(n=n, num_samples=256, seed=1)

    # Estimator side: LSB-first internal convention.
    est = pauli_marginal_mismatch(
        sim, theta, X_target,
        k_values=(k,),
        num_subsets=num_subsets,
        n_expval_samples=2000,
        seed=0,
    )
    est_stats = est.per_k[k]
    mean_tv_est = float(est_stats["mean_tv"])
    sigma_est = float(est_stats["sigma"])
    est_num_subsets = int(est_stats["num_subsets"])

    # Exact reference: full 2^n distribution via sim.probs(theta); MSB-first
    # internal convention in per_order_marginal_mismatch's helpers.
    q_exact = np.asarray(sim.probs(jnp.asarray(theta)), dtype=np.float64)
    assert q_exact.shape == (2 ** n,)
    assert abs(q_exact.sum() - 1.0) < 1e-6

    mk_exact = per_order_marginal_mismatch(
        q_exact, X_target, n=n,
        max_subsets_per_order=num_subsets,
        seed=0,
    )
    mean_tv_exact = float(mk_exact[k]["mean_tv"])
    exact_num_subsets = int(mk_exact[k]["num_subsets"])

    # Sanity: both enumerated all C(4,2)=6 subsets at k=2.
    assert est_num_subsets == 6, est_num_subsets
    assert exact_num_subsets == 6, exact_num_subsets

    # Sanity: neither side produced a zero (would be suspicious for
    # Bernoulli(0.3) empirical vs random-theta IQP).
    assert mean_tv_est > 0.0, f"estimator mean_tv was zero: {mean_tv_est}"
    assert mean_tv_exact > 0.0, f"exact mean_tv was zero: {mean_tv_exact}"

    # Core parity assertion: estimator agrees with exact reference within 3 sigma.
    # If either side silently mixed the bit convention, the TVs would diverge
    # by O(1) rather than O(sigma) and this would fail loudly.
    delta = abs(mean_tv_est - mean_tv_exact)
    tol = 3.0 * sigma_est
    assert delta <= tol, (
        f"mean_TV parity violated at n={n} k={k}: |est - exact| = "
        f"|{mean_tv_est:.6f} - {mean_tv_exact:.6f}| = {delta:.6f} "
        f"exceeds 3*sigma = 3*{sigma_est:.6f} = {tol:.6f}. "
        f"Likely cause: MSB/LSB bit-convention mismatch between "
        f"pauli_marginal_mismatch (LSB-first) and "
        f"per_order_marginal_mismatch (MSB-first)."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
