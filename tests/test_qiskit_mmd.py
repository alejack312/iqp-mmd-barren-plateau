from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from iqp_bp.mmd.kernel import spectral_weights_exact
from iqp_bp.mmd.loss import mmd2_exact_small_n
from iqp_bp.mmd.mixture import dataset_expectations_exact
from iqp_bp.qiskit import (
    build_iqp_circuit_measured,
    build_iqp_circuit_unmeasured,
    qiskit_estimate_gradient_variance,
    qiskit_grad_mmd2,
    qiskit_mmd2,
)


def _all_observables(n: int) -> np.ndarray:
    idx = np.arange(2**n, dtype=np.intp)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8)


def _make_exact_case(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    G = np.eye(n, dtype=np.uint8)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    data = rng.integers(0, 2, size=(80, n), dtype=np.uint8)
    a_samples = _all_observables(n)
    exp_p = dataset_expectations_exact(data, n)
    weights = spectral_weights_exact("gaussian", n, sigma=1.0)
    qc_unm, _ = build_iqp_circuit_unmeasured(G, parameterized=True)
    qc_meas, _ = build_iqp_circuit_measured(G, parameterized=True)
    return G, theta, data, a_samples, exp_p, weights, qc_unm, qc_meas


def test_qiskit_mmd2_statevector_matches_exact():
    G, theta, data, a_samples, exp_p, weights, qc_unm, qc_meas = _make_exact_case(n=4, seed=7)

    classical = mmd2_exact_small_n(theta, G, data, kernel="gaussian", sigma=1.0)
    quantum = qiskit_mmd2(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        theta=theta,
        G=G,
        data=data,
        kernel="gaussian",
        mode="statevector",
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )

    assert abs(quantum - classical) < 1e-10


def test_qiskit_mmd2_details_reconstruct_weighted_estimate():
    G, theta, data, a_samples, exp_p, weights, qc_unm, qc_meas = _make_exact_case(n=3, seed=9)

    details = qiskit_mmd2(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        theta=theta,
        G=G,
        data=data,
        kernel="gaussian",
        mode="statevector",
        return_details=True,
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )

    assert details["mode"] == "statevector"
    assert details["n_shots"] is None
    np.testing.assert_allclose(details["a_samples"], a_samples)
    np.testing.assert_allclose(details["exp_p"], exp_p)
    np.testing.assert_allclose(details["weights"], weights)
    assert details["mc_diagnostics"]["stderr"] == 0.0
    reconstructed = float(np.dot(weights, details["contributions"]))
    assert abs(details["mmd2"] - reconstructed) < 1e-12


def test_qiskit_grad_mmd2_statevector_matches_finite_difference():
    G, theta, data, a_samples, exp_p, weights, qc_unm, qc_meas = _make_exact_case(n=3, seed=11)
    param_idx = 1
    eps = 1e-6

    analytic = qiskit_grad_mmd2(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        theta=theta,
        G=G,
        data=data,
        param_idx=param_idx,
        kernel="gaussian",
        mode="statevector",
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )

    theta_plus = theta.copy()
    theta_minus = theta.copy()
    theta_plus[param_idx] += eps
    theta_minus[param_idx] -= eps
    f_plus = qiskit_mmd2(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        theta=theta_plus,
        G=G,
        data=data,
        kernel="gaussian",
        mode="statevector",
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )
    f_minus = qiskit_mmd2(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        theta=theta_minus,
        G=G,
        data=data,
        kernel="gaussian",
        mode="statevector",
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )
    finite_diff = (f_plus - f_minus) / (2 * eps)

    assert abs(analytic - finite_diff) < 5e-6


def test_qiskit_grad_mmd2_shots_tracks_statevector():
    G, theta, data, a_samples, exp_p, weights, qc_unm, qc_meas = _make_exact_case(n=3, seed=13)

    exact = qiskit_grad_mmd2(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        theta=theta,
        G=G,
        data=data,
        param_idx=0,
        kernel="gaussian",
        mode="statevector",
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )
    shot_est = qiskit_grad_mmd2(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        theta=theta,
        G=G,
        data=data,
        param_idx=0,
        kernel="gaussian",
        mode="shots",
        n_shots=20_000,
        seed=123,
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )

    assert abs(shot_est - exact) < 0.08


def test_qiskit_gradient_variance_matches_direct_gradients():
    G, _, data, a_samples, exp_p, weights, qc_unm, qc_meas = _make_exact_case(n=3, seed=17)
    theta_seeds = [
        np.random.default_rng(seed).uniform(-np.pi, np.pi, size=G.shape[0])
        for seed in range(5)
    ]

    result = qiskit_estimate_gradient_variance(
        qc_builder=lambda theta: (qc_unm, qc_meas),
        G=G,
        data=data,
        param_idx=0,
        theta_seeds=theta_seeds,
        kernel="gaussian",
        mode="statevector",
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        sigma=1.0,
    )

    grads = np.array(
        [
            qiskit_grad_mmd2(
                qc_unm=qc_unm,
                qc_meas=qc_meas,
                theta=theta,
                G=G,
                data=data,
                param_idx=0,
                kernel="gaussian",
                mode="statevector",
                a_samples=a_samples,
                exp_p=exp_p,
                weights=weights,
                sigma=1.0,
            )
            for theta in theta_seeds
        ],
        dtype=np.float64,
    )

    assert result["n_seeds"] == len(theta_seeds)
    np.testing.assert_allclose(result["mean"], grads.mean(), atol=1e-12)
    np.testing.assert_allclose(result["var"], grads.var(), atol=1e-12)
    np.testing.assert_allclose(result["std"], grads.std(), atol=1e-12)
    np.testing.assert_allclose(result["median"], np.median(grads), atol=1e-12)
    expected_snr = float(np.abs(grads.mean()) / grads.std()) if grads.std() > 0 else float("inf")
    np.testing.assert_allclose(result["snr"], expected_snr, atol=1e-12)
