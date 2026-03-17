"""Qiskit-based gradient estimators for cross-validation.

Supports:
  - statevector_expectation: exact ⟨Z_a⟩ via statevector simulator
  - shot_based_expectation: shot-based ⟨Z_a⟩ via Aer
  - parameter_shift_gradient: exact gradient via parameter-shift rule
"""

from __future__ import annotations

import numpy as np


def statevector_expectation(
    qc,
    a: np.ndarray,
    theta: np.ndarray,
) -> float:
    """Compute ⟨Z_a⟩ exactly using Qiskit statevector simulator.

    Args:
        qc: QuantumCircuit (without measurement, parameterized)
        a: Observable bitmask, shape (n,)
        theta: Parameter values, shape (m,)

    Returns:
        Exact expectation value
    """
    try:
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    n = len(a)
    pauli_str = "".join("Z" if a[n - 1 - i] else "I" for i in range(n))
    observable = SparsePauliOp(pauli_str)

    estimator = StatevectorEstimator()
    param_values = {qc.parameters[i]: float(theta[i]) for i in range(len(theta))}
    bound = qc.assign_parameters(param_values)
    result = estimator.run([(bound, observable)]).result()
    return float(result[0].data.evs)


def shot_based_expectation(
    qc,
    a: np.ndarray,
    theta: np.ndarray,
    n_shots: int = 10000,
) -> float:
    """Estimate ⟨Z_a⟩ from shot-based measurement.

    Args:
        qc: QuantumCircuit (with measure_all)
        a: Observable bitmask, shape (n,)
        theta: Parameter values
        n_shots: Number of measurement shots

    Returns:
        Shot-based expectation estimate
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError("Qiskit Aer required: pip install qiskit-aer")

    backend = AerSimulator()
    bound = qc.assign_parameters(
        {qc.parameters[i]: float(theta[i]) for i in range(len(theta))}
    )
    job = backend.run(bound, shots=n_shots)
    counts = job.result().get_counts()
    n = len(a)

    total = 0.0
    for bitstr, count in counts.items():
        bits = np.array([int(b) for b in bitstr[::-1]], dtype=np.uint8)
        parity = int((bits @ a) % 2)
        total += count * (1 - 2 * parity)

    return total / n_shots


def parameter_shift_gradient(
    qc,
    a: np.ndarray,
    theta: np.ndarray,
    param_idx: int,
    use_shots: bool = False,
    n_shots: int = 10000,
) -> float:
    """Compute ∂_{θ_i} ⟨Z_a⟩ via parameter-shift rule.

    ∂_{θ_i} ⟨Z_a⟩ = (1/2) [⟨Z_a⟩(θ + π/2 e_i) - ⟨Z_a⟩(θ - π/2 e_i)]
    """
    estimate_fn = shot_based_expectation if use_shots else statevector_expectation
    kwargs = {"n_shots": n_shots} if use_shots else {}

    theta_plus = theta.copy()
    theta_minus = theta.copy()
    theta_plus[param_idx] += np.pi / 2
    theta_minus[param_idx] -= np.pi / 2

    f_plus = estimate_fn(qc, a, theta_plus, **kwargs)
    f_minus = estimate_fn(qc, a, theta_minus, **kwargs)
    return 0.5 * (f_plus - f_minus)
