"""PennyLane integration for IQP circuit operations.

Provides functions to compute expectation values and MMD loss
using PennyLane quantum circuit simulation.
"""

import numpy as np
import pennylane as qml
from itertools import product


def penn_obs(op: np.ndarray) -> qml.operation.Observable:
    """Build a PennyLane observable from a bitstring Z-operator representation.

    Args:
        op: Bitstring where 1 = Z operator, 0 = identity on that qubit.

    Returns:
        PennyLane observable.
    """
    for i, z in enumerate(op):
        if i == 0:
            obs = qml.Z(i) if z else qml.I(i)
        else:
            if z:
                obs @= qml.Z(i)
    return obs


def _penn_op_expval_circ(iqp_circuit, params: np.ndarray, op: np.ndarray):
    """Circuit that computes <op> under the IQP circuit."""
    iqp_circuit.iqp_circuit(params)
    return qml.expval(penn_obs(op))


def penn_op_expval(iqp_circuit, params: np.ndarray, op: np.ndarray) -> float:
    """Compute the expectation value of a Z-operator under an IQP circuit.

    Args:
        iqp_circuit: IQP circuit with `.device` and `.n_qubits` attributes.
        params: Gate parameters.
        op: Bitstring operator.

    Returns:
        Expectation value (float).
    """
    dev = qml.device(iqp_circuit.device, wires=iqp_circuit.n_qubits)
    qnode = qml.QNode(_penn_op_expval_circ, dev)
    return qnode(iqp_circuit, params, op)


def _penn_x_circuit(x: np.ndarray):
    """Apply X gates to initialize qubits to bitstring state."""
    for i, b in enumerate(x):
        if b:
            qml.X(i)


def _penn_train_expval(x: np.ndarray, op: np.ndarray):
    """Circuit for training expectation value with bitstring initialization."""
    _penn_x_circuit(x)
    return qml.expval(penn_obs(op))


def _penn_train_expval_dev(iqp_circuit, training_set: np.ndarray, op: np.ndarray) -> float:
    """Compute average <op> over the training set."""
    dev = qml.device(iqp_circuit.device, wires=iqp_circuit.n_qubits)
    qnode = qml.QNode(_penn_train_expval, dev)
    tr_train = sum(qnode(x, op) for x in training_set) / len(training_set)
    return tr_train


def penn_mmd_loss(iqp_circuit, params: np.ndarray, training_set: np.ndarray, sigma: float) -> float:
    """Compute exact MMD loss between IQP circuit distribution and training data.

    Enumerates all 2^n operators — only feasible for small qubit counts.

    Args:
        iqp_circuit: IQP circuit instance.
        params: Gate parameters.
        training_set: Binary training samples.
        sigma: Kernel bandwidth parameter.

    Returns:
        MMD loss value.
    """
    loss = 0.0
    p_mmd = (1 - np.exp(-1 / 2 / sigma)) / 2
    for op in product([0, 1], repeat=iqp_circuit.n_qubits):
        op = np.array(op)
        tr_iqp = penn_op_expval(iqp_circuit, params, op)
        tr_train = _penn_train_expval_dev(iqp_circuit, training_set, op)
        loss += (
            (1 - p_mmd) ** (iqp_circuit.n_qubits - op.sum())
            * p_mmd ** op.sum()
            * (tr_iqp * tr_iqp - 2 * tr_iqp * tr_train + tr_train * tr_train)
        )
    return loss
