"""Build parameterized IQP circuits in Qiskit.

Implements exp(i θ_j X^{g_j}) for each generator g_j via:
  1. Apply H on support qubits (maps X → Z basis)
  2. Multi-qubit Z-rotation on parity: RZ(2θ_j) on target, CNOT ladder
  3. Apply H back
"""

from __future__ import annotations

import numpy as np


def build_iqp_circuit(
    G: np.ndarray,
    theta: np.ndarray | None = None,
    parameterized: bool = True,
):
    """Build a parameterized IQP circuit from generator matrix G.

    Args:
        G: Generator matrix, shape (m, n), uint8
        theta: Parameter values, shape (m,). If None, uses Qiskit ParameterVector.
        parameterized: If True, uses ParameterVector (for parameter-shift).

    Returns:
        qc: QuantumCircuit (Qiskit) with parameters bound or free
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    m, n = G.shape
    qc = QuantumCircuit(n)

    # Initial Hadamard layer
    qc.h(range(n))

    if parameterized:
        params = ParameterVector("θ", length=m)
    else:
        assert theta is not None
        params = theta

    for j in range(m):
        support = np.where(G[j] == 1)[0].tolist()
        if not support:
            continue
        _add_iqp_gate(qc, support, params[j], parameterized=parameterized)

    # Final Hadamard layer
    qc.h(range(n))
    qc.measure_all()

    return qc


def _add_iqp_gate(qc, support: list[int], angle, parameterized: bool):
    """Add exp(i θ X^g) gate for a given support.

    Implementation:
      - H on support to map X → Z
      - CNOT ladder to compute parity on support[-1]
      - RZ(2*angle) on parity qubit
      - Uncompute CNOT ladder
      - H on support
    """
    if len(support) == 1:
        qc.rx(2 * angle if not parameterized else 2 * angle, support[0])
        return

    # H gates
    for q in support:
        qc.h(q)

    # CNOT ladder: accumulate parity onto support[-1]
    for k in range(len(support) - 1):
        qc.cx(support[k], support[k + 1])

    # RZ on parity qubit
    qc.rz(2 * angle, support[-1])

    # Uncompute CNOT ladder
    for k in reversed(range(len(support) - 1)):
        qc.cx(support[k], support[k + 1])

    # H gates
    for q in support:
        qc.h(q)
