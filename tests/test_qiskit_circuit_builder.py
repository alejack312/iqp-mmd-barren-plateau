from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("qiskit")

from iqp_bp.qiskit.circuit_builder import (
    build_iqp_circuit_measured,
    build_iqp_circuit_unmeasured,
    transpile_circuit,
)


def test_unmeasured_has_no_classical_register():
    G = np.eye(3, dtype=np.uint8)
    qc, spec = build_iqp_circuit_unmeasured(G, parameterized=True)

    assert len(qc.clbits) == 0
    assert len(qc.parameters) == 3
    assert spec.n_qubits == 3
    assert spec.is_parameterized is True


def test_measured_has_classical_bits():
    G = np.eye(3, dtype=np.uint8)
    qc, _ = build_iqp_circuit_measured(G, parameterized=True)

    assert len(qc.clbits) == 3
    assert len(qc.parameters) == 3


def test_qasm_determinism():
    G = np.eye(4, dtype=np.uint8)
    theta = np.zeros(4)

    _, s1 = build_iqp_circuit_unmeasured(G, parameterized=False, theta=theta)
    _, s2 = build_iqp_circuit_unmeasured(G, parameterized=False, theta=theta)

    assert s1.qasm_str == s2.qasm_str


def test_transpile_metadata_json_serializable():
    G = np.eye(3, dtype=np.uint8)
    qc, _ = build_iqp_circuit_unmeasured(G, parameterized=False, theta=np.zeros(3))
    meta = transpile_circuit(qc)

    json.dumps({"depth": meta.depth, "size": meta.size, "gates": meta.basis_gates})
