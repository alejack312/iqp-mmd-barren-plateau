from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from iqp_mmd.checkpoint_export import (
    gate_support_from_object,
    generator_matrix_from_gates,
    save_deterministic_iqp_checkpoint,
)


class _GateWithWires:
    def __init__(self, wires):
        self.wires = wires


def test_gate_support_from_index_iterable():
    support = gate_support_from_object((0, 2), n_qubits=4)
    assert support == [0, 2]


def test_gate_support_from_binary_mask():
    support = gate_support_from_object([1, 0, 1, 0], n_qubits=4)
    assert support == [0, 2]


def test_gate_support_from_object_attribute():
    support = gate_support_from_object(_GateWithWires([1, 3]), n_qubits=4)
    assert support == [1, 3]


def test_generator_matrix_from_gates_builds_binary_rows():
    G = generator_matrix_from_gates([(0, 1), {"wires": [2]}, [0, 0, 1, 1]], n_qubits=4)
    assert G.dtype == np.uint8
    assert G.shape == (3, 4)
    assert np.array_equal(
        G,
        np.asarray(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        ),
    )


def test_save_deterministic_iqp_checkpoint_writes_expected_arrays():
    output_dir = Path("tests") / "_tmp_iqp_mmd_checkpoint" / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "toy_checkpoint.npz"

    try:
        save_deterministic_iqp_checkpoint(
            path=checkpoint_path,
            G=np.eye(3, dtype=np.uint8),
            theta=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            metadata={"dataset_name": "toy", "seed": 7},
        )
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            assert np.array_equal(checkpoint["G"], np.eye(3, dtype=np.uint8))
            assert np.allclose(checkpoint["theta"], np.asarray([0.1, 0.2, 0.3]))
            assert checkpoint["dataset_name"].tolist() == "toy"
            assert checkpoint["seed"].tolist() == 7
    finally:
        for child in output_dir.glob("*"):
            child.unlink(missing_ok=True)
        output_dir.rmdir()
