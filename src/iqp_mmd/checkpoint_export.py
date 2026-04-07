"""Helpers for exporting trained IQP artifacts into deterministic checkpoints.

The deterministic validation path in ``iqp_bp`` expects a `.npz` checkpoint
containing:

- ``G``: a binary generator matrix of shape ``(m, n)``
- ``theta``: a parameter vector of shape ``(m,)``

The original ``iqp_mmd`` training code only persisted parameter pickles. This
module adds a small, dependency-light bridge so the copied training stack can
emit the checkpoint format needed by the newer deterministic validation tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def gate_support_from_object(gate: Any, *, n_qubits: int) -> list[int]:
    """Extract qubit indices touched by one gate.

    Supported gate representations:

    - iterable of integer qubit indices, e.g. ``(0, 1)``
    - binary mask of length ``n_qubits``, e.g. ``[1, 0, 1, 0]``
    - dict-like objects with ``wires`` / ``qubits`` / ``support`` / ``indices``
    - objects exposing one of those attributes
    """
    raw_support = _support_payload(gate)
    array_support = np.asarray(raw_support)

    if array_support.ndim == 0:
        indices = [int(array_support.item())]
    elif (
        array_support.ndim == 1
        and array_support.size == n_qubits
        and np.all(np.isin(array_support, [0, 1, False, True]))
    ):
        indices = np.flatnonzero(array_support.astype(np.uint8)).tolist()
    else:
        indices = [int(value) for value in np.ravel(array_support).tolist()]

    if not indices:
        raise ValueError("Gate support cannot be empty.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Gate support contains duplicate qubit indices: {indices}.")
    if any(index < 0 or index >= n_qubits for index in indices):
        raise ValueError(
            f"Gate support indices must lie in [0, {n_qubits}); got {indices}."
        )
    return indices


def generator_matrix_from_gates(gates: Any, *, n_qubits: int) -> np.ndarray:
    """Convert an iterable of gate descriptors into a binary generator matrix."""
    rows: list[np.ndarray] = []
    for gate in list(gates):
        support = gate_support_from_object(gate, n_qubits=n_qubits)
        row = np.zeros(n_qubits, dtype=np.uint8)
        row[support] = 1
        rows.append(row)

    if not rows:
        raise ValueError("Cannot build a generator matrix from an empty gate list.")
    return np.stack(rows, axis=0)


def save_deterministic_iqp_checkpoint(
    *,
    path: str | Path,
    G: np.ndarray,
    theta: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a deterministic `.npz` checkpoint compatible with `iqp_bp`."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "G": np.asarray(G, dtype=np.uint8),
        "theta": np.asarray(theta, dtype=np.float64),
    }
    for key, value in (metadata or {}).items():
        arrays[str(key)] = np.asarray(value)

    np.savez(checkpoint_path, **arrays)
    return checkpoint_path


def _support_payload(gate: Any) -> Any:
    """Resolve the raw support payload from common gate container shapes."""
    if isinstance(gate, dict):
        for key in ("wires", "qubits", "support", "indices"):
            if key in gate:
                return gate[key]
        return gate

    for attr in ("wires", "qubits", "support", "indices"):
        if hasattr(gate, attr):
            return getattr(gate, attr)

    return gate
