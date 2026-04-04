"""Qiskit validation experiment runner.

Compares classical IQP estimator against:
  - Qiskit statevector (exact, noise-free)
  - Qiskit shot-based simulation
  - Qiskit Aer noise models
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> None:
    """Entry point called by CLI."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.jsonl"

    # Import Qiskit lazily
    try:
        from iqp_bp.qiskit.circuit_builder import build_iqp_circuit
        from iqp_bp.qiskit.estimators import (
            statevector_expectation,
            shot_based_expectation,
        )
    except ImportError as e:
        raise ImportError("Qiskit not installed. Run: pip install qiskit qiskit-aer") from e

    families = cfg["circuit"]["family"]
    if not isinstance(families, list):
        families = [families]
    n_qubits_list = cfg["circuit"]["n_qubits"]
    qiskit_cfg = cfg.get("qiskit", {})
    max_n = qiskit_cfg.get("max_n", 20)

    # TODO: Week 5 (D6.2/D6.3) implement the actual classical/statevector/shots/noise
    # cross-check and record gradient-SNR plus shots-needed curves per regime.
    # Read first: QuantumCircuit https://quantum.cloud.ibm.com/docs/api/qiskit/2.1/qiskit.circuit.QuantumCircuit ;
    # Qiskit primitives https://quantum.cloud.ibm.com/docs/api/qiskit/dev/primitives ;
    # AerSimulator https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html ;
    # NoiseModel https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html
    with open(out_path, "w") as fout:
        for family in families:
            for n in n_qubits_list:
                if n > max_n:
                    log.warning(f"Skipping n={n} > max_n={max_n}")
                    continue

                # TODO: Week 5 (D6.2) build the circuit, compare classical vs
                # statevector vs shot-based estimators, and store the raw cross-check data.
                # Read first: qasm2 https://docs.quantum.ibm.com/api/qiskit/qasm2 ;
                # transpile https://quantum.cloud.ibm.com/docs/api/qiskit/0.39/qiskit.compiler.transpile ;
                # json https://docs.python.org/3/library/json.html
                record = {
                    "family": family,
                    "n": n,
                    "status": "stub — not yet implemented",
                }
                fout.write(json.dumps(record) + "\n")
                log.info(f"Qiskit validation: family={family} n={n} — stub")
