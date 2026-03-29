"""Qiskit Aer noise model builders for hardware-realistic simulations."""

from __future__ import annotations


def depolarizing_noise_model(error_rate: float):
    """Build depolarizing noise model for all single and two-qubit gates.

    Args:
        error_rate: Depolarizing error probability per gate

    Returns:
        NoiseModel (Qiskit Aer)
    """
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        raise ImportError("Qiskit Aer required: pip install qiskit-aer")

    noise_model = NoiseModel()
    error_1q = depolarizing_error(error_rate, 1)
    error_2q = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ["h", "rx", "rz"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
    return noise_model


def readout_noise_model(p0_given_1: float, p1_given_0: float):
    """Build readout (measurement) error noise model.

    Args:
        p0_given_1: P(measure 0 | state is 1) — bit-flip probability
        p1_given_0: P(measure 1 | state is 0)

    Returns:
        NoiseModel
    """
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError
    except ImportError:
        raise ImportError("Qiskit Aer required: pip install qiskit-aer")

    noise_model = NoiseModel()
    readout_error = ReadoutError([
        [1 - p1_given_0, p1_given_0],
        [p0_given_1, 1 - p0_given_1],
    ])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def combined_noise_model(error_rate: float, readout_rate: float | None = None):
    """Combined depolarizing + readout noise model.

    Args:
        error_rate: Gate depolarizing error rate
        readout_rate: Readout error probability (defaults to error_rate / 10)
    """
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    except ImportError:
        raise ImportError("Qiskit Aer required: pip install qiskit-aer")

    # TODO: Week 6 (D8.2) add amplitude-damping and backend-inspired presets so the
    # noise study covers the SMART comparison set beyond depolarizing/readout noise.
    if readout_rate is None:
        readout_rate = error_rate / 10.0

    noise_model = NoiseModel()
    error_1q = depolarizing_error(error_rate, 1)
    error_2q = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ["h", "rx", "rz"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    readout_error = ReadoutError([
        [1 - readout_rate, readout_rate],
        [readout_rate, 1 - readout_rate],
    ])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


NOISE_MODEL_BUILDERS = {
    "depolarizing": depolarizing_noise_model,
    "readout": readout_noise_model,
    "combined": combined_noise_model,
}
