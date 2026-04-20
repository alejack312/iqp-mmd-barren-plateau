"""Qiskit Aer noise model builders for hardware-realistic simulations.

Noise-model randomness (shot sampling, stochastic error application) is seeded
at the ``AerSimulator.run()`` call site via the ``seed_simulator`` keyword.
Callers should derive that seed from ``STREAM_QISKIT`` using
:func:`iqp_bp.rng.experiment_stream_bundle` so that all Qiskit randomness
participates in the same named-stream policy as the rest of the experiment.

This module provides a family of noise-channel constructors, each returning
a fully-populated ``qiskit_aer.noise.NoiseModel`` that can be passed to
``AerSimulator`` or to the shot-based estimators in ``estimators.py``. The
channels cover the main noise axes NISQ devices exhibit:

    * depolarizing          — uniform random Pauli error, the textbook
                              single-parameter noise knob
    * readout (measurement) — classical bit-flip at measurement time
    * amplitude damping     — energy relaxation (|1> -> |0>)
    * phase damping         — pure dephasing (no energy loss)
    * thermal relaxation    — combined amplitude + phase damping derived
                              from physical T1/T2 times and gate durations
    * backend preset        — a calibrated combination of thermal +
                              depolarizing + readout that matches a specific
                              IBM device profile (see ``DEVICE_PROFILES``)

Each of these is wired up as an "all-qubit error" — every qubit sees the
same channel. That is adequate for cross-validation sweeps; truly
per-qubit calibration would need ``noise_model_from_fake_backend`` instead.
"""

from __future__ import annotations

from typing import Any

# The three gate names below match what ``circuit_builder.py`` actually
# emits post-transpile. If we start using other single/two-qubit gates,
# their names need adding here so the noise channels get attached.
_DEFAULT_1Q_GATES = ["h", "rx", "rz"]
_DEFAULT_2Q_GATES = ["cx"]

# Approximate backend-inspired device profiles based on public IBM calibration
# ranges. Times are in nanoseconds; probabilities are dimensionless.
# ``ibm_brisbane_like``  — a recent IBM Heron-class-ish device
# ``ibm_kyiv_like``      — an IBM Eagle-class-ish device
# ``noisy_nisq``         — a deliberately-worse profile for stress tests
DEVICE_PROFILES: dict[str, dict[str, float]] = {
    "ibm_brisbane_like": {
        "t1_ns": 220_000.0,       # Energy-relaxation time
        "t2_ns": 140_000.0,       # Dephasing time (t2 <= 2*t1 required)
        "gate_time_1q_ns": 60.0,  # Duration of a 1-qubit gate
        "gate_time_2q_ns": 660.0, # Duration of a 2-qubit gate (CX-class)
        "gate_err_1q": 2.5e-4,    # Depolarizing error per 1q gate
        "gate_err_2q": 7.5e-3,    # Depolarizing error per 2q gate
        "readout_err": 1.3e-2,    # Symmetric measurement bit-flip prob
    },
    "ibm_kyiv_like": {
        "t1_ns": 280_000.0,
        "t2_ns": 110_000.0,
        "gate_time_1q_ns": 60.0,
        "gate_time_2q_ns": 540.0,
        "gate_err_1q": 3.0e-4,
        "gate_err_2q": 9.0e-3,
        "readout_err": 1.5e-2,
    },
    "noisy_nisq": {
        # Deliberately degraded: shorter coherence, longer gates, higher
        # errors. Good for seeing how a predictor behaves when noise hits.
        "t1_ns": 80_000.0,
        "t2_ns": 40_000.0,
        "gate_time_1q_ns": 100.0,
        "gate_time_2q_ns": 800.0,
        "gate_err_1q": 1.0e-3,
        "gate_err_2q": 2.0e-2,
        "readout_err": 3.0e-2,
    },
}


def _raise_qiskit_aer_import_error() -> None:
    """Uniform error message when qiskit-aer isn't installed."""
    raise ImportError("Qiskit Aer required: pip install qiskit-aer")


def _validate_probability(name: str, value: float) -> float:
    """Validator: coerce to float and require 0 <= value <= 1."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return value


def _validate_positive(name: str, value: float) -> float:
    """Validator: coerce to float and require strict positivity."""
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _independent_two_qubit_error(single_qubit_error: Any):
    """Tensor a single-qubit channel with itself to get a 2-qubit channel.

    Used when we have an Aer channel defined on one qubit (amplitude /
    phase damping, thermal relaxation) but need to apply it to a 2-qubit
    gate. ``a.expand(b)`` is Qiskit's "tensor product of channels". Here
    both operands are the same channel so the noise on the two qubits is
    independent but identically distributed.
    """
    return single_qubit_error.expand(single_qubit_error)


def depolarizing_noise_model(error_rate: float):
    """Build depolarizing noise model for all single and two-qubit gates.

    Depolarizing channel: with prob ``error_rate`` replace the state with
    the maximally-mixed one; otherwise pass through. The simplest, most
    caricatured noise model — good as a baseline, coarse for realism.
    """
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        _raise_qiskit_aer_import_error()

    error_rate = _validate_probability("error_rate", error_rate)

    noise_model = NoiseModel()
    # One channel per gate-count; same error rate used for both for
    # simplicity. Aer's ``depolarizing_error(p, n)`` returns a channel on
    # ``n`` qubits, so we get the right dimensionality for each gate.
    error_1q = depolarizing_error(error_rate, 1)
    error_2q = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, _DEFAULT_1Q_GATES)
    noise_model.add_all_qubit_quantum_error(error_2q, _DEFAULT_2Q_GATES)
    return noise_model


def readout_noise_model(
    p0_given_1: float | None = None,
    p1_given_0: float | None = None,
    *,
    error_rate: float | None = None,
):
    """Build readout (measurement) error noise model.

    ``error_rate`` is kept as a symmetric compatibility alias.

    Readout channel: classical bit-flip at measurement time. ``p0_given_1``
    is Prob(measure 0 | true state |1>), and conversely ``p1_given_0``.
    These differ slightly on real hardware (asymmetric relaxation leaks
    more in one direction); the symmetric shortcut ``error_rate`` sets
    both to the same value.
    """
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError
    except ImportError:
        _raise_qiskit_aer_import_error()

    # Symmetric alias fills in whichever asymmetric field is missing.
    if error_rate is not None:
        symmetric = _validate_probability("error_rate", error_rate)
        if p0_given_1 is None:
            p0_given_1 = symmetric
        if p1_given_0 is None:
            p1_given_0 = symmetric
    if p0_given_1 is None or p1_given_0 is None:
        raise TypeError(
            "readout_noise_model requires p0_given_1 and p1_given_0, "
            "or a symmetric error_rate"
        )

    p0_given_1 = _validate_probability("p0_given_1", p0_given_1)
    p1_given_0 = _validate_probability("p1_given_0", p1_given_0)

    noise_model = NoiseModel()
    # ReadoutError takes a 2x2 stochastic matrix:
    #   row 0 = [P(0|0), P(1|0)] — outcome distribution given true |0>
    #   row 1 = [P(0|1), P(1|1)]
    readout_error = ReadoutError(
        [
            [1.0 - p1_given_0, p1_given_0],
            [p0_given_1, 1.0 - p0_given_1],
        ]
    )
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def combined_noise_model(error_rate: float, readout_rate: float | None = None):
    """Combined depolarizing + readout noise model.

    Realistic enough to capture the main effects of a NISQ device without
    the parameter-count overhead of a full thermal model. ``readout_rate``
    defaults to 1/10 of ``error_rate`` (readout is usually weaker noise
    than gate noise on modern devices).
    """
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except ImportError:
        _raise_qiskit_aer_import_error()

    error_rate = _validate_probability("error_rate", error_rate)
    if readout_rate is None:
        readout_rate = error_rate / 10.0
    readout_rate = _validate_probability("readout_rate", readout_rate)

    noise_model = NoiseModel()
    # Depolarizing on gates.
    error_1q = depolarizing_error(error_rate, 1)
    error_2q = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, _DEFAULT_1Q_GATES)
    noise_model.add_all_qubit_quantum_error(error_2q, _DEFAULT_2Q_GATES)

    # Symmetric readout bit-flip.
    readout_error = ReadoutError(
        [
            [1.0 - readout_rate, readout_rate],
            [readout_rate, 1.0 - readout_rate],
        ]
    )
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def amplitude_damping_noise_model(
    gamma_1q: float,
    gamma_2q: float | None = None,
):
    """Build amplitude-damping noise for all gates.

    Amplitude damping = energy relaxation |1> -> |0> with probability
    ``gamma``. Asymmetric: it pushes states toward |0>. Good isolating
    knob for studying thermal-contraction effects.
    """
    try:
        from qiskit_aer.noise import NoiseModel, amplitude_damping_error
    except ImportError:
        _raise_qiskit_aer_import_error()

    gamma_1q = _validate_probability("gamma_1q", gamma_1q)
    if gamma_2q is None:
        gamma_2q = gamma_1q
    gamma_2q = _validate_probability("gamma_2q", gamma_2q)

    noise_model = NoiseModel()
    err_1q = amplitude_damping_error(gamma_1q)
    # Qiskit's amplitude_damping_error is 1-qubit; tensor with itself for
    # 2-qubit application (independent damping on both qubits).
    err_2q = _independent_two_qubit_error(amplitude_damping_error(gamma_2q))
    noise_model.add_all_qubit_quantum_error(err_1q, _DEFAULT_1Q_GATES)
    noise_model.add_all_qubit_quantum_error(err_2q, _DEFAULT_2Q_GATES)
    return noise_model


def phase_damping_noise_model(
    lam_1q: float,
    lam_2q: float | None = None,
):
    """Build pure phase-damping noise for all gates.

    Phase damping = pure dephasing with probability ``lambda``. Destroys
    off-diagonal coherences without changing populations. Keeps the |0>/|1>
    probabilities invariant; useful when you want to isolate coherence-
    sensitive effects from relaxation.
    """
    try:
        from qiskit_aer.noise import NoiseModel, phase_damping_error
    except ImportError:
        _raise_qiskit_aer_import_error()

    lam_1q = _validate_probability("lam_1q", lam_1q)
    if lam_2q is None:
        lam_2q = lam_1q
    lam_2q = _validate_probability("lam_2q", lam_2q)

    noise_model = NoiseModel()
    err_1q = phase_damping_error(lam_1q)
    err_2q = _independent_two_qubit_error(phase_damping_error(lam_2q))
    noise_model.add_all_qubit_quantum_error(err_1q, _DEFAULT_1Q_GATES)
    noise_model.add_all_qubit_quantum_error(err_2q, _DEFAULT_2Q_GATES)
    return noise_model


def thermal_relaxation_noise_model(
    t1: float | None = None,
    t2: float | None = None,
    gate_time_1q: float | None = None,
    gate_time_2q: float | None = None,
    *,
    t1_ns: float | None = None,
    t2_ns: float | None = None,
    gate_time_1q_ns: float | None = None,
    gate_time_2q_ns: float | None = None,
):
    """Build T1/T2 thermal relaxation noise with gate-time dependence.

    Thermal relaxation = combined amplitude + phase damping parameterised
    by physical times:
        T1       — amplitude-decay time constant
        T2       — dephasing time constant (must be <= 2*T1)
        gate_time — duration of the gate the channel is attached to
    The longer the gate relative to T1/T2, the larger the induced error.
    Accepts both bare (``t1``) and namespaced (``t1_ns``) keyword styles
    for compatibility with legacy configs.
    """
    try:
        from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
    except ImportError:
        _raise_qiskit_aer_import_error()

    # Accept either key style; the *_ns variants are the new canonical names.
    if t1 is None:
        t1 = t1_ns
    if t2 is None:
        t2 = t2_ns
    if gate_time_1q is None:
        gate_time_1q = gate_time_1q_ns
    if gate_time_2q is None:
        gate_time_2q = gate_time_2q_ns
    if t1 is None or t2 is None or gate_time_1q is None or gate_time_2q is None:
        raise TypeError(
            "thermal_relaxation_noise_model requires t1, t2, gate_time_1q, "
            "and gate_time_2q (or their *_ns aliases)"
        )

    t1 = _validate_positive("t1", t1)
    t2 = _validate_positive("t2", t2)
    gate_time_1q = _validate_positive("gate_time_1q", gate_time_1q)
    gate_time_2q = _validate_positive("gate_time_2q", gate_time_2q)
    # Physical constraint: dephasing cannot outpace relaxation by more
    # than a factor of 2 (else the channel is unphysical).
    if t2 > 2.0 * t1:
        raise ValueError(
            f"Thermal relaxation requires t2 <= 2*t1; got t1={t1}, t2={t2}"
        )

    noise_model = NoiseModel()
    err_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    # 2-qubit gates see the channel independently on each qubit, so we
    # tensor the single-qubit channel with itself.
    err_2q_single = thermal_relaxation_error(t1, t2, gate_time_2q)
    err_2q = _independent_two_qubit_error(err_2q_single)
    noise_model.add_all_qubit_quantum_error(err_1q, _DEFAULT_1Q_GATES)
    noise_model.add_all_qubit_quantum_error(err_2q, _DEFAULT_2Q_GATES)
    return noise_model


def backend_preset_noise_model(preset: str):
    """Build a backend-inspired thermal + depolarizing + readout model.

    Layers three physical effects in one channel:
        1. Thermal relaxation from T1/T2/gate-time (composed first).
        2. Depolarizing gate error (composed second).
        3. Symmetric readout bit-flip at measurement time.

    Uses the calibration numbers from ``DEVICE_PROFILES[preset]``.
    """
    try:
        from qiskit_aer.noise import (
            NoiseModel,
            ReadoutError,
            depolarizing_error,
            thermal_relaxation_error,
        )
    except ImportError:
        _raise_qiskit_aer_import_error()

    if preset not in DEVICE_PROFILES:
        valid = sorted(DEVICE_PROFILES)
        raise ValueError(f"Unknown preset {preset!r}. Valid: {valid}")
    profile = DEVICE_PROFILES[preset]

    t1 = _validate_positive("t1_ns", profile["t1_ns"])
    t2 = _validate_positive("t2_ns", profile["t2_ns"])
    if t2 > 2.0 * t1:
        raise ValueError(
            f"Device profile {preset!r} violates thermal relaxation constraint: "
            f"t2={t2} > 2*t1={2.0 * t1}"
        )

    # 1q gate noise: thermal first, then depolarizing. ``.compose`` applies
    # the first channel and then the second, in quantum-channel order.
    thermal_1q = thermal_relaxation_error(t1, t2, profile["gate_time_1q_ns"])
    depol_1q = depolarizing_error(profile["gate_err_1q"], 1)
    err_1q = thermal_1q.compose(depol_1q)

    # 2q gate noise: same construction, but with the thermal channel
    # replicated across both qubits via tensor product.
    thermal_2q_single = thermal_relaxation_error(t1, t2, profile["gate_time_2q_ns"])
    thermal_2q = _independent_two_qubit_error(thermal_2q_single)
    depol_2q = depolarizing_error(profile["gate_err_2q"], 2)
    err_2q = thermal_2q.compose(depol_2q)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(err_1q, _DEFAULT_1Q_GATES)
    noise_model.add_all_qubit_quantum_error(err_2q, _DEFAULT_2Q_GATES)

    # Readout layer attached at measurement time, using the symmetric
    # profile error rate.
    readout_rate = _validate_probability("readout_err", profile["readout_err"])
    readout_error = ReadoutError(
        [
            [1.0 - readout_rate, readout_rate],
            [readout_rate, 1.0 - readout_rate],
        ]
    )
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def noise_model_from_fake_backend(backend_name: str):
    """Build a noise model from a qiskit_ibm_runtime fake backend.

    The "fake backends" shipped by qiskit-ibm-runtime carry a full
    calibration snapshot (per-qubit T1/T2, per-gate errors, topology).
    This is the most realistic option we have without hitting hardware.
    """
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime import fake_provider
    except ImportError:
        raise ImportError(
            "Requires qiskit-aer and qiskit-ibm-runtime. Install: "
            "pip install qiskit-aer qiskit-ibm-runtime"
        )

    # Resolve the backend class dynamically from its string name so we
    # don't have to import a zillion FakeXxx classes up top.
    backend_cls = getattr(fake_provider, backend_name, None)
    if backend_cls is None:
        raise ValueError(f"Unknown fake backend {backend_name!r}")
    # ``NoiseModel.from_backend`` walks the backend's calibration data and
    # constructs per-qubit channels automatically.
    return NoiseModel.from_backend(backend_cls())


# Public registry used by ``get_noise_model`` for string-dispatch from config
# files. Keep in sync with the function definitions above.
NOISE_MODEL_BUILDERS = {
    "depolarizing": depolarizing_noise_model,
    "readout": readout_noise_model,
    "combined": combined_noise_model,
    "amplitude_damping": amplitude_damping_noise_model,
    "phase_damping": phase_damping_noise_model,
    "thermal_relaxation": thermal_relaxation_noise_model,
    "backend_preset": backend_preset_noise_model,
}


def get_noise_model(name: str, **params):
    """Instantiate a registered noise model by name.

    Thin dispatcher so config files can say ``noise.model: "combined"``
    and have the runner look up the right builder without a hard-coded
    ``if`` chain.
    """
    if name not in NOISE_MODEL_BUILDERS:
        valid = sorted(NOISE_MODEL_BUILDERS)
        raise ValueError(f"Unknown noise model {name!r}. Valid: {valid}")
    return NOISE_MODEL_BUILDERS[name](**params)


__all__ = [
    "DEVICE_PROFILES",
    "NOISE_MODEL_BUILDERS",
    "amplitude_damping_noise_model",
    "backend_preset_noise_model",
    "combined_noise_model",
    "depolarizing_noise_model",
    "get_noise_model",
    "noise_model_from_fake_backend",
    "phase_damping_noise_model",
    "readout_noise_model",
    "thermal_relaxation_noise_model",
]
