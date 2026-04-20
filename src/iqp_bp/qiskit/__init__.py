"""Qiskit circuit builders and estimators for IQP validation.

This package is the Qiskit-backed half of the project's cross-validation
layer. The main Python stack (``iqp_bp.iqp``) computes IQP expectations and
MMD^2 gradients in closed form using the Fourier decomposition. That's fast
and exact — but if the closed-form math has a bug, no internal test will
find it. We therefore re-compute the same quantities from first principles
with Qiskit (construct the circuit, simulate it, measure observables) and
compare. An agreement to within shot noise confirms both implementations.

Sub-modules:
    circuit_builder  — compile an IQP hypergraph matrix ``G`` into a Qiskit
        ``QuantumCircuit``: outer Hadamard layers + one diagonal phase
        gadget ``exp(i theta_j Z^{g_j})`` per generator row.
    estimators       — compute ``<Z_a>`` via either statevector (exact) or
        shot-based (Aer) simulation, and parameter-shift gradients thereof.
    mmd              — wrap the above into MMD^2 and grad(MMD^2) estimators
        that mirror the API of ``iqp_bp.mmd`` but source ``<Z_a>_q`` from
        Qiskit.
    noise            — Aer noise-model constructors (depolarizing, readout,
        thermal relaxation, T1/T2, device presets) for NISQ-realistic runs.

Seeding: every stochastic step in this package accepts an integer ``seed``.
Callers should derive those seeds from ``STREAM_QISKIT`` via
``iqp_bp.rng.experiment_stream_bundle`` so Qiskit randomness participates
in the same named-stream policy as the rest of the pipeline.
"""

# Re-export the public surface so callers can do e.g.
# ``from iqp_bp.qiskit import qiskit_mmd2`` rather than reaching into the
# sub-module layout.
from iqp_bp.qiskit.circuit_builder import (
    CircuitSpec,
    TranspileMetadata,
    build_iqp_circuit_measured,
    build_iqp_circuit_unmeasured,
    transpile_circuit,
)
from iqp_bp.qiskit.estimators import (
    batch_param_shift_gradients_shots,
    batch_param_shift_gradients_statevector,
    parameter_shift_gradient,
    shot_based_expectation,
    statevector_expectation,
)
from iqp_bp.qiskit.mmd import (
    qiskit_estimate_gradient_variance,
    qiskit_grad_mmd2,
    qiskit_mmd2,
)
from iqp_bp.qiskit.noise import (
    DEVICE_PROFILES,
    amplitude_damping_noise_model,
    backend_preset_noise_model,
    combined_noise_model,
    depolarizing_noise_model,
    get_noise_model,
    noise_model_from_fake_backend,
    phase_damping_noise_model,
    readout_noise_model,
    thermal_relaxation_noise_model,
)

__all__ = [
    "build_iqp_circuit_unmeasured",
    "build_iqp_circuit_measured",
    "transpile_circuit",
    "CircuitSpec",
    "TranspileMetadata",
    "statevector_expectation",
    "shot_based_expectation",
    "parameter_shift_gradient",
    "batch_param_shift_gradients_statevector",
    "batch_param_shift_gradients_shots",
    "qiskit_mmd2",
    "qiskit_grad_mmd2",
    "qiskit_estimate_gradient_variance",
    "DEVICE_PROFILES",
    "depolarizing_noise_model",
    "readout_noise_model",
    "combined_noise_model",
    "amplitude_damping_noise_model",
    "phase_damping_noise_model",
    "thermal_relaxation_noise_model",
    "backend_preset_noise_model",
    "noise_model_from_fake_backend",
    "get_noise_model",
]
