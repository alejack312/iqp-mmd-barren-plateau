from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("depolarizing", {"error_rate": 0.01}),
        ("readout", {"p0_given_1": 0.01, "p1_given_0": 0.02}),
        ("combined", {"error_rate": 0.01}),
        ("amplitude_damping", {"gamma_1q": 0.01}),
        ("phase_damping", {"lam_1q": 0.01}),
        (
            "thermal_relaxation",
            {
                "t1": 100_000.0,
                "t2": 50_000.0,
                "gate_time_1q": 60.0,
                "gate_time_2q": 600.0,
            },
        ),
        ("backend_preset", {"preset": "ibm_brisbane_like"}),
    ],
)
def test_get_noise_model_produces_valid_noise_model(name, params):
    from qiskit_aer.noise import NoiseModel

    from iqp_bp.qiskit.noise import get_noise_model

    noise_model = get_noise_model(name, **params)
    assert isinstance(noise_model, NoiseModel)
    assert len(noise_model.noise_instructions) > 0


def test_amplitude_damping_biases_toward_zero():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from iqp_bp.qiskit.noise import get_noise_model

    qc = QuantumCircuit(1, 1)
    qc.rx(np.pi, 0)
    qc.measure(0, 0)

    probs_zero: list[float] = []
    for gamma in [0.0, 0.2, 0.5, 0.9]:
        noise_model = get_noise_model("amplitude_damping", gamma_1q=gamma)
        counts = AerSimulator(noise_model=noise_model).run(
            qc,
            shots=4000,
            seed_simulator=42,
        ).result().get_counts()
        probs_zero.append(counts.get("0", 0) / 4000.0)

    assert all(probs_zero[i] < probs_zero[i + 1] for i in range(len(probs_zero) - 1))


def test_phase_damping_preserves_populations_and_kills_coherence():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from iqp_bp.qiskit.noise import get_noise_model

    z_basis = QuantumCircuit(1, 1)
    z_basis.h(0)
    z_basis.measure(0, 0)

    x_basis = QuantumCircuit(1, 1)
    x_basis.h(0)
    x_basis.h(0)
    x_basis.measure(0, 0)

    noise_model = get_noise_model("phase_damping", lam_1q=0.95)
    simulator = AerSimulator(noise_model=noise_model)

    z_counts = simulator.run(
        z_basis,
        shots=4000,
        seed_simulator=7,
    ).result().get_counts()
    x_counts = simulator.run(
        x_basis,
        shots=4000,
        seed_simulator=7,
    ).result().get_counts()

    prob_zero_z = z_counts.get("0", 0) / 4000.0
    prob_zero_x = x_counts.get("0", 0) / 4000.0

    assert abs(prob_zero_z - 0.5) < 0.1
    assert prob_zero_x < 0.75


def test_thermal_relaxation_rejects_t2_gt_2t1():
    from iqp_bp.qiskit.noise import get_noise_model

    with pytest.raises(ValueError, match=r"t2 <= 2\*t1"):
        get_noise_model(
            "thermal_relaxation",
            t1=10.0,
            t2=30.0,
            gate_time_1q=1.0,
            gate_time_2q=10.0,
        )


def test_backend_presets_have_distinct_characteristics():
    from iqp_bp.qiskit.noise import DEVICE_PROFILES, get_noise_model

    assert {"ibm_brisbane_like", "ibm_kyiv_like", "noisy_nisq"} <= set(DEVICE_PROFILES)
    signatures = {
        (
            profile["t1_ns"],
            profile["t2_ns"],
            profile["gate_err_1q"],
            profile["gate_err_2q"],
            profile["readout_err"],
        )
        for profile in DEVICE_PROFILES.values()
    }
    assert len(signatures) == len(DEVICE_PROFILES)

    for preset in DEVICE_PROFILES:
        noise_model = get_noise_model("backend_preset", preset=preset)
        assert "measure" in noise_model.noise_instructions


def test_unknown_noise_model_raises():
    from iqp_bp.qiskit.noise import get_noise_model

    with pytest.raises(ValueError, match="Unknown noise model"):
        get_noise_model("not_a_real_model")


def test_backend_preset_applies_readout_error():
    from iqp_bp.qiskit.noise import get_noise_model

    noise_model = get_noise_model("backend_preset", preset="ibm_brisbane_like")
    assert "measure" in noise_model.noise_instructions
