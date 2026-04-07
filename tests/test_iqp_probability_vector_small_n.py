"""Tests for exact small-n IQP probability-vector extraction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iqp_bp.hypergraph.families import bounded_degree
from iqp_bp.iqp.model import IQPModel


def _random_model(n: int, m: int, seed: int) -> IQPModel:
    """Build a small random IQP model for exact-probability tests."""
    rng = np.random.default_rng(seed)
    G = bounded_degree(n=n, m=m, max_weight=2, rng=rng)
    theta = rng.uniform(-0.4, 0.4, size=m)
    return IQPModel(G=G, theta=theta)


def _z_expectation_from_probabilities(probabilities: np.ndarray, a: np.ndarray) -> float:
    """Reconstruct <Z_a> directly from the exact output probability vector."""
    n = len(a)
    basis_indices = np.arange(len(probabilities), dtype=np.uint64)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    basis_bits = ((basis_indices[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)
    parities = (basis_bits @ a) % 2
    signs = 1.0 - 2.0 * parities.astype(np.float64)
    return float(np.sum(probabilities * signs))


def _probability_vector_via_explicit_matrix(model: IQPModel) -> np.ndarray:
    """Build the exact IQP probability vector with an explicit H D H matrix circuit."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    full_hadamard = hadamard
    for _ in range(model.n - 1):
        full_hadamard = np.kron(full_hadamard, hadamard)

    diagonal_entries = []
    for z in model._basis_bits_exact(model.n):
        phase = 0.0
        for theta_j, generator in zip(model.theta, model.G, strict=False):
            parity = int((z @ generator) % 2)
            phase += theta_j * (1 - 2 * parity)
        diagonal_entries.append(np.exp(-1j * phase))

    diagonal = np.diag(diagonal_entries)
    initial_state = np.zeros(2**model.n, dtype=np.complex128)
    initial_state[0] = 1.0
    state = full_hadamard @ diagonal @ full_hadamard @ initial_state
    probabilities = np.abs(state) ** 2
    probabilities /= probabilities.sum()
    return probabilities


@pytest.mark.parametrize("n,seed", [(3, 0), (5, 1), (6, 2)])
def test_probability_vector_exact_is_normalized(n: int, seed: int):
    """The exact probability vector should be nonnegative and sum to one."""
    model = _random_model(n=n, m=n, seed=seed)

    probabilities = model.probability_vector_exact()

    assert probabilities.shape == (2**n,)
    assert np.all(probabilities >= -1e-14)
    assert math.isclose(float(np.sum(probabilities)), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_probability_vector_exact_at_zero_theta_is_delta_on_zero_state():
    """At theta=0 the circuit is H^n I H^n = I, so the output is |0...0>."""
    n = 4
    G = bounded_degree(n=n, m=n, max_weight=2, rng=np.random.default_rng(0))
    model = IQPModel(G=G, theta=np.zeros(n))

    probabilities = model.probability_vector_exact()

    expected = np.zeros(2**n, dtype=np.float64)
    expected[0] = 1.0
    assert np.allclose(probabilities, expected, atol=1e-12)


@pytest.mark.parametrize("n,seed", [(3, 10), (4, 11), (5, 12)])
def test_probability_vector_exact_matches_explicit_matrix_circuit(n: int, seed: int):
    """The fast exact path should match an explicit H D H statevector construction."""
    model = _random_model(n=n, m=n, seed=seed)
    probabilities = model.probability_vector_exact()
    explicit = _probability_vector_via_explicit_matrix(model)

    assert np.allclose(probabilities, explicit, atol=1e-12)


@pytest.mark.parametrize("n,seed", [(3, 20), (4, 21), (5, 22)])
def test_probability_vector_exact_gives_valid_z_parity_expectations(n: int, seed: int):
    """Z-parity expectations reconstructed from the probability vector stay in [-1, 1]."""
    model = _random_model(n=n, m=n, seed=seed)
    probabilities = model.probability_vector_exact()
    rng = np.random.default_rng(seed + 999)

    for _ in range(4):
        a = rng.integers(0, 2, size=n, dtype=np.uint8)
        expectation = _z_expectation_from_probabilities(probabilities, a)
        assert -1.0 - 1e-12 <= expectation <= 1.0 + 1e-12


def test_probability_vector_exact_rejects_large_n_when_capped():
    """The exact path should fail loudly when callers exceed the small-n cap."""
    model = _random_model(n=5, m=5, seed=123)

    with pytest.raises(ValueError, match="Exact probability vector infeasible"):
        model.probability_vector_exact(max_qubits=4)
