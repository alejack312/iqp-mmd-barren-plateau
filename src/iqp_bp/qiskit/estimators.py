"""Qiskit-based gradient estimators for cross-validation.

Supports:
  - statevector_expectation: exact <Z_a> via statevector simulator
  - shot_based_expectation: shot-based <Z_a> via Aer
  - parameter_shift_gradient: exact gradient via parameter-shift rule

Every function here ultimately computes the same quantity the closed-form
path in ``iqp_bp.iqp.expectation`` does, but routed through a real quantum
simulator so we can cross-check the two implementations. The shot-based
variants additionally let us study noise and variance behaviour on circuits
that the closed form cannot easily express (e.g. with Aer noise models).
"""

from __future__ import annotations

import numpy as np

from iqp_bp.rng import derive_seed

# Parameter-shift shift size specific to the IQP gate structure. The gadget
# ``exp(i theta Z^g)`` has eigenvalues ``exp(±i theta)`` (a two-term
# cosine/sine dependency), which the π/4 shift resolves exactly into
# ``f(θ+π/4) − f(θ−π/4)`` without the 1/2 prefactor of the usual π/2 rule.
_IQP_PARAMETER_SHIFT = np.pi / 4


def _parameter_sort_key(param) -> tuple[str, int]:
    """Sort ParameterVector entries numerically instead of lexicographically.

    Qiskit's ``ParameterView`` iterates in name-sorted order, so naively
    ``"theta[10]"`` would sort before ``"theta[2]"`` and bindings would be
    scrambled. We split the name into ``(prefix, numeric_index)`` and sort
    on that composite key so parameter-vector order matches theta's index.
    """
    name = getattr(param, "name", str(param))
    if "[" in name and name.endswith("]"):
        prefix, index = name[:-1].rsplit("[", 1)
        if index.isdigit():
            return prefix, int(index)
    # Non-indexed parameters fall back to lexicographic order with a
    # sentinel index of -1 so they always precede the indexed ones.
    return name, -1


def _bind_circuit_parameters(qc, theta: np.ndarray):
    """Bind a numeric theta vector to a parameterized circuit.

    Qiskit's ParameterView sorts lexicographically, so theta[10] would sort
    before theta[2]. The explicit numeric sort keeps ParameterVector order.
    """
    # Sort parameters into the same order the Python theta array uses.
    params = sorted(qc.parameters, key=_parameter_sort_key)
    # Guard against length mismatch — a clear error here saves hours of
    # silently-wrong expectations downstream.
    if len(params) != len(theta):
        raise ValueError(
            f"Expected {len(params)} parameter values for circuit binding, got {len(theta)}"
        )
    # `.assign_parameters` returns a new circuit with every symbol replaced
    # by the corresponding numeric value. Cast to Python float so Qiskit
    # never sees a numpy scalar (it sometimes chokes).
    return qc.assign_parameters({param: float(theta[i]) for i, param in enumerate(params)})


def _observable_to_pauli(a: np.ndarray) -> str:
    """Convert a binary mask into Qiskit's little-endian Pauli-string order.

    Our observables come from ``iqp_bp.mmd.kernel`` as ``a = (a_0, ..., a_{n-1})``
    with ``a_i == 1`` meaning "apply Z to qubit i". Qiskit's SparsePauliOp
    reads strings right-to-left — qubit 0 is the *last* character. Reversing
    the index lookup puts everything back in sync.
    """
    n = len(a)
    return "".join("Z" if a[n - 1 - i] else "I" for i in range(n))


def statevector_expectation(
    qc,
    a: np.ndarray,
    theta: np.ndarray,
) -> float:
    """Compute <Z_a> exactly using Qiskit statevector simulation."""
    # Delayed Qiskit import — see package __init__ docstring.
    try:
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    # Encode the binary observable mask as a Pauli string; SparsePauliOp
    # wraps it into the operator object the Estimator expects.
    observable = SparsePauliOp(_observable_to_pauli(a))
    estimator = StatevectorEstimator()
    # Bind theta to the symbolic circuit so Qiskit has a concrete thing to
    # simulate. The estimator API is (circuit, observable) pub-based in
    # recent Qiskit primitives V2.
    bound = _bind_circuit_parameters(qc, theta)
    result = estimator.run([(bound, observable)]).result()
    # ``evs`` is a numpy scalar; cast for clean JSON serialization.
    return float(result[0].data.evs)


def batch_statevector_expectations(
    qc_unmeasured,
    observables: list[np.ndarray],
    theta: np.ndarray,
) -> np.ndarray:
    """Compute multiple <Z_a> values in a single StatevectorEstimator call.

    Batching matters: the estimator's fixed overhead (statevector
    construction) is paid once instead of B times, which is an order-of-
    magnitude speedup when B ~ num_a_samples.
    """
    try:
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    # One binding; reused across every pub below.
    bound = _bind_circuit_parameters(qc_unmeasured, theta)
    # "pub" = "primitive unified block" in Qiskit's primitives API: each
    # entry is the tuple ``(circuit, observable[, parameter_values])`` the
    # estimator evaluates. We reuse the same bound circuit and vary only
    # the observable.
    pubs = [(bound, SparsePauliOp(_observable_to_pauli(a))) for a in observables]

    estimator = StatevectorEstimator()
    result = estimator.run(pubs).result()
    # Collect the expectation values in input order.
    return np.array(
        [float(result[k].data.evs) for k in range(len(observables))],
        dtype=np.float64,
    )


def shot_based_expectation(
    qc,
    a: np.ndarray,
    theta: np.ndarray,
    n_shots: int = 10000,
    seed: int | None = None,
) -> float:
    """Estimate <Z_a> from shot-based measurement.

    Wraps the batched path for API symmetry with ``statevector_expectation``.
    """
    expectations = batch_shot_expectations(
        qc_measured=qc,
        observables=[a],
        theta=theta,
        n_shots=n_shots,
        seed=seed,
        noise_model=None,
        return_counts=False,
    )
    return float(expectations[0])


def batch_shot_expectations(
    qc_measured,
    observables: list[np.ndarray],
    theta: np.ndarray,
    n_shots: int,
    seed: int | None = None,
    noise_model=None,
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Estimate multiple <Z_a> values from a single AerSimulator run.

    We only ever sample from the IQP output once (the bit-string counts);
    every observable just re-projects those counts onto a parity, so B
    observables cost the same shots as 1.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError("Qiskit Aer required: pip install qiskit-aer")

    # AerSimulator(noise_model=...) returns either an ideal or noise-
    # polluted sampler, depending on whether noise_model is passed.
    backend = AerSimulator(noise_model=noise_model)
    bound = _bind_circuit_parameters(qc_measured, theta)

    # Build the kwargs dict incrementally: Aer's `seed_simulator` cannot
    # be None, so we only include it when the caller provided a seed.
    run_kwargs: dict = {"shots": n_shots}
    if seed is not None:
        run_kwargs["seed_simulator"] = seed
    # Execute the quantum job and pull the classical register counts.
    counts = backend.run(bound, **run_kwargs).result().get_counts()

    # All observables in a batch must share the same qubit count; pull from
    # the first one.
    n = len(observables[0])
    expectations = []
    for a in observables:
        total = 0.0
        for bitstr, count in counts.items():
            # Qiskit bit-string order is big-endian (qubit n-1 is the
            # leftmost character); reverse so bits[i] corresponds to qubit i.
            bits = np.array([int(bit) for bit in bitstr[::-1]], dtype=np.uint8)
            # Parity of the dot product a . bits gives ±1 for Z_a's eigenvalue.
            parity = int((bits[:n] @ a) % 2)
            # Contribute count * (+1 for even parity, -1 for odd) to the sum.
            total += count * (1 - 2 * parity)
        # Divide by shots to turn the signed count into an expectation value.
        expectations.append(total / n_shots)

    arr = np.array(expectations, dtype=np.float64)
    # Optionally return raw counts too — useful for debugging distribution
    # shapes (e.g. is the IQP output concentrated near |0...0>?).
    return (arr, counts) if return_counts else arr


def parameter_shift_gradient(
    qc,
    a: np.ndarray,
    theta: np.ndarray,
    param_idx: int,
    use_shots: bool = False,
    n_shots: int = 10000,
    seed: int | None = None,
) -> float:
    """Compute d/dtheta_i <Z_a> via the parameter-shift rule.

    Parameter-shift rule for IQP phase gadgets:
        d/dtheta <Z_a> = f(theta + pi/4) - f(theta - pi/4)
    where f(theta) is the original expectation at parameter value theta.
    Unlike the more-common pi/2 rule, the pi/4 variant avoids the 1/2
    normalisation prefactor because the IQP gate has a pure ±1 spectrum.
    """
    # Pick the underlying expectation estimator based on the caller's mode.
    estimate_fn = shot_based_expectation if use_shots else statevector_expectation
    kwargs: dict = {"n_shots": n_shots} if use_shots else {}
    if use_shots and seed is not None:
        kwargs["seed"] = seed

    # Build the two shifted parameter vectors.
    theta_plus = theta.copy()
    theta_minus = theta.copy()
    theta_plus[param_idx] += _IQP_PARAMETER_SHIFT
    theta_minus[param_idx] -= _IQP_PARAMETER_SHIFT

    # Evaluate and subtract.
    f_plus = estimate_fn(qc, a, theta_plus, **kwargs)
    f_minus = estimate_fn(qc, a, theta_minus, **kwargs)
    return f_plus - f_minus


def batch_param_shift_gradients_statevector(
    qc_unmeasured,
    observables: list[np.ndarray] | np.ndarray,
    theta: np.ndarray,
    param_idx: int,
) -> np.ndarray:
    """Compute d/dtheta_i <Z_a> for a batch of observables via parameter-shift.

    Two batched statevector evaluations (one per shifted theta) instead of
    2B individual calls. The resulting gradient array is in input order.
    """
    theta_plus = theta.copy()
    theta_minus = theta.copy()
    theta_plus[param_idx] += _IQP_PARAMETER_SHIFT
    theta_minus[param_idx] -= _IQP_PARAMETER_SHIFT

    f_plus = batch_statevector_expectations(qc_unmeasured, list(observables), theta_plus)
    f_minus = batch_statevector_expectations(qc_unmeasured, list(observables), theta_minus)
    # Element-wise subtraction yields the per-observable gradient.
    return f_plus - f_minus


def batch_param_shift_gradients_shots(
    qc_measured,
    observables: list[np.ndarray] | np.ndarray,
    theta: np.ndarray,
    param_idx: int,
    n_shots: int,
    seed: int | None = None,
    noise_model=None,
) -> np.ndarray:
    """Shot-based parameter-shift gradients for a batch of observables.

    Uses a *different* derived seed for each shift direction so the two
    sampling paths are independent; re-using the same seed would correlate
    f_plus and f_minus and bias the gradient estimate toward zero.
    """
    theta_plus = theta.copy()
    theta_minus = theta.copy()
    theta_plus[param_idx] += _IQP_PARAMETER_SHIFT
    theta_minus[param_idx] -= _IQP_PARAMETER_SHIFT

    seed_plus = None
    seed_minus = None
    if seed is not None:
        # Derive two distinct sub-seeds so the +/- simulations don't share
        # the same RNG path. ``derive_seed`` is a stable hash; same inputs
        # always give the same output, which keeps runs reproducible.
        seed_plus = derive_seed(int(seed), "param_shift", int(param_idx), "plus")
        seed_minus = derive_seed(int(seed), "param_shift", int(param_idx), "minus")

    f_plus = batch_shot_expectations(
        qc_measured=qc_measured,
        observables=list(observables),
        theta=theta_plus,
        n_shots=n_shots,
        seed=seed_plus,
        noise_model=noise_model,
        return_counts=False,
    )
    f_minus = batch_shot_expectations(
        qc_measured=qc_measured,
        observables=list(observables),
        theta=theta_minus,
        n_shots=n_shots,
        seed=seed_minus,
        noise_model=noise_model,
        return_counts=False,
    )
    return f_plus - f_minus
