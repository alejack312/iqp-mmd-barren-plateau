"""Build parameterized IQP circuits in Qiskit.

Implements the diagonal phase gadgets exp(i theta_j Z^{g_j}) between the
global Hadamard layers. This matches the classical expectation formula used in
``iqp_bp.iqp.expectation`` and is equivalent to the logical IQP X-generator
picture after conjugation by the outer ``H^n`` layers.

Shape of the circuit this module produces, for a hypergraph with ``n``
qubits and ``m`` generators:

    |0...0>  ─ H^n ─ exp(i theta_0 Z^{g_0}) ─ ... ─ exp(i theta_{m-1} Z^{g_{m-1}}) ─ H^n ─

Each ``exp(i theta_j Z^{g_j})`` is realised with a CX ladder that folds the
support of ``g_j`` onto a single qubit, an ``Rz(2 theta_j)`` rotation, and
the reversed CX ladder to restore the original basis. The factor of 2 in
``Rz(2 theta)`` comes from Qiskit's Rz convention ``exp(-i theta/2 Z)`` —
we want the phase ``exp(i theta Z)``, which requires angle ``-2 theta``,
but for a single-qubit Z the sign only shifts a global phase; we use
``+2 theta`` to keep the argument conventions aligned with the downstream
parameter-shift rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CircuitSpec:
    """Stable metadata describing the logical IQP circuit.

    Separated from the ``QuantumCircuit`` object itself so callers can
    serialize / log it without pulling Qiskit into the result JSON.
    """

    # Number of qubits (matches G.shape[1]).
    n_qubits: int
    # Number of generators, i.e. phase gadgets (matches G.shape[0]).
    n_generators: int
    # True if the circuit still carries symbolic Parameter objects; False if
    # it has been bound to concrete angles.
    is_parameterized: bool
    # OpenQASM 2 text rendering — useful for archival and cross-checking.
    qasm_str: str


@dataclass
class TranspileMetadata:
    """Small JSON-serializable subset of transpilation diagnostics.

    We only persist the numbers that matter downstream: depth (critical for
    noise simulation), size (total op count), and the sorted basis-gate
    list (to spot unexpected gate-set drift).
    """

    depth: int
    size: int
    basis_gates: list[str]


def build_iqp_circuit_unmeasured(
    G: np.ndarray,
    theta: np.ndarray | None = None,
    parameterized: bool = True,
):
    """Build an IQP circuit without measurements and return its spec.

    Unmeasured form is what the StatevectorEstimator wants — measurements
    would collapse the state and defeat the exact-expectation path.
    """
    # Delegate actual gate wiring to the shared private helper.
    qc = _build_iqp_circuit_bare(G, theta, parameterized)
    # Capture the QASM rendering once here; downstream CircuitSpec consumers
    # are happy with the string and don't need to call Qiskit again.
    qasm_str = _dump_qasm(qc)
    m, n = G.shape
    spec = CircuitSpec(
        n_qubits=n,
        n_generators=m,
        is_parameterized=parameterized,
        qasm_str=qasm_str,
    )
    return qc, spec


def build_iqp_circuit_measured(
    G: np.ndarray,
    theta: np.ndarray | None = None,
    parameterized: bool = True,
):
    """Build an IQP circuit with terminal measurements and return its spec.

    Measured form is required for shot-based ``AerSimulator`` runs: without
    measurements Aer has nothing to report counts for.
    """
    # Reuse the unmeasured builder, then append measure_all to collapse all
    # qubits into classical bits. `inplace=False` keeps the caller's copy
    # pristine in case they want the unmeasured form too.
    qc, spec = build_iqp_circuit_unmeasured(G, theta=theta, parameterized=parameterized)
    qc_measured = qc.measure_all(inplace=False)
    return qc_measured, spec


def build_iqp_circuit(
    G: np.ndarray,
    theta: np.ndarray | None = None,
    parameterized: bool = True,
):
    """Backwards-compatible alias for the measured circuit builder.

    Older callers used a single factory that returned just the circuit; this
    thin wrapper preserves that signature.
    """
    qc, _ = build_iqp_circuit_measured(G, theta=theta, parameterized=parameterized)
    return qc


def _build_iqp_circuit_bare(
    G: np.ndarray,
    theta: np.ndarray | None = None,
    parameterized: bool = True,
):
    """Build the logical IQP circuit before any measurement layer is added."""
    # Delayed import: Qiskit is heavy and optional for projects that only
    # use the closed-form classical path. Raise cleanly if it's missing.
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    # G has shape (m, n): m generators, n qubits. One row per generator.
    m, n = G.shape
    # Allocate an n-qubit register. IQP has no ancillas.
    qc = QuantumCircuit(n)

    # Opening Hadamard layer H^n on every qubit — puts the register into
    # the uniform superposition |+>^n before the diagonal phases hit it.
    qc.h(range(n))

    # Choose either symbolic ParameterVector atoms (for QASM export and
    # later binding) or concrete numeric theta values. ParameterVector is
    # required when the caller plans to re-bind with different thetas
    # (e.g. parameter-shift or variance sweeps).
    if parameterized:
        params = ParameterVector("theta", length=m)
    else:
        if theta is None:
            raise ValueError("theta must be provided when parameterized=False")
        params = theta

    # One phase gadget per generator row.
    for j in range(m):
        # Support of generator j = qubit indices where G[j] == 1.
        support = np.where(G[j] == 1)[0].tolist()
        # A weight-0 generator is the identity phase; skip it to keep the
        # transpile output minimal.
        if not support:
            continue
        # Add the CX-ladder / Rz / reversed-CX-ladder sequence in place.
        _add_iqp_gate(qc, support, params[j], parameterized=parameterized)

    # Closing Hadamard layer H^n, converting Z-basis phase accumulation
    # back into the X-basis the IQP measurement picture uses.
    qc.h(range(n))
    return qc


def _dump_qasm(qc) -> str:
    """Export QASM, binding dummy zeros when the circuit is still symbolic."""
    try:
        import qiskit.qasm2
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    # QASM 2 cannot represent free parameters — it only serializes concrete
    # angles. If we still have ParameterVector entries we temporarily bind
    # them all to 0 so we can emit *something*. The result is only used
    # for archival / structural inspection; the original ``qc`` keeps its
    # symbols intact via ``inplace=False``.
    if qc.parameters:
        dummy = qc.assign_parameters({param: 0.0 for param in qc.parameters}, inplace=False)
        return qiskit.qasm2.dumps(dummy)
    # Fully bound already — straight dump.
    return qiskit.qasm2.dumps(qc)


def transpile_circuit(qc, backend=None, optimization_level: int = 1) -> TranspileMetadata:
    """Compile a circuit and return compact metadata about the compiled form.

    ``optimization_level`` = 1 is Qiskit's default trade-off (decompose
    to basis gates + light single-qubit optimisations). Higher levels pay
    more compilation time for slightly shorter circuits.
    """
    try:
        from qiskit.compiler import transpile
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    # Run the actual transpile. No backend pins it to a generic gate set.
    transpiled = transpile(qc, backend=backend, optimization_level=optimization_level)
    # Extract only the three numbers we care about downstream — the full
    # transpiled circuit object is not JSON-friendly.
    return TranspileMetadata(
        depth=transpiled.depth(),
        size=transpiled.size(),
        # Sorted for stable diffs across runs.
        basis_gates=sorted(transpiled.count_ops().keys()),
    )


def _add_iqp_gate(qc, support: list[int], angle, parameterized: bool):
    """Add the diagonal phase gadget exp(i theta Z^g) for a given support.

    Single-qubit case reduces to a direct Rz. Multi-qubit case uses the
    standard CX-ladder trick:

        CX(q0, q1); CX(q1, q2); ... CX(q_{k-2}, q_{k-1})
        Rz(2*theta, q_{k-1})
        CX(q_{k-2}, q_{k-1}); ... CX(q1, q2); CX(q0, q1)

    The forward ladder XORs the parities of qubits q0..q_{k-2} onto
    q_{k-1}, the Rz applies phase ``exp(-i theta * parity)`` there, and
    the reversed ladder undoes the basis change.
    """
    # Trivial weight-1 case: the phase is already on a single qubit.
    if len(support) == 1:
        qc.rz(2 * angle, support[0])
        return

    # Forward CX ladder: chain all qubits in ``support`` into the last one,
    # accumulating their parity there.
    for k in range(len(support) - 1):
        qc.cx(support[k], support[k + 1])

    # Central Rz applies the phase proportional to the accumulated parity.
    # The factor 2 matches Qiskit's ``Rz(theta) = exp(-i theta/2 Z)``
    # convention so the induced operator is ``exp(-i theta Z)``.
    qc.rz(2 * angle, support[-1])

    # Reversed CX ladder: unwinds the parity accumulation so the other
    # qubits are left unchanged (up to the phase just applied).
    for k in reversed(range(len(support) - 1)):
        qc.cx(support[k], support[k + 1])
