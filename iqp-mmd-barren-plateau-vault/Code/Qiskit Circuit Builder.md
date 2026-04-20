---
title: Qiskit Circuit Builder
tags:
  - code
  - iqp_bp
  - qiskit
---

# `iqp_bp.qiskit.circuit_builder`

Translates a generator matrix `G` into a parameterized Qiskit `QuantumCircuit`. This is the bridge from the binary-arithmetic-only classical pipeline to concrete gate-level circuits that can run on a simulator or hardware.

**File:** [`src/iqp_bp/qiskit/circuit_builder.py`](../src/iqp_bp/qiskit/circuit_builder.py)

## Public API

```python
def build_iqp_circuit(G, theta=None, parameterized=True) -> QuantumCircuit
```

- `G` — shape `(m, n)` uint8 generator matrix
- `theta` — optional parameter values; required if `parameterized=False`
- `parameterized` — if True, uses Qiskit `ParameterVector` (for parameter-shift gradient)

## Gate Decomposition

Each IQP generator $\exp(i \theta_j X^{g_j})$ is built via standard basis-change:

1. **H on support** — maps $X \to Z$ on the qubits in the generator's support
2. **CNOT ladder** — accumulates parity onto `support[-1]`
3. **RZ(2θ)** — the actual phase rotation on the parity qubit
4. **Uncompute CNOT ladder** — back to the support qubits
5. **H on support** — back to the X basis

The factor of 2 in `RZ(2θ)` matches the factor of 2 in the classical [[IQP Expectation|phase formula]].

## Circuit Structure

```python
qc = QuantumCircuit(n)
qc.h(range(n))  # initial Hadamard layer
for j in range(m):
    support = np.where(G[j] == 1)[0]
    _add_iqp_gate(qc, support, params[j], parameterized)
qc.h(range(n))  # final Hadamard layer
qc.measure_all()
```

Matches the mathematical structure of $H^{\otimes n} D_\theta H^{\otimes n}$ (see [[IQP Circuits]]).

## Single-Qubit Optimization

For weight-1 generators the builder shortcuts to an `RX` gate directly:

```python
if len(support) == 1:
    qc.rx(2 * angle, support[0])
    return
```

This saves the H/CNOT/H wrapping on `product_state` family circuits.

## Open Work

From the TODO in this file:

- **D6.1** — split measured vs unmeasured builders and emit QASM + transpilation metadata so statevector, shots, and export paths use the same circuit spec.

## Related

- [[IQP Circuits]]
- [[Generator Matrix]]
- [[Qiskit Runner]]
- [[Parity Algebra]]
