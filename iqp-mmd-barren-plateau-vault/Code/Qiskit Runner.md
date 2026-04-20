---
title: Qiskit Runner
tags:
  - code
  - runner
  - qiskit
  - validation
---

# `iqp_bp.experiments.run_qiskit`

The Qiskit cross-check runner. Intended to compare the classical IQP estimator against statevector, shot-based, and noisy Qiskit executions.

**File:** [`src/iqp_bp/experiments/run_qiskit.py`](../src/iqp_bp/experiments/run_qiskit.py)

## Current Status

> [!warning] Stub
> The runner currently writes a `{"status": "stub — not yet implemented"}` record for each (family, n) pair. The actual Qiskit cross-check is open work under **D6.2 / D6.3** in the [[TODO Roadmap]].

## Intended Behavior

Per the [[SMART Spec]], the runner should compare:

1. **Classical estimator** — the [[IQP Expectation|`iqp_expectation`]] path
2. **Qiskit statevector** — exact reference (noise-free)
3. **Qiskit shot-based** — finite-sample with `n_shots ∈ {1000, 10000, 100000}`
4. **Qiskit noise models** — Aer depolarizing / amplitude damping / thermal

For each regime, compute gradient variance and record:

- Gradient SNR (signal-to-noise ratio)
- Shots needed to distinguish gradient from zero
- Depth/connectivity impact on noise

## Config

```yaml
qiskit:
  backend: statevector
  n_shots: [1000, 10000, 100000]
  max_n: 20
  noise:
    enabled: false
    model: depolarizing
    error_rate: [0.001, 0.005, 0.01]
```

## Role in the Overall Study

Answers Research [[Research Questions|Q3]]:

> Even if analytic gradients do not vanish exponentially, does finite-shot estimation or noise induce effective plateaus? Is classical trainability preserved under realistic execution constraints?

## Supporting Modules

- [[Qiskit Circuit Builder]] — builds a Qiskit `QuantumCircuit` from `G`
- `iqp_bp.qiskit.estimators` — `statevector_expectation`, `shot_based_expectation`
- `iqp_bp.qiskit.noise` — Aer noise model builders

## Related

- [[Qiskit Circuit Builder]]
- [[TODO Roadmap]] — D6.2, D6.3
- [[Research Questions]] — Q3
- [[SMART Spec]]
