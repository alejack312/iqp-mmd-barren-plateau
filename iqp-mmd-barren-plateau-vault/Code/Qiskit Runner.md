---
title: Qiskit Runner
tags:
  - code
  - runner
  - qiskit
  - validation
---

# `iqp_bp.experiments.run_qiskit`

The Qiskit cross-check runner compares the classical IQP expectation path against Qiskit statevector, shot-based, and noisy simulator executions.

**File:** [`src/iqp_bp/experiments/run_qiskit.py`](../src/iqp_bp/experiments/run_qiskit.py)

## Current Status

> [!success] Implemented
> The runner now writes `results.jsonl`, `raw/*.json`, and optional `qasm/*.qasm` artifacts. The full `configs/experiments/qiskit_validation.yaml` sweep remains the larger Q3 target; the closing report uses the smaller durable smoke config `configs/experiments/qiskit_validation_report_smoke.yaml`.

Report-smoke artifact:

- Config: `configs/experiments/qiskit_validation_report_smoke.yaml`
- Results: `results/qiskit_validation_report_smoke/results.jsonl`
- Summary: `results/final_report/qiskit_summary.json`

## Behavior

Per the [[SMART Spec]], the runner compares:

1. **Classical estimator** - the [[IQP Expectation|`iqp_expectation`]] path.
2. **Qiskit statevector** - exact noise-free reference.
3. **Qiskit shot-based simulation** - finite-sample estimates with configured `n_shots`.
4. **Qiskit noise models** - Aer depolarizing, readout, combined, damping, thermal, or backend presets where configured.

For each regime, the runner records expectation errors, MMD values under each backend, circuit depth/size, raw expectation payloads, and optional QASM.

## Config

```yaml
qiskit:
  backend: aer_simulator
  n_shots: [1000, 10000, 100000]
  max_n: 20
  noise:
    enabled: true
    model: combined
    error_rate: [0.0, 0.001, 0.005, 0.01]
```

The small report config is:

```bash
python -m iqp_bp.cli run-qiskit configs/experiments/qiskit_validation_report_smoke.yaml
```

## Role in the Overall Study

Answers Research [[Research Questions|Q3]]:

> Even if analytic gradients do not vanish exponentially, does finite-shot estimation or noise induce effective plateaus? Is classical trainability preserved under realistic execution constraints?

The report smoke run supports the implementation bridge, not the full large-n hardware conclusion. It shows statevector expectation agreement at machine precision and visible finite-shot/noise error at small n.

## Supporting Modules

- [[Qiskit Circuit Builder]] - builds a Qiskit `QuantumCircuit` from `G`
- `iqp_bp.qiskit.estimators` - `statevector_expectation`, `shot_based_expectation`
- `iqp_bp.qiskit.noise` - Aer noise model builders

## Related

- [[Qiskit Circuit Builder]]
- [[Research Questions]] - Q3
- [[SMART Spec]]
- [[Final Findings - IQP MMD Barren Plateaus]]
