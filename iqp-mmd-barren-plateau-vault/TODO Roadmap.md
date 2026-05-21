---
title: TODO Roadmap
tags:
  - planning
  - todos
---

# TODO Roadmap

Dependency-ordered task list derived from [[SMART Spec]] and the `TODO:` markers in `src/iqp_bp`. See [`TODOS.md`](../TODOS.md) for the full source-of-truth list.

## Legend

- `[x]` complete
- `[~]` partial
- `[ ]` open

## Foundation and Theory

- `[x] T1` Confirm the exact Gaussian spectral normalization - see [[Gaussian Convention]]
- `[x] T2` Lock and validate the Laplacian MMD2 decomposition - see [[Laplacian Kernel]]
- `[x] T3` Replace generic 2D patch sampler with exact nearest-neighbor ZZ lattice family - see [[ZZ Lattice Family]]

## Reproducibility and Core Pipeline

- `[x] P1` Config validation, persisted resolved experiment grid, and grid preview CLI - see [[Config]]
- `[x] P2` Reserved named RNG streams for circuit, data, theta, kernel, Qiskit sampling - see [[RNG]]
- `[x] P3` Stable batching/streaming in the classical IQP expectation engine - see [[IQP Expectation]]
- `[x] P4` Family and generation metadata preserved on IQP models for Qiskit provenance - see [[IQP Model]]
- `[x] P5` Per-observable contributions and confidence diagnostics in the MMD2 estimator - see [[MMD Loss Module]]

## Anti-Concentration Track (Due Apr 8, 2026) - Complete

- `[x] AC1` Write the anti-concentration technical note and lock the finite-$n$ decision rule - see [[Anti-Concentration]]
- `[x] AC2` Add exact small-$n$ IQP output-probability extraction and normalization checks - see [[IQP Model]], [[Walsh-Hadamard Transform]]
- `[x] AC3` Deterministic validation runner + unit tests - see [[Validation Runner]], [[Tests]]
- `[x] AC4` Validation runner loads `.npz` checkpoints and serializes JSON/CSV artifacts - see [[Validation Runner]], [[Checkpoint Bridge]]
- `[x] AC5` Deterministic AC pass/fail boundary tests - see [[Tests]]
- `[x] AC6` Emit AC summaries and checkpoint plots alongside scaling outputs - see [[Scaling Runner#Anti-Concentration Block]]

## Anti-Concentration Extension - Supervisor 2026-04-19

Design rationale in [[Design Decisions - AC7 to AC12]]. Implementation review, run instructions, trajectory schema, and supervisor FAQ in [[AC7 to AC12 Implementation]].

Code and execution artifacts are shipped for AC7-AC12.

- `[x] AC7` Marginal computation module, exact and sample paths - `src/iqp_bp/distributions/marginals.py`
- `[x] AC8` Marginal-mismatch metrics, stratified by order - `src/iqp_bp/distributions/marginal_metrics.py`
- `[x] AC9` Minimal MMD training loop on the existing analytic gradient - `src/iqp_bp/training/trainer.py`
- `[x] AC10` Per-step AC + marginal diagnostics on the training trajectory - `src/iqp_bp/experiments/run_training.py`
- `[x] AC11` Ghosh-Kim learned-distribution AC exact and sampled sweeps - results in `results/ac_ghosh_kim/` and [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]]
- `[x] AC12` Bandwidth sweep: marginal matching vs sigma - results in `results/bandwidth_marginal_sweep/` and [[AC12 Bandwidth Sweep Results 2026-04-21]]

## Scaling Inputs

- `[x] S1` Calibrate sparse Erdos-Renyi family to SMART bounded-degree regime - see [[Erdos-Renyi Family]]
- `[x] S2` Ising-like synthetic target and structured binary mixture - see [[Ising Dataset]], [[Binary Mixture Dataset]]
- `[x] S3` Cached parity statistics and structured target-data helpers - see [[Mixture Module]]
- `[x] S6` Expand scaling runner to sweep the full Cartesian grid of experiment axes - see [[Scaling Runner]], [[How a Scaling Run Works#The Explicit Grid]]

## Validation Layer

- `[x] V1` Narrow Hypothesis layer to SMART families - see [[Hypothesis Strategies]]
- `[x] V2` MC-vs-exact expectation validation plus analytic/autodiff/finite-difference gradient checks - see [[IQP Expectation]], [[Gradients Module]]

## Structured Search (Weeks 3-4)

- `[x] D4.1` Centralized primary four-family sweep + parameter-count policy in the runners - see [[Hypergraph Families]]
- `[x] D4.2 / D4.3` Aggregate gradient-norm proxies, heavy-tail checks, median-of-means stats - see [[Gradient Variance]]

## Qiskit Validation (Weeks 5-6)

- `[x] D6.1` Split measured vs unmeasured circuit builders; emit QASM + transpilation metadata - see [[Qiskit Circuit Builder]]
- `[x] D6.2 / D6.3` Implement classical/statevector/shots/noise cross-check; record gradient-SNR curves - see [[Qiskit Runner]]

Q2/Q3 closure: Qiskit noise presets include amplitude damping, phase damping, thermal relaxation, backend-inspired profiles, and fake-backend dispatch via `get_noise_model(...)`; Qiskit MMD2, MMD gradients, and gradient-SNR summaries are exposed through `qiskit_mmd2(...)`, `qiskit_grad_mmd2(...)`, and `qiskit_estimate_gradient_variance(...)`.

## Kernel Validation (Week 6)

- `[x] D8.1` Validate multi-scale Gaussian against exact mixture formula and wire the phase-2 grid - see [[Multi-Scale Gaussian Kernel]]

## Forge (Week 7)

- `[x] D9.2 / D9.3` Automated structural searches save machine-readable results, sidecars, and predicate agreement outputs - see [[Forge Runner]]

## Status Summary (as of 2026-05-08)

- Anti-concentration track (`AC1`-`AC6`) is complete.
- Learned-distribution infrastructure and experiments (`AC7`-`AC12`) are complete.
- Foundation, validation, scaling-summary, gradient-diagnostic, Qiskit, and Forge TODOs are mostly complete.
- Multi-scale Gaussian phase-2 validation/sweep wiring (`D8.1` / `K1` / `K2`) is complete in the working tree.

## Related

- [[SMART Spec]]
- [[Scope Lock]]
- [[Implementation Choices]]
- [[Planning MOC]]
