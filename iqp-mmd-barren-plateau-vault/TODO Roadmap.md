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

- `[x] T1` Confirm the exact Gaussian spectral normalization — see [[Gaussian Convention]]
- `[~] T2` Keep Laplacian kernel path as explicit stub until MMD² decomposition is derived — see [[Laplacian Kernel]]
- `[x] T3` Replace generic 2D patch sampler with exact nearest-neighbor ZZ lattice family — see [[ZZ Lattice Family]]

## Reproducibility and Core Pipeline

- `[~] P1` Add config validation and persisted resolved experiment grid for reproducible reruns — see [[Config]]
- `[~] P2` Reserve named RNG streams for circuit, data, theta, kernel, Qiskit sampling — see [[RNG]]
- `[~] P3` Add stable batching/streaming to the classical IQP expectation engine — see [[IQP Expectation]]
- `[ ] P4` Preserve family and generation metadata on IQP models for Qiskit provenance — see [[IQP Model]]
- `[~] P5` Expose per-observable contributions and confidence diagnostics in the MMD² estimator — see [[MMD Loss Module]]

## Anti-Concentration Track (Due Apr 8, 2026) ✅ Complete

- `[x] AC1` Write the anti-concentration technical note and lock the finite-$n$ decision rule — see [[Anti-Concentration]]
- `[x] AC2` Add exact small-$n$ IQP output-probability extraction and normalization checks — see [[IQP Model]], [[Walsh-Hadamard Transform]]
- `[x] AC3` Deterministic validation runner + unit tests — see [[Validation Runner]], [[Tests]]
- `[x] AC4` Validation runner loads `.npz` checkpoints and serializes JSON/CSV artifacts — see [[Validation Runner]], [[Checkpoint Bridge]]
- `[x] AC5` Deterministic AC pass/fail boundary tests (uniform pass, delta fail, sample-to-exact convergence) — see [[Tests]]
- `[x] AC6` Emit AC summaries and checkpoint plots alongside scaling outputs — see [[Scaling Runner#Anti-Concentration Block]]. Landed in commit `da9db4a`.

## Scaling Inputs

- `[x] S1` Calibrate sparse Erdős–Rényi family to SMART bounded-degree regime — see [[Erdos-Renyi Family]]
- `[~] S2` Implement Ising-like synthetic target and structured binary mixture — see [[Ising Dataset]], [[Binary Mixture Dataset]]
- `[~] S3` Cached parity statistics and structured target-data helpers — see [[Mixture Module]]
- `[x] S6` Expand scaling runner to sweep the full Cartesian grid of experiment axes — see [[Scaling Runner]], [[How a Scaling Run Works#The Explicit Grid]]

## Validation Layer

- `[x] V1` Narrow Hypothesis layer to SMART families — see [[Hypothesis Strategies]]
- `[ ] V2` Analytic vs finite-difference gradient regression test on small $n$ — see [[Gradients Module]]

## Structured Search (Weeks 3-4)

- `[ ] D4.1` Centralize primary four-family sweep + parameter-count policy in the runners — see [[Hypergraph Families]]
- `[ ] D4.2 / D4.3` Aggregate gradient-norm proxies, heavy-tail checks, median-of-means stats — see [[Gradient Variance]]

## Qiskit Validation (Weeks 5-6)

- `[ ] D6.1` Split measured vs unmeasured circuit builders; emit QASM + transpilation metadata — see [[Qiskit Circuit Builder]]
- `[ ] D6.2 / D6.3` Implement classical/statevector/shots/noise cross-check; record gradient-SNR curves — see [[Qiskit Runner]]

## Kernel Validation (Week 6)

- `[ ] D8.1` Validate multi-scale Gaussian against exact mixture formula; add component sweep to grid — see [[Multi-Scale Gaussian Kernel]]

## Forge (Week 7)

- `[ ] D9.2 / D9.3` Replace export-only Forge mode with automated structural searches that save counterexamples/invariants — see [[Forge Runner]]

## Status Summary (as of 2026-04-08)

- ✅ **Anti-concentration track (AC1–AC6) is complete** — shipped on the Apr 8 10:30 critical-path deadline.
- ✅ **Scaling grid expansion (S6) is complete** — the runner now sweeps the full Cartesian product.
- Foundation and validation TODOs are mostly complete.
- Remaining open work centers on the Qiskit cross-check, multi-scale Gaussian validation, Forge structural search, and the deeper gradient-variance diagnostics (heavy-tail, curve fitting, memo artifacts).

## Related

- [[SMART Spec]]
- [[Scope Lock]]
- [[Implementation Choices]]
- [[Planning MOC]]
