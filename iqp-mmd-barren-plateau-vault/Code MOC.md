---
title: Code MOC
tags:
  - moc
  - code
---

# Code Map of Content

Module-by-module documentation of the two packages in `src/`. The primary package for this vault is `iqp_bp` (the barren plateau study); `iqp_mmd` is the sibling training toolkit.

## Package Overview

- [[Package Layout]] — directory tree and where to find things
- [[How a Scaling Run Works]] — end-to-end walkthrough of `iqp-bp run-scaling`
- [[Checkpoint Bridge]] — how `iqp_mmd` trained models feed into `iqp_bp` validation

## `iqp_bp` — Barren Plateau Study

### Core Infrastructure

- [[CLI]] — `iqp_bp/cli.py` — three subcommands dispatcher
- [[Config]] — `iqp_bp/config.py` — YAML loading and deep-merge
- [[RNG]] — `iqp_bp/rng.py` — deterministic named seed streams

### Circuit Layer

- [[Hypergraph Families]] — `iqp_bp/hypergraph/families.py` — `G` generators
- [[Hypothesis Strategies]] — `iqp_bp/hypergraph/hypothesis_strategies.py` — property-based tests
- [[IQP Model]] — `iqp_bp/iqp/model.py` — `IQPModel` wrapper
- [[IQP Expectation]] — `iqp_bp/iqp/expectation.py` — classical $\langle Z_a \rangle$

### Loss Layer

- [[Kernel Module]] — `iqp_bp/mmd/kernel.py` — kernels and Z-word samplers
- [[MMD Loss Module]] — `iqp_bp/mmd/loss.py` — `mmd2()`
- [[Mixture Module]] — `iqp_bp/mmd/mixture.py` — data-side parity expectations
- [[Gradients Module]] — `iqp_bp/mmd/gradients.py` — analytic gradients + variance estimator

### Experiment Runners

- [[Scaling Runner]] — `iqp_bp/experiments/run_scaling.py` — the main sweep
- [[Validation Runner]] — `iqp_bp/experiments/run_validation.py` — anti-concentration
- [[Qiskit Runner]] — `iqp_bp/experiments/run_qiskit.py` — Qiskit cross-check
- [[Forge Runner]] — `iqp_bp/experiments/run_forge.py` — structural modeling export
- [[Data Factory]] — `iqp_bp/experiments/data_factory.py` — dataset generation

### Qiskit Layer

- [[Qiskit Circuit Builder]] — `iqp_bp/qiskit/circuit_builder.py` — G → `QuantumCircuit`

### Forge Layer

- [[Forge Export]] — `iqp_bp/forge/export_instances.py` — export G to `.frg`

## `iqp_mmd` — Upstream Training Toolkit

- [[iqp_mmd Package]] — high-level overview (models, training, sampling, metrics)
- [[Checkpoint Export]] — `iqp_mmd/checkpoint_export.py` — the bridge that emits `.npz` files for `iqp_bp`

## Tests

- [[Tests]] — the test suite and what each file covers

## Configs

- [[Configs]] — `configs/base.yaml` and `configs/experiments/*.yaml`
