---
title: iqp_mmd Package
tags:
  - code
  - iqp_mmd
  - training
---

# `iqp_mmd` — Upstream Training Toolkit

The sibling Python package to [[Code MOC|`iqp_bp`]]. Based on [XanaduAI/scaling-gqml](https://github.com/XanaduAI/scaling-gqml), the "Train on classical, deploy on quantum" paper ([[References#Scaling GQML|arXiv:2503.02934]]).

## Role in the Project

`iqp_mmd` is the **training pipeline** used to actually fit IQP circuits to data. `iqp_bp` is the **gradient-variance study** that investigates whether those training runs should even be possible. They share the IQP mathematical core but operate in different phases:

- `iqp_mmd` → train models, save parameters
- `iqp_bp` → measure gradient variance scaling, check distribution anti-concentration

The two meet at the [[Checkpoint Bridge]].

## Package Layout

```
src/iqp_mmd/
├── checkpoint_export.py    # → iqp_bp .npz bridge (see Checkpoint Export)
├── circuits/               # PennyLane IQP circuit integration
├── cli/                    # iqp-train, iqp-evaluate, iqp-sample, iqp-dataset
├── config/                 # Path + hyperparameter loading
├── datasets/               # Ising, blobs, MNIST, D-Wave, genomic
├── metrics/                # MMD loss, KGEL, covariance matrices
├── models/                 # IqpSimulator, DeepGraphEBM, GraphEBM wrappers
├── observables/            # Hamiltonian moments (magnetization, energy)
├── sampling/               # Unified sampling interface for all model types
└── training/               # Training pipelines + hyperparam search
```

## Supported Models

| Model | Type | Description |
|---|---|---|
| `IqpSimulator` | Quantum | Variational IQP circuit (exact classical simulation) |
| `IqpSimulatorBitflip` | Quantum | IQP circuit with bitflip noise channel |
| `RestrictedBoltzmannMachine` | Classical | RBM baseline via `qml-benchmarks` |
| `DeepEBM` | Classical | Deep energy-based model (contrastive divergence) |
| `DeepGraphEBM` | Classical | Graph-structured EBM with MaskedMLP |

## Supported Datasets

- 2D Ising lattice (9–16 spins) — MCMC sampling
- Scale-free Ising (10–1000 spins) — Barabási–Albert network + MCMC
- Spin blobs (16) — synthetic classification
- MNIST (784, binarized) — `torchvision`
- D-Wave (484) — Zenodo quantum annealer samples
- Genomic SNP (805 / 10K) — INRIA 1000 Genomes

## Console Scripts

| Script | Role |
|---|---|
| `iqp-train` | `iqp_mmd.cli.train:main` — fit a model |
| `iqp-evaluate` | `iqp_mmd.cli.evaluate:main` — compute metrics on samples |
| `iqp-sample` | `iqp_mmd.cli.sample:main` — draw samples from a trained model |
| `iqp-dataset` | `iqp_mmd.cli.generate_dataset:main` — materialize a dataset file |

## External Dependencies

- [IQPopt](https://github.com/XanaduAI/iqpopt) — IQP circuit optimization
- [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks) — classical baseline models

## Relationship to `iqp_bp`

> [!info] Separation of concerns
> - **`iqp_mmd`** — end-to-end training with the XanaduAI pipeline; primary user surface is the `iqp-*` CLI set
> - **`iqp_bp`** — scientific study of trainability; primary user surface is `iqp-bp run-scaling`
>
> These used to be closer together, but the SMART spec formalized the split. Anti-concentration evaluation lives on the `iqp_bp` side; the [[Checkpoint Export]] bridge lets trained `iqp_mmd` models flow into it.

## Related

- [[Checkpoint Export]]
- [[Checkpoint Bridge]]
- [[Package Layout]]
- [[References#Scaling GQML]]
