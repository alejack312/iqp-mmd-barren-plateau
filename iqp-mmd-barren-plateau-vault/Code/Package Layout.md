---
title: Package Layout
tags:
  - code
  - structure
---

# Package Layout

The repo ships two packages under `src/` plus auxiliary artifact folders. Both packages are installed via the root `pyproject.toml`.

## Repo Root

```
iqp-mmd-barren-plateau/
├── architecture.excalidraw      # architectural sketches
├── configs/                     # experiment YAMLs (base + experiments/)
├── docs/                        # technical docs, proposal, papers
├── forge/                       # Forge models + runs
├── notebooks/                   # sanity checks and plotting
├── results/                     # JSONL experiment outputs
├── scripts/                     # setup + run helpers
├── src/
│   ├── iqp_bp/                  # primary package (barren plateau study)
│   └── iqp_mmd/                 # sibling package (training toolkit)
├── tests/                       # pytest suite
├── pyproject.toml               # hatchling build, Python >=3.10
├── README.md                    # package quick-start
├── ScopeLock.md                 # locked study definition
└── TODOS.md                     # dependency-ordered task queue
```

## `src/iqp_bp/` — Barren Plateau Study

This is the primary target of the vault.

```
src/iqp_bp/
├── cli.py                          # Entry point: three subcommands
├── config.py                       # YAML loading + deep merge
├── rng.py                          # Deterministic named seed streams
├── hypergraph/
│   ├── families.py                 # G matrix generators (SMART + legacy)
│   └── hypothesis_strategies.py    # Hypothesis strategies for property tests
├── iqp/
│   ├── model.py                    # IQPModel wrapper + exact prob vector
│   └── expectation.py              # Classical ⟨Z_a⟩ estimator
├── mmd/
│   ├── kernel.py                   # Kernels + Z-word samplers
│   ├── loss.py                     # MMD² estimator
│   ├── mixture.py                  # Dataset parity expectations
│   └── gradients.py                # Analytic gradient + variance
├── qiskit/
│   ├── circuit_builder.py          # G → QuantumCircuit
│   ├── estimators.py               # Statevector / shots / noise
│   └── noise.py                    # Aer noise models
├── forge/
│   └── export_instances.py         # Export G to Forge .frg
└── experiments/
    ├── data_factory.py             # Runtime dataset generation
    ├── run_scaling.py              # Main scaling sweep
    ├── run_validation.py           # Anti-concentration runner
    ├── run_qiskit.py               # Qiskit validation
    └── run_forge.py                # Forge structural modeling
```

See [[Code MOC]] for module-by-module links.

## `src/iqp_mmd/` — Upstream Training Toolkit

Based on [XanaduAI/scaling-gqml](https://github.com/XanaduAI/scaling-gqml).

```
src/iqp_mmd/
├── checkpoint_export.py    # Emits .npz files for iqp_bp validation
├── circuits/               # PennyLane IQP circuit integration
├── cli/                    # iqp-train / iqp-evaluate / iqp-sample / iqp-dataset
├── config/                 # Path + hyperparameter loading
├── datasets/               # Ising, blobs, MNIST, D-Wave, genomic
├── metrics/                # MMD loss, KGEL, covariance
├── models/                 # IqpSimulator, DeepGraphEBM, GraphEBM
├── observables/            # Hamiltonian moments
├── sampling/               # Unified sampling interface
└── training/               # Training pipelines
```

See [[iqp_mmd Package]].

## Console Scripts

Defined in `pyproject.toml`:

| Script | Entry point |
|---|---|
| `iqp-bp` | `iqp_bp.cli:main` |
| `iqp-train` | `iqp_mmd.cli.train:main` |
| `iqp-evaluate` | `iqp_mmd.cli.evaluate:main` |
| `iqp-sample` | `iqp_mmd.cli.sample:main` |
| `iqp-dataset` | `iqp_mmd.cli.generate_dataset:main` |

## Supporting Directories

- **`configs/`** — YAML configs. `base.yaml` is the global default; `experiments/*.yaml` override.
- **`docs/`** — proposal, SMART spec, technical notes, papers (see [[Planning MOC]])
- **`tests/`** — pytest suite; focused on invariants, not numeric snapshots
- **`forge/`** — Forge models (`.frg`) and run outputs
- **`notebooks/`** — sanity checks and scaling plots
- **`results/`** — JSONL experiment outputs (gitignored in practice)
- **`scripts/`** — `setup_dev.sh`, `run_scaling.sh`

## Related

- [[Code MOC]]
- [[How a Scaling Run Works]]
- [[Checkpoint Bridge]]
