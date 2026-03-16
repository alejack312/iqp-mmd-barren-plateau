# IQP-MMD: Modular Toolkit for IQP Circuit Training with MMD Loss

A modular Python package for training and evaluating generative quantum machine learning models using IQP (Instantaneous Quantum Polynomial) circuits with Maximum Mean Discrepancy (MMD) loss.

Based on [XanaduAI/scaling-gqml](https://github.com/XanaduAI/scaling-gqml) — *"Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits"* ([arXiv:2503.02934](https://arxiv.org/abs/2503.02934)).

## Package Structure

```
src/iqp_mmd/
    circuits/        # PennyLane IQP circuit integration
    cli/             # Command-line interfaces (train, evaluate, sample, generate_dataset)
    config/          # Path management, hyperparameter loading, model display config
    datasets/        # Dataset generation (Ising, blobs) and download (MNIST, D-Wave, genomic)
    metrics/         # Evaluation: MMD loss, KGEL, covariance matrices
    models/          # Model wrappers: IqpSimulator (sklearn), DeepGraphEBM, GraphEBM
    observables/     # Hamiltonian moments: magnetization, energy (circuit + sample-based)
    sampling/        # Unified sampling interface for all model types
    training/        # Training pipelines: IQP, RBM, DeepEBM, DeepGraphEBM + hyperparam search
configs/
    hyperparameters.yaml   # Best hyperparameters from the paper
tests/
    test_imports.py        # Import and basic functionality tests
```

## Installation

```bash
# Core package
pip install -e .

# With dataset generation dependencies
pip install -e ".[datasets]"

# With classical baseline models (qml-benchmarks)
pip install -e ".[benchmarks]"

# Development
pip install -e ".[dev]"
```

### External Dependencies

This package depends on two Xanadu libraries:

- [IQPopt](https://github.com/XanaduAI/iqpopt) — IQP circuit optimization
- [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks) — Classical baseline models (RBM, DeepEBM)

## Quick Start

### Python API

```python
from iqp_mmd.datasets.loaders import load_csv_dataset, normalize_binary
from iqp_mmd.config.hyperparams import load_hyperparams
from iqp_mmd.training import train_iqp
from iqp_mmd.sampling import sample_from_model
from iqp_mmd.metrics.mmd_eval import evaluate_mmd_loss

# Load data
X_train = normalize_binary(load_csv_dataset("data/ising_train.csv"))

# Load hyperparameters
hp = load_hyperparams("configs/hyperparameters.yaml")

# Train an IQP circuit
result = train_iqp(
    hyperparams=hp["IqpSimulator"]["2D_ising"],
    X_train=X_train,
    dataset_name="2D_ising",
    output_dir="./output",
)

# Evaluate
X_test = normalize_binary(load_csv_dataset("data/ising_test.csv"))
samples = result["model"].sample(result["params"], 5000)
mmd = evaluate_mmd_loss(ground_truth=X_test, model_samples=samples, sigma=1.0)
print(f"MMD loss: {mmd['mean']:.6f} +/- {mmd['std']:.6f}")
```

### CLI

```bash
# Generate datasets
iqp-dataset --dataset ising_lattice --output-dir ./data/ising --width 4 --temperature 3.0

# Train a model
iqp-train --model IqpSimulatorBitflip --dataset 2D_ising \
    --dataset-path ./data/ising/train.csv \
    --hyperparams ./configs/hyperparameters.yaml \
    --output-dir ./output

# Sample from trained model
iqp-sample --model IqpSimulatorBitflip --dataset 2D_ising \
    --params-path ./output/trained_parameters/params_IqpSimulatorBitflip_2D_ising.pkl \
    --hyperparams ./configs/hyperparameters.yaml \
    --ref-data ./data/ising/test.csv \
    --num-samples 5000 --output ./output/samples.csv

# Evaluate
iqp-evaluate --metric mmd \
    --samples-path ./output/samples.csv \
    --test-path ./data/ising/test.csv \
    --sigma 0.6 1.3
```

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| `IqpSimulator` | Quantum | Variational IQP circuit (exact classical simulation) |
| `IqpSimulatorBitflip` | Quantum | IQP circuit with bitflip noise channel |
| `RestrictedBoltzmannMachine` | Classical | RBM baseline via qml-benchmarks |
| `DeepEBM` | Classical | Deep energy-based model (contrastive divergence) |
| `DeepGraphEBM` | Classical | Graph-structured EBM with MaskedMLP |

## Supported Datasets

| Dataset | Spins | Source |
|---------|-------|--------|
| 2D Ising lattice | 9-16 | MCMC sampling |
| Scale-free Ising | 10-1000 | Barabasi-Albert network + MCMC |
| Spin blobs | 16 | Synthetic classification |
| MNIST (binarized) | 784 | torchvision |
| D-Wave | 484 | Zenodo (quantum annealer samples) |
| Genomic SNP | 805 / 10K | INRIA 1000 Genomes |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
