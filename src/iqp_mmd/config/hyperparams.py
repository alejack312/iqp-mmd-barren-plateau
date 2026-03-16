"""Hyperparameter configuration loading and defaults."""

from pathlib import Path

import yaml

DATASET_NAMES = [
    "8_blobs",
    "2D_ising",
    "spin_glass",
    "scale_free",
    "MNIST",
    "genomic-805",
    "dwave",
]

MODEL_NAMES = [
    "IqpSimulator",
    "IqpSimulatorBitflip",
    "RestrictedBoltzmannMachine",
    "DeepEBM",
    "DeepGraphEBM",
]

MODEL_COLORS = {
    "IqpSimulator": "#027ab0",
    "IqpSimulatorBitflip": "#1ebecd",
    "RestrictedBoltzmannMachine": "#49997c",
    "DeepEBM": "#ae3918",
    "DeepGraphEBM": "#ae3918",
    "True": "black",
    "Random": "#d19c2f",
    "RBM": "#49997c",
    "GAN": "#ae3918",
}

MODEL_DISPLAY_NAMES = {
    "IqpSimulator": "IQP",
    "IqpSimulatorBitflip": "Bitflip",
    "RestrictedBoltzmannMachine": "RBM",
    "DeepEBM": "DeepEBM",
    "DeepGraphEBM": "DeepGraphEBM",
    "True": "True",
    "Random": "Random",
    "RBM": "RBM",
    "GAN": "GAN",
}

USES_SAMPLES = {
    "8_blobs": {"IqpSimulator": False, "IqpSimulatorBitflip": True},
    "2D_ising": {"IqpSimulator": False, "IqpSimulatorBitflip": True},
    "spin_glass": {"IqpSimulator": False, "IqpSimulatorBitflip": True},
    "scale_free": {"IqpSimulator": False, "IqpSimulatorBitflip": True},
    "MNIST": {"IqpSimulator": False, "IqpSimulatorBitflip": True},
    "genomic-805": {"IqpSimulator": False, "IqpSimulatorBitflip": True},
    "dwave": {"IqpSimulator": False, "IqpSimulatorBitflip": True},
}


def load_hyperparams(path: str | Path) -> dict:
    """Load hyperparameters from a YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Nested dictionary: model_name -> dataset_name -> config.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_trained_sigmas(hyperparams: dict) -> dict:
    """Extract training sigma values from hyperparameter config.

    Args:
        hyperparams: Full hyperparameter dictionary.

    Returns:
        Dictionary: dataset_name -> list of sigma values.
    """
    sigmas = {}
    iqp_params = hyperparams.get("IqpSimulator", {})
    for dataset in DATASET_NAMES:
        if dataset in iqp_params:
            sigmas[dataset] = iqp_params[dataset].get("loss_config", {}).get("sigma", [])
    return sigmas
