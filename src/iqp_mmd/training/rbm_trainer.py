"""Restricted Boltzmann Machine training pipeline."""

from pathlib import Path
import pickle

import numpy as np


def train_rbm(
    hyperparams: dict,
    X_train: np.ndarray,
    dataset_name: str,
    seed: int = 666,
    output_dir: str | Path = None,
) -> dict:
    """Train a Restricted Boltzmann Machine.

    Args:
        hyperparams: RBM hyperparameters (n_components, learning_rate, etc.).
        X_train: Training data.
        dataset_name: Dataset identifier.
        seed: Random seed.
        output_dir: Directory for saving the trained model.

    Returns:
        Dictionary with 'model'.
    """
    from qml_benchmarks.models.energy_based_model import RestrictedBoltzmannMachine

    np.random.seed(seed)

    model = RestrictedBoltzmannMachine(**hyperparams, random_state=np.random.randint(0, 99999))
    model.fit(X_train)

    if output_dir:
        output_dir = Path(output_dir)
        params_dir = output_dir / "trained_parameters"
        params_dir.mkdir(parents=True, exist_ok=True)
        with open(params_dir / f"params_RestrictedBoltzmannMachine_{dataset_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    return {"model": model}
