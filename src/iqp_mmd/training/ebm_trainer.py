"""Energy-Based Model training pipelines (DeepEBM and DeepGraphEBM)."""

from pathlib import Path
import pickle

import numpy as np


def train_ebm(
    hyperparams: dict,
    X_train: np.ndarray,
    dataset_name: str,
    seed: int = 666,
    output_dir: str | Path = None,
) -> dict:
    """Train a Deep Energy-Based Model.

    Args:
        hyperparams: DeepEBM hyperparameters.
        X_train: Training data.
        dataset_name: Dataset identifier.
        seed: Random seed.
        output_dir: Directory for saving outputs.

    Returns:
        Dictionary with 'model', 'params', 'loss_history'.
    """
    from qml_benchmarks.models.energy_based_model import DeepEBM

    np.random.seed(seed)

    model = DeepEBM(**hyperparams, random_state=np.random.randint(0, 99999))
    model.fit(X_train)

    if output_dir:
        output_dir = Path(output_dir)
        params_dir = output_dir / "trained_parameters"
        params_dir.mkdir(parents=True, exist_ok=True)
        with open(params_dir / f"params_DeepEBM_{dataset_name}.pkl", "wb") as f:
            pickle.dump(model.params_, f)

        np.savetxt(
            output_dir / f"train_losses_DeepEBM_{dataset_name}.csv",
            model.loss_history_,
            delimiter=",",
        )

    return {
        "model": model,
        "params": model.params_,
        "loss_history": model.loss_history_,
    }


def train_graph_ebm(
    hyperparams: dict,
    X_train: np.ndarray,
    dataset_name: str,
    graph=None,
    seed: int = 666,
    output_dir: str | Path = None,
) -> dict:
    """Train a Graph-structured Deep Energy-Based Model.

    Args:
        hyperparams: DeepGraphEBM hyperparameters.
        X_train: Training data.
        dataset_name: Dataset identifier.
        graph: NetworkX graph object defining the structure.
        seed: Random seed.
        output_dir: Directory for saving outputs.

    Returns:
        Dictionary with 'model', 'params', 'loss_history'.
    """
    from iqp_mmd.models.graph_ebm import DeepGraphEBM

    np.random.seed(seed)

    if graph is None:
        raise ValueError("Must provide a NetworkX graph for DeepGraphEBM.")

    model = DeepGraphEBM(G=graph, **hyperparams, random_state=np.random.randint(0, 99999))
    model.fit(X_train)

    if output_dir:
        output_dir = Path(output_dir)
        params_dir = output_dir / "trained_parameters"
        params_dir.mkdir(parents=True, exist_ok=True)
        with open(params_dir / f"params_DeepGraphEBM_{dataset_name}.pkl", "wb") as f:
            pickle.dump(model.params_, f)

        np.savetxt(
            output_dir / f"train_losses_DeepGraphEBM_{dataset_name}.csv",
            model.loss_history_,
            delimiter=",",
        )

    return {
        "model": model,
        "params": model.params_,
        "loss_history": model.loss_history_,
    }
