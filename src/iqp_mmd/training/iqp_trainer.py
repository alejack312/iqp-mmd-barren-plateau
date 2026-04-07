"""IQP circuit training pipeline."""

from pathlib import Path
import pickle
import warnings

import numpy as np
import jax
import jax.numpy as jnp

import iqpopt.gen_qml as gen
from iqpopt.utils import initialize_from_data
from iqpopt import IqpSimulator, Trainer

from iqp_mmd.checkpoint_export import (
    generator_matrix_from_gates,
    save_deterministic_iqp_checkpoint,
)


def prepare_iqp_training(
    hyperparams: dict,
    X_train: jnp.ndarray,
    bitflip: bool = False,
    param_init_file: str = None,
    val_frac: float = None,
) -> tuple:
    """Set up objects needed for IQP circuit training.

    Args:
        hyperparams: Hyperparameter dictionary for this model/dataset combo.
        X_train: Training data array.
        bitflip: Use bitflip model variant.
        param_init_file: Path to pre-trained parameters (optional).
        val_frac: Validation split fraction (optional).

    Returns:
        Tuple of (model, trainer, loss_kwargs, val_kwargs, train_config).
    """
    from sklearn.model_selection import train_test_split
    import iqpopt.utils as iqp_utils

    gates_config = dict(hyperparams["gates_config"])
    if gates_config["name"] == "gates_from_covariance":
        gates_config["kwargs"]["data"] = X_train

    gate_fn = getattr(iqp_utils, gates_config["name"])
    gates = gate_fn(**gates_config["kwargs"])

    model = IqpSimulator(gates=gates, **hyperparams["model_config"], bitflip=bitflip)
    model._gates_export = gates
    trainer = Trainer(loss=gen.mmd_loss_iqp, **hyperparams["trainer_config"])
    train_config = hyperparams["train_config"]

    X_val = None
    if val_frac is not None:
        X_train, X_val = train_test_split(X_train, test_size=val_frac)

    loss_kwargs = dict(hyperparams["loss_config"])
    loss_kwargs["iqp_circuit"] = model
    loss_kwargs["ground_truth"] = X_train
    loss_kwargs["sqrt_loss"] = False

    if param_init_file is not None:
        with open(param_init_file, "rb") as f:
            params_init = jnp.array(pickle.load(f))
    else:
        params_init = initialize_from_data(
            gates,
            X_train,
            scale=hyperparams["init_config"]["init_scale"],
            param_noise=hyperparams["init_config"]["param_noise"],
        )

    loss_kwargs["params"] = params_init
    loss_kwargs["wires"] = list(range(X_train.shape[-1]))

    val_kwargs = None
    if X_val is not None:
        val_kwargs = dict(loss_kwargs)
        del val_kwargs["params"]
        val_kwargs["ground_truth"] = X_val

    return model, trainer, loss_kwargs, val_kwargs, train_config


def train_iqp(
    hyperparams: dict,
    X_train: jnp.ndarray,
    dataset_name: str,
    bitflip: bool = False,
    param_init_file: str = None,
    val_frac: float = None,
    turbo=None,
    seed: int = 666,
    output_dir: str | Path = None,
) -> dict:
    """Train an IQP circuit model end-to-end.

    Args:
        hyperparams: Hyperparameter config for this model/dataset.
        X_train: Training data.
        dataset_name: Name identifier for the dataset.
        bitflip: Use bitflip variant.
        param_init_file: Pre-trained parameters path.
        val_frac: Validation split fraction.
        turbo: Turbo mode setting.
        seed: Random seed.
        output_dir: Directory for saving outputs (loss plots, parameters).

    Returns:
        Dictionary with 'model', 'trainer', 'params', 'losses'.
    """
    np.random.seed(seed)

    model, trainer, loss_kwargs, val_kwargs, train_config = prepare_iqp_training(
        hyperparams, X_train, bitflip, param_init_file, val_frac
    )

    trainer.train(
        **train_config,
        loss_kwargs=loss_kwargs,
        val_kwargs=val_kwargs,
        turbo=turbo,
        random_state=np.random.randint(0, 99999),
    )

    name = "IqpSimulatorBitflip" if bitflip else "IqpSimulator"

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(output_dir / f"train_losses_{name}_{dataset_name}.csv", trainer.losses, delimiter=",")
        if val_frac is not None and trainer.val_losses:
            np.savetxt(output_dir / f"val_losses_{name}_{dataset_name}.csv", trainer.val_losses, delimiter=",")

        params_dir = output_dir / "trained_parameters"
        params_dir.mkdir(parents=True, exist_ok=True)
        with open(params_dir / f"params_{name}_{dataset_name}.pkl", "wb") as f:
            pickle.dump(trainer.final_params, f)

        try:
            checkpoint_path = params_dir / f"checkpoint_{name}_{dataset_name}.npz"
            gates = getattr(model, "_gates_export", None)
            if gates is None:
                raise ValueError("Model does not expose `_gates_export`; cannot reconstruct G.")

            n_qubits = int(hyperparams["model_config"]["n_qubits"])
            generator_matrix = generator_matrix_from_gates(gates, n_qubits=n_qubits)
            save_deterministic_iqp_checkpoint(
                path=checkpoint_path,
                G=generator_matrix,
                theta=np.asarray(trainer.final_params, dtype=np.float64),
                metadata={
                    "dataset_name": dataset_name,
                    "gate_fn": hyperparams["gates_config"]["name"],
                    "model_name": name,
                    "n_qubits": n_qubits,
                    "seed": int(seed),
                    "bitflip": bool(bitflip),
                    "source": "iqp_mmd.train_iqp",
                },
            )
        except Exception as exc:
            warnings.warn(
                "Failed to export deterministic IQP checkpoint for anti-concentration "
                f"validation: {exc}",
                RuntimeWarning,
            )

    return {
        "model": model,
        "trainer": trainer,
        "params": trainer.final_params,
        "losses": trainer.losses,
        "val_losses": getattr(trainer, "val_losses", None),
    }
