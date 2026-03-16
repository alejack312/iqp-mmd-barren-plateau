"""Unified sampling interface for all model types."""

from pathlib import Path
import pickle

import numpy as np
import iqpopt as iqp


def sample_from_model(
    model_name: str,
    params_path: str | Path,
    hyperparams: dict,
    num_samples: int = 5000,
    X_ref: np.ndarray = None,
    graph=None,
    num_steps_multiplier: int = 200,
    output_path: str | Path = None,
) -> np.ndarray:
    """Sample from a trained generative model.

    Args:
        model_name: One of 'IqpSimulator', 'IqpSimulatorBitflip',
                     'RestrictedBoltzmannMachine', 'DeepEBM', 'DeepGraphEBM'.
        params_path: Path to pickled model parameters.
        hyperparams: Model hyperparameters.
        num_samples: Number of samples to generate.
        X_ref: Reference data (used for initialization of some models).
        graph: NetworkX graph (required for DeepGraphEBM).
        num_steps_multiplier: MCMC steps = multiplier * n_features.
        output_path: If provided, save samples to this CSV path.

    Returns:
        Array of generated samples.
    """
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    n_features = X_ref.shape[-1] if X_ref is not None else None
    num_steps = num_steps_multiplier * n_features if n_features else 10000

    if model_name == "RestrictedBoltzmannMachine":
        samples = params.sample(num_samples, num_steps=num_steps)

    elif model_name == "DeepEBM":
        from qml_benchmarks.models import DeepEBM

        model = DeepEBM(**hyperparams)
        model.initialize(X_ref[:1000])
        model.params_ = params
        samples = model.sample(num_samples, num_steps=num_steps, max_chunk_size=100)

    elif model_name == "DeepGraphEBM":
        from iqp_mmd.models.graph_ebm import DeepGraphEBM

        model = DeepGraphEBM(G=graph, **hyperparams)
        model.initialize(X_ref[:1000])
        model.params_ = params
        samples = model.sample(num_samples)

    elif model_name in ("IqpSimulator", "IqpSimulatorBitflip"):
        import iqpopt.utils as iqp_utils

        bitflip = model_name == "IqpSimulatorBitflip"
        gates_config = dict(hyperparams["gates_config"])

        if gates_config["name"] == "gates_from_covariance" and X_ref is not None:
            gates_config["kwargs"]["data"] = X_ref[:1000]

        gate_fn = getattr(iqp_utils, gates_config["name"])
        gates = gate_fn(**gates_config["kwargs"])

        iqp_circuit = iqp.IqpSimulator(gates=gates, **hyperparams["model_config"], bitflip=bitflip)
        samples = iqp_circuit.sample(params, num_samples)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, samples, fmt="%d")

    return samples
