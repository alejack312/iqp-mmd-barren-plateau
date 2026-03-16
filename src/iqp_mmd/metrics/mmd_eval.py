"""MMD loss evaluation across models and sigma values."""

import numpy as np
import jax
import iqpopt.gen_qml as gen


def evaluate_mmd_loss(
    ground_truth: np.ndarray,
    model_samples: np.ndarray = None,
    sigma: float = 1.0,
    n_repeats: int = 20,
    iqp_kwargs: dict = None,
) -> dict:
    """Compute MMD loss between ground truth and model output.

    Supports both sample-based and IQP circuit-based evaluation.

    Args:
        ground_truth: Test data samples.
        model_samples: Generated samples (for sample-based models).
        sigma: Kernel bandwidth.
        n_repeats: Number of repeated evaluations for standard error.
        iqp_kwargs: IQP circuit kwargs (for circuit-based evaluation).
            Must include: iqp_circuit, params, n_ops, n_samples, key, wires.

    Returns:
        Dictionary with 'mean', 'std', 'losses'.
    """
    losses = []

    if model_samples is not None:
        gt_splits = np.array_split(ground_truth, n_repeats)
        sample_splits = np.array_split(model_samples, n_repeats)
        for gt, ms in zip(gt_splits, sample_splits):
            losses.append(float(gen.mmd_loss_samples(ground_truth=gt, model_samples=ms, sigma=sigma)))
    elif iqp_kwargs is not None:
        gt_splits = np.array_split(ground_truth, n_repeats)
        key = iqp_kwargs.pop("key", jax.random.PRNGKey(0))
        for gt in gt_splits:
            key, subkey = jax.random.split(key)
            losses.append(float(gen.mmd_loss_iqp(**iqp_kwargs, ground_truth=gt, sigma=sigma, key=subkey)))
        iqp_kwargs["key"] = key
    else:
        raise ValueError("Must provide either model_samples or iqp_kwargs.")

    losses = np.array(losses)
    return {
        "mean": float(np.mean(losses)),
        "std": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
        "losses": losses,
    }
