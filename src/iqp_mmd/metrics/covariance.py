"""Covariance matrix computation for model evaluation."""

import numpy as np
from iqpopt.utils import construct_convariance_matrix


def compute_covariance_matrix(
    model_samples: np.ndarray = None,
    iqp_kwargs: dict = None,
) -> np.ndarray:
    """Compute the covariance matrix of model outputs.

    Args:
        model_samples: Generated samples (for sample-based models).
        iqp_kwargs: IQP circuit kwargs with circuit, params, n_samples, key, etc.

    Returns:
        Covariance matrix as numpy array.
    """
    if model_samples is not None:
        samples = model_samples
        if 0 in samples:
            samples = 1 - 2 * samples
        return np.cov(samples.T)
    elif iqp_kwargs is not None:
        return construct_convariance_matrix(**iqp_kwargs)
    else:
        raise ValueError("Must provide either model_samples or iqp_kwargs.")
