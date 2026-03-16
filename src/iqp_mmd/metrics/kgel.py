"""KGEL (Kernel Goodness-of-fit via Expected Likelihood) metric."""

import numpy as np
import jax
import iqpopt.gen_qml as gen


def evaluate_kgel(
    ground_truth: np.ndarray,
    witnesses: np.ndarray,
    sigma: float,
    model_samples: np.ndarray = None,
    iqp_kwargs: dict = None,
) -> dict:
    """Compute KGEL metric for a generative model.

    Args:
        ground_truth: Test data samples.
        witnesses: Witness point samples.
        sigma: Kernel bandwidth.
        model_samples: Generated samples (sample-based models).
        iqp_kwargs: IQP circuit kwargs (circuit-based models).

    Returns:
        Dictionary with 'kgel' value and 'pi' weights.
    """
    if model_samples is not None:
        kgel, pi = gen.kgel_opt_samples(
            ground_truth=ground_truth,
            model_samples=model_samples,
            witnesses=witnesses,
            sigma=sigma,
        )
    elif iqp_kwargs is not None:
        key = iqp_kwargs.pop("key", jax.random.PRNGKey(0))
        key, subkey = jax.random.split(key)
        kgel, pi = gen.kgel_opt_iqp(
            **iqp_kwargs,
            ground_truth=ground_truth,
            witnesses=witnesses,
            sigma=sigma,
            key=subkey,
        )
        iqp_kwargs["key"] = key
    else:
        raise ValueError("Must provide either model_samples or iqp_kwargs.")

    return {"kgel": float(kgel), "pi": pi}
