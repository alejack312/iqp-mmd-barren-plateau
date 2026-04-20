"""MMD² loss estimator.

Estimates MMD²(p, q_θ) via Monte Carlo over Z-words a ~ P_k:

    MMD²(p, q_θ) ≈ (1/B) Σ_{b=1}^B (⟨Z_{a_b}⟩_p - ⟨Z_{a_b}⟩_{q_θ})²

where a_b ~ P_k and ⟨Z_a⟩_{q_θ} is estimated via iqp_expectation.
"""

from __future__ import annotations

import numpy as np

from iqp_bp.iqp.expectation import iqp_expectation, iqp_expectation_exact
from iqp_bp.mmd.kernel import sample_a, spectral_weights_exact
from iqp_bp.mmd.mixture import dataset_expectations_batch
from iqp_bp.rng import split_rng


def mmd2(
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
    batch_size: int | None = None,
    return_details: bool = False,
    **kernel_params,
) -> float | dict:
    """Estimate MMD²(p, q_θ).

    Args:
        theta: IQP parameters, shape (m,)
        G: Generator matrix, shape (m, n)
        data: Dataset samples from p, shape (N, n), values in {0,1}
        kernel: Kernel type string
        num_a_samples: Number of Z-word samples B
        num_z_samples: Number of z samples for each ⟨Z_a⟩_{q_θ} estimate
        rng: Seeded RNG
        batch_size: Passed to :func:`iqp_expectation` to enable bounded-memory
            streaming over z-samples (see that function for details).
        return_details: If False (default), return a scalar float.  If True,
            return a dict with per-observable diagnostics and Monte Carlo
            uncertainty estimates (see Returns below).
        **kernel_params: Passed to kernel sampler (e.g., sigma=1.0)

    Returns:
        If ``return_details=False``: MMD² point estimate (scalar float).

        If ``return_details=True``: dict with keys:

        * ``mmd2`` – scalar point estimate (same as the default return value)
        * ``a_samples`` – sampled Z-word observables, shape ``(B, n)``, uint8
        * ``exp_p`` – ⟨Z_a⟩_p for each observable, shape ``(B,)``
        * ``exp_q`` – ⟨Z_a⟩_{q_θ} for each observable, shape ``(B,)``
        * ``contributions`` – per-observable squared differences
          ``(exp_p - exp_q)²``, shape ``(B,)``; ``mmd2 == contributions.mean()``
        * ``mc_diagnostics`` – dict with Monte Carlo uncertainty for the
          observable mixture:

          - ``point_estimate`` – same as ``mmd2``
          - ``sample_std`` – sample std of the ``contributions`` array
          - ``stderr`` – ``sample_std / sqrt(B)``
          - ``num_samples`` – B (number of sampled observables)
    """
    if rng is None:
        rng = np.random.default_rng()
    if num_a_samples <= 0:
        raise ValueError("num_a_samples must be positive")
    n = G.shape[1]

    # Split into independent substreams so kernel-word draws and IQP z-draws
    # do not interfere: refactoring num_a_samples does not shift iqp_rng state.
    kernel_rng, iqp_rng = split_rng(rng, 2)

    # Sample Z-words a ~ P_k
    a_samples = sample_a(kernel=kernel, n=n, num_a_samples=num_a_samples, rng=kernel_rng, **kernel_params)

    # Estimate ⟨Z_a⟩_p for all sampled a (vectorized)
    exp_p = dataset_expectations_batch(data, a_samples)

    # Estimate ⟨Z_a⟩_{q_θ} for each a
    exp_q = np.array([
        iqp_expectation(theta, G, a, num_z_samples=num_z_samples, rng=iqp_rng,
                        batch_size=batch_size)[0]
        for a in a_samples
    ])

    contributions = (exp_p - exp_q) ** 2
    estimate = float(np.mean(contributions))

    if not return_details:
        return estimate

    sample_std = float(np.std(contributions))
    return {
        "mmd2": estimate,
        "a_samples": a_samples,
        "exp_p": exp_p,
        "exp_q": exp_q,
        "contributions": contributions,
        "mc_diagnostics": {
            "point_estimate": estimate,
            "sample_std": sample_std,
            "stderr": sample_std / np.sqrt(len(contributions)),
            "num_samples": len(contributions),
        },
    }


def mmd2_exact_small_n(
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    kernel: str = "gaussian",
    return_details: bool = False,
    **kernel_params,
) -> float | dict:
    """Exact MMD²(p, q_θ) by summing over all 2^n observables.

    Enumerates the full observable support instead of sampling a ~ P_k.
    Uses the same spectral-weight convention as :func:`mmd2` so results
    converge as num_a_samples grows.  Only practical for n ≤ ~12.

    Args:
        theta: IQP parameters, shape (m,)
        G: Generator matrix, shape (m, n)
        data: Dataset samples from p, shape (N, n), values in {0,1}
        kernel: Kernel type string
        return_details: If True, return dict with per-observable arrays.
        **kernel_params: Kernel hyperparameters (e.g., sigma=1.0)

    Returns:
        Scalar float, or dict with keys:
        ``mmd2``, ``a_samples`` (all 2^n observables), ``exp_p``,
        ``exp_q``, ``contributions``, ``weights``.
    """
    n = G.shape[1]
    if n > 20:
        raise ValueError(f"mmd2_exact_small_n: n={n} > 20, too large to enumerate")

    # 1. Enumerate all 2^n observables (LSB = qubit 0, consistent with weights)
    idx = np.arange(2**n, dtype=np.intp)
    all_a = ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8)  # (2^n, n)

    # 2. Normalized spectral weights for each observable (same formula as MC sampler)
    weights = spectral_weights_exact(kernel, n, **kernel_params)  # (2^n,)

    # 3. Empirical ⟨Z_a⟩_p — vectorized over all observables
    exp_p = dataset_expectations_batch(data, all_a)  # (2^n,)

    # 4. Exact ⟨Z_a⟩_{q_θ} — enumerate all 2^n bitstrings per observable
    exp_q = np.array([iqp_expectation_exact(theta, G, a) for a in all_a])  # (2^n,)

    # 5. Weighted sum (exact, no Monte Carlo variance)
    contributions = (exp_p - exp_q) ** 2  # (2^n,)
    mmd2_val = float(np.dot(weights, contributions))

    if not return_details:
        return mmd2_val

    return {
        "mmd2": mmd2_val,
        "a_samples": all_a,
        "exp_p": exp_p,
        "exp_q": exp_q,
        "contributions": contributions,
        "weights": weights,
    }
