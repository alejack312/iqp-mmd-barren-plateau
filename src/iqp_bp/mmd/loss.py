"""MMD² loss estimator.

Estimates MMD²(p, q_θ) via Monte Carlo over Z-words a ~ P_k:

    MMD²(p, q_θ) ≈ (1/B) Σ_{b=1}^B (⟨Z_{a_b}⟩_p - ⟨Z_{a_b}⟩_{q_θ})²

where a_b ~ P_k and ⟨Z_a⟩_{q_θ} is estimated via iqp_expectation.
"""

from __future__ import annotations

import numpy as np

from iqp_bp.iqp.expectation import iqp_expectation
from iqp_bp.mmd.kernel import sample_a
from iqp_bp.mmd.mixture import dataset_expectations_batch


def mmd2(
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
    **kernel_params,
) -> float:
    """Estimate MMD²(p, q_θ).

    Args:
        theta: IQP parameters, shape (m,)
        G: Generator matrix, shape (m, n)
        data: Dataset samples from p, shape (N, n), values in {0,1}
        kernel: Kernel type string
        num_a_samples: Number of Z-word samples B
        num_z_samples: Number of z samples for each ⟨Z_a⟩_{q_θ} estimate
        rng: Seeded RNG
        **kernel_params: Passed to kernel sampler (e.g., sigma=1.0)

    Returns:
        MMD² estimate (scalar)
    """
    if rng is None:
        rng = np.random.default_rng()
    # TODO: Week 1 (D1.3) expose per-a contributions and confidence diagnostics so
    # the minimal working pipeline can inspect one full MMD^2 estimate end-to-end.
    # Read first: json https://docs.python.org/3/library/json.html ; NumPy Generator
    # https://numpy.org/doc/stable/reference/random/generator.html
    n = G.shape[1]

    # Sample Z-words a ~ P_k
    a_samples = sample_a(kernel=kernel, n=n, num_a_samples=num_a_samples, rng=rng, **kernel_params)

    # Estimate ⟨Z_a⟩_p for all sampled a (vectorized)
    exp_p = dataset_expectations_batch(data, a_samples)

    # Estimate ⟨Z_a⟩_{q_θ} for each a
    exp_q = np.array([
        iqp_expectation(theta, G, a, num_z_samples=num_z_samples, rng=rng)[0]
        for a in a_samples
    ])

    # TODO: Week 2 (D2.1) add an exact small-n MMD^2 path here so each supported
    # kernel can be checked against brute force for n <= 10.
    # Read first: pytest parametrize https://docs.pytest.org/en/7.1.x/how-to/parametrize.html ;
    # itertools.product https://docs.python.org/3/library/itertools.html#itertools.product
    return float(np.mean((exp_p - exp_q) ** 2))
