"""Gradient estimators for MMD²(p, q_θ) with respect to θ.

Primary: JAX autodiff through the Monte Carlo MMD² estimator.
Secondary: finite differences (for validation and Qiskit comparisons).

Gradient formula (analytically):

    ∂_{θ_i} ⟨Z_a⟩_{q_θ} = -2 · (a·g_i mod 2) · E_{z~U}[sin(Φ(θ,z,a)) · (-1)^{z·g_i}]

    ∂_{θ_i} MMD²(p, q_θ) = -2 · E_{a~P_k}[(⟨Z_a⟩_p - ⟨Z_a⟩_{q_θ}) · ∂_{θ_i}⟨Z_a⟩_{q_θ}]
"""

from __future__ import annotations

import numpy as np

from iqp_bp.iqp.expectation import iqp_expectation, iqp_phase
from iqp_bp.mmd.kernel import sample_a
from iqp_bp.mmd.mixture import dataset_expectations_batch
from iqp_bp.rng import split_rng


def grad_expectation_analytic(
    theta: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
    param_idx: int,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
    batch_size: int | None = None,
) -> float:
    """Compute ∂_{θ_i} ⟨Z_a⟩_{q_θ} analytically.

    Args:
        theta: shape (m,)
        G: shape (m, n)
        a: shape (n,)
        param_idx: index i of the parameter to differentiate
        num_z_samples: Monte Carlo budget for z
        rng: seeded RNG
        batch_size: If given, draw z-samples in chunks of this size so peak
            memory stays O(batch_size · n).  The running mean of
            sin(Φ) · (-1)^{z·g_i} is accumulated exactly.

    Returns:
        Gradient estimate (scalar)
    """
    if rng is None:
        rng = np.random.default_rng()
    if num_z_samples <= 0:
        raise ValueError("num_z_samples must be positive")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided")
    # TODO: Week 2 (D2.1) implement the JAX autodiff estimator promised in the
    # SMART spec and compare it against this analytic path on small-n problems.
    # Read first: jax.grad https://docs.jax.dev/en/latest/_autosummary/jax.grad.html ;
    # jax.value_and_grad https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html ;
    # jax.vmap https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html ;
    # jax.jit https://docs.jax.dev/en/latest/_autosummary/jax.jit.html ;
    # jax.random https://docs.jax.dev/en/latest/jax.random.html
    n = G.shape[1]
    g_i = G[param_idx]

    # (a · g_i mod 2)
    a_dot_gi = int((a @ g_i) % 2)
    if a_dot_gi == 0:
        return 0.0  # generator g_i doesn't contribute to Z_a

    if batch_size is None:
        # Single-batch path
        z = rng.integers(0, 2, size=(num_z_samples, n), dtype=np.uint8)
        phases = iqp_phase(theta, G, z, a)
        z_dot_gi = (z @ g_i) % 2  # shape (B,)
        sign_i = 1 - 2 * z_dot_gi.astype(float)
        # ∂_{θ_i} ⟨Z_a⟩ = -2 · (a·g_i mod 2) · E[sin(Φ) · (-1)^{z·g_i}]
        return float(-2.0 * a_dot_gi * np.mean(np.sin(phases) * sign_i))

    # Streaming path — accumulate the running sum of sin(Φ)·(-1)^{z·g_i}
    # over bounded z-chunks; divide by total count at the end.
    running_sum = 0.0
    remaining = num_z_samples
    while remaining > 0:
        chunk = min(batch_size, remaining)
        z_chunk = rng.integers(0, 2, size=(chunk, n), dtype=np.uint8)
        phases = iqp_phase(theta, G, z_chunk, a)
        z_dot_gi = (z_chunk @ g_i) % 2
        sign_i = 1 - 2 * z_dot_gi.astype(float)
        running_sum += float(np.sum(np.sin(phases) * sign_i))
        remaining -= chunk

    return float(-2.0 * a_dot_gi * running_sum / num_z_samples)


def grad_mmd2_analytic(
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    param_idx: int,
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
    batch_size: int | None = None,
    a_samples: np.ndarray | None = None,
    exp_p: np.ndarray | None = None,
    observable_weights: np.ndarray | None = None,
    **kernel_params,
) -> float:
    """Estimate ∂_{θ_i} MMD²(p, q_θ) analytically.

    Args:
        batch_size: Passed to :func:`iqp_expectation` and
            :func:`grad_expectation_analytic` to enable bounded-memory
            streaming over z-samples.
        a_samples: Pre-computed Z-word observables, shape ``(B, n)``.  When
            supplied together with ``exp_p``, the kernel-sampling step is
            skipped and both arrays are used directly.  Pass the values from a
            ``mmd2(return_details=True)`` call to keep gradient debugging and
            MMD debugging numerically aligned on identical observables.
        exp_p: Pre-computed ⟨Z_a⟩_p expectations, shape ``(B,)``.  Must be
            provided together with ``a_samples`` or not at all.
        observable_weights: Optional normalized weights for each observable in
            ``a_samples``. When omitted the estimator uses the simple sample
            mean, matching the Monte Carlo path. When provided it uses the
            weighted sum, which is needed for exact small-n observable sweeps.

    Returns:
        gradient estimate (scalar)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = G.shape[1]

    if (a_samples is None) != (exp_p is None):
        raise ValueError("Provide both a_samples and exp_p, or neither.")
    if observable_weights is not None and a_samples is None:
        raise ValueError("observable_weights requires pre-computed a_samples and exp_p")

    if a_samples is None:
        # Split into independent substreams: kernel-word draws must not share
        # state with IQP z-draws so that changing num_a_samples does not shift
        # iqp_rng.
        kernel_rng, iqp_rng = split_rng(rng, 2)
        a_samples = sample_a(kernel=kernel, n=n, num_a_samples=num_a_samples, rng=kernel_rng, **kernel_params)
        exp_p = dataset_expectations_batch(data, a_samples)
    else:
        # a_samples and exp_p were pre-computed (e.g. from mmd2(return_details=True));
        # skip kernel-sampling and use rng directly for IQP z-draws only.
        _, iqp_rng = split_rng(rng, 2)

    contributions = []
    for a, ep in zip(a_samples, exp_p):
        eq, _ = iqp_expectation(theta, G, a, num_z_samples=num_z_samples, rng=iqp_rng,
                                batch_size=batch_size)
        dq = grad_expectation_analytic(theta, G, a, param_idx, num_z_samples, iqp_rng,
                                       batch_size=batch_size)
        contributions.append((ep - eq) * dq)

    # ∂_{θ_i} MMD² = -2 · E_{a~P_k}[(⟨Z_a⟩_p - ⟨Z_a⟩_q) · ∂_{θ_i}⟨Z_a⟩_q]
    contribution_array = np.asarray(contributions, dtype=np.float64)
    if observable_weights is None:
        return float(-2.0 * np.mean(contribution_array))

    weights = np.asarray(observable_weights, dtype=np.float64)
    if weights.shape != contribution_array.shape:
        raise ValueError(
            "observable_weights must match the observable count; "
            f"got {weights.shape} vs {contribution_array.shape}"
        )
    return float(-2.0 * np.dot(weights, contribution_array))


def grad_mmd2_finite_diff(
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    param_idx: int,
    eps: float = 1e-4,
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    num_z_samples: int = 2048,
    rng: np.random.Generator | None = None,
    batch_size: int | None = None,
    **kernel_params,
) -> float:
    """Finite-difference estimate of ∂_{θ_i} MMD²(p, q_θ).

    Used for correctness validation only (higher variance).

    Args:
        batch_size: Passed to :func:`mmd2` for bounded-memory z-streaming.
    """
    from iqp_bp.mmd.loss import mmd2
    if rng is None:
        rng = np.random.default_rng()

    theta_plus = theta.copy()
    theta_minus = theta.copy()
    theta_plus[param_idx] += eps
    theta_minus[param_idx] -= eps

    seed = int(rng.integers(0, 2**31))
    f_plus = mmd2(theta_plus, G, data, kernel=kernel,
                  num_a_samples=num_a_samples, num_z_samples=num_z_samples,
                  rng=np.random.default_rng(seed), batch_size=batch_size,
                  **kernel_params)
    f_minus = mmd2(theta_minus, G, data, kernel=kernel,
                   num_a_samples=num_a_samples, num_z_samples=num_z_samples,
                   rng=np.random.default_rng(seed), batch_size=batch_size,
                   **kernel_params)

    return (f_plus - f_minus) / (2 * eps)


def estimate_gradient_variance(
    G: np.ndarray,
    data: np.ndarray,
    param_idx: int,
    theta_seeds: list[np.ndarray],
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
    batch_size: int | None = None,
    **kernel_params,
) -> dict:
    """Estimate Var_{θ~D}[∂_{θ_i} MMD²] over a set of θ seeds.

    Args:
        G: Generator matrix
        data: Dataset samples
        param_idx: Parameter index i
        theta_seeds: List of T parameter vectors, each shape (m,)
        kernel, num_a_samples, num_z_samples, **kernel_params: passed through
        batch_size: Passed to :func:`grad_mmd2_analytic` for bounded-memory
            z-streaming.

    Returns:
        dict with keys: mean, var, std, median, n_seeds
    """
    if rng is None:
        rng = np.random.default_rng()
    # TODO: Weeks 3-4 (D4.2/D4.3) extend this summary to include aggregate gradient-norm
    # proxies, heavy-tail checks, and median-of-means statistics for plateau diagnosis.
    # Read first: scipy.optimize.curve_fit
    # https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.optimize.curve_fit.html

    # Give each theta seed its own independent substream so that adding or
    # reordering seeds does not shift draws for other indices.
    per_theta_rngs = split_rng(rng, len(theta_seeds))
    grads = [
        grad_mmd2_analytic(
            theta=th, G=G, data=data, param_idx=param_idx,
            kernel=kernel, num_a_samples=num_a_samples,
            num_z_samples=num_z_samples, rng=child_rng,
            batch_size=batch_size, **kernel_params
        )
        for th, child_rng in zip(theta_seeds, per_theta_rngs)
    ]
    grads = np.array(grads)
    return {
        "mean": float(grads.mean()),
        "var": float(grads.var()),
        "std": float(grads.std()),
        "median": float(np.median(grads)),
        "n_seeds": len(theta_seeds),
    }
