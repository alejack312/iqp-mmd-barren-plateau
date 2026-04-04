"""Gradient estimators for MMD²(p, q_θ) with respect to θ.

Primary: JAX autodiff through the Monte Carlo MMD² estimator.
Secondary: finite differences (for validation and Qiskit comparisons).

Gradient formula (analytically):

    ∂_{θ_i} ⟨Z_a⟩_{q_θ} = -2 · (a·g_i mod 2) · E_{z~U}[sin(Φ(θ,z,a)) · (-1)^{z·g_i}]

    ∂_{θ_i} MMD²(p, q_θ) = -2 · E_{a~P_k}[(⟨Z_a⟩_p - ⟨Z_a⟩_{q_θ}) · ∂_{θ_i}⟨Z_a⟩_{q_θ}]
"""

from __future__ import annotations

import numpy as np

from iqp_bp.iqp.expectation import iqp_phase
from iqp_bp.mmd.kernel import sample_a
from iqp_bp.mmd.mixture import dataset_expectations_batch


def grad_expectation_analytic(
    theta: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
    param_idx: int,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute ∂_{θ_i} ⟨Z_a⟩_{q_θ} analytically.

    Args:
        theta: shape (m,)
        G: shape (m, n)
        a: shape (n,)
        param_idx: index i of the parameter to differentiate
        num_z_samples: Monte Carlo budget for z
        rng: seeded RNG

    Returns:
        Gradient estimate (scalar)
    """
    if rng is None:
        rng = np.random.default_rng()
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

    # Sample z ~ U({0,1}^n)
    z = rng.integers(0, 2, size=(num_z_samples, n), dtype=np.uint8)

    # Phases Φ(θ, z, a)
    phases = iqp_phase(theta, G, z, a)

    # (-1)^{z · g_i}
    z_dot_gi = (z @ g_i) % 2  # shape (B,)
    sign_i = 1 - 2 * z_dot_gi.astype(float)

    # ∂_{θ_i} ⟨Z_a⟩ = -2 · (a·g_i mod 2) · E[sin(Φ) · (-1)^{z·g_i}]
    return float(-2.0 * a_dot_gi * np.mean(np.sin(phases) * sign_i))


def grad_mmd2_analytic(
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    param_idx: int,
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    num_z_samples: int = 1024,
    rng: np.random.Generator | None = None,
    **kernel_params,
) -> float:
    """Estimate ∂_{θ_i} MMD²(p, q_θ) analytically.

    Returns:
        gradient estimate (scalar)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = G.shape[1]

    a_samples = sample_a(kernel=kernel, n=n, num_a_samples=num_a_samples, rng=rng, **kernel_params)
    exp_p = dataset_expectations_batch(data, a_samples)

    contributions = []
    for a, ep in zip(a_samples, exp_p):
        from iqp_bp.iqp.expectation import iqp_expectation
        eq, _ = iqp_expectation(theta, G, a, num_z_samples=num_z_samples, rng=rng)
        dq = grad_expectation_analytic(theta, G, a, param_idx, num_z_samples, rng)
        contributions.append((ep - eq) * dq)

    # ∂_{θ_i} MMD² = -2 · E_{a~P_k}[(⟨Z_a⟩_p - ⟨Z_a⟩_q) · ∂_{θ_i}⟨Z_a⟩_q]
    return float(-2.0 * np.mean(contributions))


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
    **kernel_params,
) -> float:
    """Finite-difference estimate of ∂_{θ_i} MMD²(p, q_θ).

    Used for correctness validation only (higher variance).
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
                  rng=np.random.default_rng(seed), **kernel_params)
    f_minus = mmd2(theta_minus, G, data, kernel=kernel,
                   num_a_samples=num_a_samples, num_z_samples=num_z_samples,
                   rng=np.random.default_rng(seed), **kernel_params)

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
    **kernel_params,
) -> dict:
    """Estimate Var_{θ~D}[∂_{θ_i} MMD²] over a set of θ seeds.

    Args:
        G: Generator matrix
        data: Dataset samples
        param_idx: Parameter index i
        theta_seeds: List of T parameter vectors, each shape (m,)
        kernel, num_a_samples, num_z_samples, **kernel_params: passed through

    Returns:
        dict with keys: mean, var, std, median, n_seeds
    """
    if rng is None:
        rng = np.random.default_rng()
    # TODO: Weeks 3-4 (D4.2/D4.3) extend this summary to include aggregate gradient-norm
    # proxies, heavy-tail checks, and median-of-means statistics for plateau diagnosis.
    # Read first: scipy.optimize.curve_fit
    # https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.optimize.curve_fit.html
    grads = [
        grad_mmd2_analytic(
            theta=th, G=G, data=data, param_idx=param_idx,
            kernel=kernel, num_a_samples=num_a_samples,
            num_z_samples=num_z_samples, rng=rng, **kernel_params
        )
        for th in theta_seeds
    ]
    grads = np.array(grads)
    return {
        "mean": float(grads.mean()),
        "var": float(grads.var()),
        "std": float(grads.std()),
        "median": float(np.median(grads)),
        "n_seeds": len(theta_seeds),
    }
