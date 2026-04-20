"""Qiskit-backed MMD^2 and gradient estimators.

MMD^2 between the data distribution ``p`` and the IQP output ``q_theta``,
under a kernel k that admits a Fourier decomposition with non-negative
weights w(a), can be written as an average over randomly-sampled Z-word
observables ``a``:

    MMD^2(p, q_theta) = E_{a ~ w} [ (<Z_a>_p - <Z_a>_q_theta)^2 ]

The classical path computes ``<Z_a>_q_theta`` analytically (Fourier formula).
The functions in this module replace that with either Qiskit statevector
simulation (exact) or Aer shot-based sampling (with optional noise model),
giving us a ground-truth cross-check. The parameter-shift gradient of MMD^2
follows directly from the chain rule:

    d/dtheta_i MMD^2 = -2 * E_{a ~ w} [ (<Z_a>_p - <Z_a>_q) * d<Z_a>_q / dtheta_i ]

so we just need per-observable expectations and per-observable gradients,
both of which ``estimators.py`` already provides in batched form.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from iqp_bp.mmd.kernel import sample_a
from iqp_bp.mmd.mixture import dataset_expectations_batch
from iqp_bp.qiskit.estimators import (
    batch_param_shift_gradients_shots,
    batch_param_shift_gradients_statevector,
    batch_shot_expectations,
    batch_statevector_expectations,
)
from iqp_bp.rng import split_rng


def _normalize_observable_inputs(
    *,
    G: np.ndarray,
    data: np.ndarray,
    kernel: str,
    num_a_samples: int,
    rng: np.random.Generator,
    a_samples: np.ndarray | None,
    exp_p: np.ndarray | None,
    weights: np.ndarray | None,
    kernel_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resolve sampled or exact observable batches into aligned arrays.

    The caller may supply either:
      * nothing — we sample ``num_a_samples`` fresh a-vectors from the
        kernel's Fourier distribution and compute ``<Z_a>_p`` ourselves,
      * ``a_samples`` — the caller has pinned which observables to evaluate
        (useful for variance studies that reuse the same a-set),
      * ``a_samples + exp_p`` — the caller has also precomputed the data
        expectations (saves a pass over ``data``),
      * ``a_samples + exp_p + weights`` — importance weights for enumerated
        exact evaluation rather than Monte-Carlo sampling.
    """
    # Path 1: no pre-supplied observables → draw from the kernel's Fourier
    # weights. sample_a encapsulates the kernel-specific sampling logic.
    if a_samples is None:
        a_samples = sample_a(
            kernel=kernel,
            n=G.shape[1],
            num_a_samples=num_a_samples,
            rng=rng,
            **kernel_params,
        )
    else:
        # Path 2+: coerce caller-supplied observables into the canonical dtype
        # and double-check the shape — a silent shape mismatch here would
        # produce wrong expectations later.
        a_samples = np.asarray(a_samples, dtype=np.uint8)
        if a_samples.ndim != 2 or a_samples.shape[1] != G.shape[1]:
            raise ValueError(
                f"a_samples must have shape (B, {G.shape[1]}), got {a_samples.shape}"
            )

    # If <Z_a>_p wasn't precomputed, do it now. Batched over all a_samples.
    if exp_p is None:
        exp_p = dataset_expectations_batch(data, a_samples)
    else:
        exp_p = np.asarray(exp_p, dtype=np.float64)
        # Shape lock: one value per a-sample.
        if exp_p.shape != (len(a_samples),):
            raise ValueError(
                f"exp_p must have shape ({len(a_samples)},), got {exp_p.shape}"
            )

    # Weights are optional — present when the caller wants exact enumeration
    # with Fourier-weight re-normalisation instead of Monte-Carlo mean.
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(a_samples),):
            raise ValueError(
                f"weights must have shape ({len(a_samples)},), got {weights.shape}"
            )
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        # Normalise so the weights are a probability simplex — the MMD^2
        # formula becomes an explicit weighted expectation.
        weights = weights / total

    return a_samples, exp_p, weights


def _evaluate_exp_q(
    *,
    qc_unm,
    qc_meas,
    observables: list[np.ndarray],
    theta: np.ndarray,
    mode: Literal["statevector", "shots"],
    n_shots: int,
    seed: int | None,
    noise_model,
) -> np.ndarray:
    """Dispatch <Z_a>_q evaluation to the statevector or shots backend."""
    if mode == "statevector":
        # Exact: no measurement collapse, no noise model, no shots.
        return batch_statevector_expectations(qc_unm, observables, theta)
    if mode == "shots":
        # Noise-aware: go through AerSimulator with measurements attached.
        return batch_shot_expectations(
            qc_measured=qc_meas,
            observables=observables,
            theta=theta,
            n_shots=n_shots,
            seed=seed,
            noise_model=noise_model,
            return_counts=False,
        )
    raise ValueError(f"Unknown mode: {mode!r}")


def _evaluate_dexp_q(
    *,
    qc_unm,
    qc_meas,
    observables: list[np.ndarray],
    theta: np.ndarray,
    param_idx: int,
    mode: Literal["statevector", "shots"],
    n_shots: int,
    seed: int | None,
    noise_model,
) -> np.ndarray:
    """Dispatch d<Z_a>_q/dtheta_i to the matching parameter-shift backend."""
    if mode == "statevector":
        return batch_param_shift_gradients_statevector(
            qc_unmeasured=qc_unm,
            observables=observables,
            theta=theta,
            param_idx=param_idx,
        )
    if mode == "shots":
        return batch_param_shift_gradients_shots(
            qc_measured=qc_meas,
            observables=observables,
            theta=theta,
            param_idx=param_idx,
            n_shots=n_shots,
            seed=seed,
            noise_model=noise_model,
        )
    raise ValueError(f"Unknown mode: {mode!r}")


def qiskit_mmd2(
    qc_unm,
    qc_meas,
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    rng: np.random.Generator | None = None,
    mode: Literal["statevector", "shots"] = "statevector",
    n_shots: int = 10_000,
    seed: int | None = None,
    noise_model=None,
    return_details: bool = False,
    a_samples: np.ndarray | None = None,
    exp_p: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    **kernel_params,
):
    """Estimate MMD^2(p, q_theta) with <Z_a>_q evaluated via Qiskit.

    The returned scalar is the sample-mean (or weighted mean) of
    ``(<Z_a>_p - <Z_a>_q)^2`` across the drawn / supplied observables.
    ``return_details=True`` adds the component arrays plus a small dict of
    Monte-Carlo diagnostics — useful when plotting convergence or
    decomposing the MMD by observable.
    """
    # Default RNG so the function is callable without plumbing through
    # ``iqp_bp.rng`` for quick experiments. Production callers should pass
    # an explicit seeded Generator.
    if rng is None:
        rng = np.random.default_rng()
    if num_a_samples <= 0:
        raise ValueError("num_a_samples must be positive")

    # Resolve all the optional observables/weights into concrete arrays.
    a_samples, exp_p, weights = _normalize_observable_inputs(
        G=G,
        data=data,
        kernel=kernel,
        num_a_samples=num_a_samples,
        rng=rng,
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        kernel_params=kernel_params,
    )
    # Convert list-of-rows into list-of-uint8-arrays, which is what the
    # batched estimator helpers expect.
    observables = [np.asarray(a, dtype=np.uint8) for a in a_samples]
    exp_q = _evaluate_exp_q(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        observables=observables,
        theta=theta,
        mode=mode,
        n_shots=n_shots,
        seed=seed,
        noise_model=noise_model,
    )

    # Per-observable squared differences: the integrand of MMD^2's
    # Monte-Carlo estimator.
    contributions = (exp_p - exp_q) ** 2
    # Average (weighted if the caller supplied weights).
    estimate = (
        float(np.dot(weights, contributions))
        if weights is not None
        else float(np.mean(contributions))
    )

    # Scalar return is enough for most callers.
    if not return_details:
        return estimate

    # Rich return: useful when plotting per-observable decomposition or
    # verifying that MC variance matches the expected asymptote.
    sample_std = float(np.std(contributions))
    # Weighted averages don't have a natural stderr; report 0 there so the
    # consumer can tell Monte-Carlo vs. enumerated runs apart.
    stderr = 0.0 if weights is not None else sample_std / np.sqrt(len(contributions))
    return {
        "mmd2": estimate,
        "a_samples": a_samples,
        "exp_p": exp_p,
        "exp_q": exp_q,
        "contributions": contributions,
        "weights": weights,
        "mc_diagnostics": {
            "point_estimate": estimate,
            "sample_std": sample_std,
            "stderr": stderr,
            "num_samples": len(contributions),
        },
        "mode": mode,
        "n_shots": n_shots if mode == "shots" else None,
    }


def qiskit_grad_mmd2(
    qc_unm,
    qc_meas,
    theta: np.ndarray,
    G: np.ndarray,
    data: np.ndarray,
    param_idx: int,
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    rng: np.random.Generator | None = None,
    mode: Literal["statevector", "shots"] = "statevector",
    n_shots: int = 10_000,
    seed: int | None = None,
    noise_model=None,
    a_samples: np.ndarray | None = None,
    exp_p: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    **kernel_params,
) -> float:
    """Estimate d/dtheta_i MMD^2 using Qiskit's parameter-shift rule.

    Chain-rule derivative of the MMD^2 estimator above:

        d/dtheta_i MMD^2 = -2 * E_{a ~ w} [(exp_p - exp_q) * dexp_q/dtheta_i]

    The ``-2`` factor comes from differentiating ``(exp_p - exp_q)^2`` and
    is applied once after the weighted mean.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Same observable / weight resolution as qiskit_mmd2.
    a_samples, exp_p, weights = _normalize_observable_inputs(
        G=G,
        data=data,
        kernel=kernel,
        num_a_samples=num_a_samples,
        rng=rng,
        a_samples=a_samples,
        exp_p=exp_p,
        weights=weights,
        kernel_params=kernel_params,
    )
    observables = [np.asarray(a, dtype=np.uint8) for a in a_samples]
    # We need both the current expectation and its parameter-shift gradient
    # for every observable in the batch.
    exp_q = _evaluate_exp_q(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        observables=observables,
        theta=theta,
        mode=mode,
        n_shots=n_shots,
        seed=seed,
        noise_model=noise_model,
    )
    dexp_q = _evaluate_dexp_q(
        qc_unm=qc_unm,
        qc_meas=qc_meas,
        observables=observables,
        theta=theta,
        param_idx=param_idx,
        mode=mode,
        n_shots=n_shots,
        seed=seed,
        noise_model=noise_model,
    )

    # Per-observable gradient integrand.
    terms = (exp_p - exp_q) * dexp_q
    # Weighted vs plain mean, then the -2 from the chain rule.
    if weights is None:
        return float(-2.0 * np.mean(terms))
    return float(-2.0 * np.dot(weights, terms))


def qiskit_estimate_gradient_variance(
    qc_builder,
    G: np.ndarray,
    data: np.ndarray,
    param_idx: int,
    theta_seeds: list[np.ndarray],
    kernel: str = "gaussian",
    num_a_samples: int = 512,
    rng: np.random.Generator | None = None,
    mode: Literal["statevector", "shots"] = "statevector",
    n_shots: int = 10_000,
    seed: int | None = None,
    noise_model=None,
    a_samples: np.ndarray | None = None,
    exp_p: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    **kernel_params,
) -> dict:
    """Estimate Var_theta[d/dtheta_i MMD^2] using Qiskit-evaluated gradients.

    This is the Qiskit-backed analogue of
    ``iqp_bp.mmd.gradients.estimate_gradient_variance``: for each theta seed
    in the list, compute the gradient and tally mean/var/std/snr across the
    ensemble. Used to directly test for barren-plateau decay in the Qiskit
    simulation (as opposed to the closed-form path).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Split the parent RNG into one independent child per theta seed so the
    # observable sampling for each gradient is uncorrelated.
    per_theta_rngs = split_rng(rng, len(theta_seeds))
    grads = []
    for idx, (theta, child_rng) in enumerate(zip(theta_seeds, per_theta_rngs)):
        # Call the injected builder, which instantiates fresh (unmeasured,
        # measured) circuits — this lets callers rebuild with re-transpiled
        # Qiskit output or different basis gates per theta.
        qc_unm, qc_meas = qc_builder(theta)
        # Each theta gets its own seed so the Aer sampler is reproducible
        # but independent across theta seeds.
        theta_seed = None if seed is None else int(child_rng.integers(0, 2**31 - 1))
        grads.append(
            qiskit_grad_mmd2(
                qc_unm=qc_unm,
                qc_meas=qc_meas,
                theta=theta,
                G=G,
                data=data,
                param_idx=param_idx,
                kernel=kernel,
                num_a_samples=num_a_samples,
                rng=child_rng,
                mode=mode,
                n_shots=n_shots,
                seed=theta_seed,
                noise_model=noise_model,
                a_samples=a_samples,
                exp_p=exp_p,
                weights=weights,
                **kernel_params,
            )
        )

    # Standard ensemble statistics — the barren-plateau signature is
    # ``var -> 0`` (and ``snr -> 0``) as n grows.
    grads = np.asarray(grads, dtype=np.float64)
    std = float(grads.std())
    mean = float(grads.mean())
    return {
        "mean": mean,
        "var": float(grads.var()),
        "std": std,
        "median": float(np.median(grads)),
        # Signal-to-noise ratio: |mean| / std. Falls with n if the gradient
        # collapses to zero faster than its variance shrinks.
        "snr": float(np.abs(mean) / std) if std > 0.0 else float("inf"),
        "n_seeds": len(theta_seeds),
        "mode": mode,
        "n_shots": n_shots if mode == "shots" else None,
    }
