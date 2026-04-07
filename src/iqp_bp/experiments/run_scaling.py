"""Scaling experiment runner.

Sweeps over (family × kernel × init × n) and estimates gradient variance.
Outputs one JSONL record per (family, kernel, init, n, seed) setting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.experiments.run_validation import (
    evaluate_anti_concentration_from_model,
    save_iqp_checkpoint,
)
from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.iqp.model import IQPModel
from iqp_bp.mmd.gradients import estimate_gradient_variance
from iqp_bp.mmd.mixture import dataset_expectations_batch
from iqp_bp.rng import split_seeds

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> None:
    """Entry point called by CLI with loaded config dict."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.jsonl"

    families = _as_list(cfg["circuit"]["family"])
    n_qubits_list = cfg["circuit"]["n_qubits"]
    kernels = _as_list(cfg["kernel"]["type"])
    inits = _as_list(cfg["init"]["scheme"])
    num_seeds = cfg["estimation"]["num_seeds"]
    num_a = cfg["estimation"]["num_a_samples"]
    num_z = cfg["estimation"]["num_z_samples"]
    base_seed = cfg["experiment"]["seed"]

    # Placeholder data: product Bernoulli (⟨Z_a⟩_p = 0 for |a|≥1)
    # TODO: Weeks 3-4 (D4.1) expand this runner to sweep the full Cartesian grid:
    # bandwidth list, small-angle std list, ER p list, and all SMART dataset targets.
    # Read first: itertools.product https://docs.python.org/3/library/itertools.html#itertools.product ;
    # pathlib.Path https://docs.python.org/3/library/pathlib.html#pathlib.Path ; json
    # https://docs.python.org/3/library/json.html
    dataset_cfg = cfg["dataset"]
    anti_concentration_cfg = cfg.get("anti_concentration", {})
    anti_concentration_enabled = bool(anti_concentration_cfg.get("enabled", True))
    anti_concentration_max_n = int(anti_concentration_cfg.get("max_n", 16))
    anti_concentration_alphas = tuple(
        float(alpha) for alpha in anti_concentration_cfg.get("alphas", (0.5, 1.0, 2.0))
    )
    anti_concentration_primary_alpha = float(
        anti_concentration_cfg.get("primary_alpha", 1.0)
    )
    anti_concentration_beta_min = float(anti_concentration_cfg.get("beta_min", 0.25))
    anti_concentration_second_moment_threshold = float(
        anti_concentration_cfg.get("second_moment_threshold", 1.0)
    )
    export_checkpoint = bool(anti_concentration_cfg.get("export_checkpoint", False))
    checkpoints_dir = output_dir / anti_concentration_cfg.get("checkpoint_dir", "checkpoints")

    total = len(families) * len(kernels) * len(inits) * len(n_qubits_list)
    done = 0

    with open(out_path, "w") as fout:
        for family in families:
            for kernel in kernels:
                for init_scheme in inits:
                    for n in n_qubits_list:
                        m = _compute_m(n, cfg["circuit"]["n_generators"])
                        circuit_seed, *theta_seeds_raw = split_seeds(base_seed, num_seeds + 1)

                        rng_circuit = np.random.default_rng(circuit_seed)
                        G = _make_G(family, n, m, cfg["circuit"], rng_circuit)
                        actual_m = G.shape[0]

                        data = _make_data(dataset_cfg, n, base_seed)

                        theta_list = [
                            _make_theta(init_scheme, G, data, cfg["init"], seed)
                            for seed in theta_seeds_raw[:num_seeds]
                        ]

                        kernel_params = _get_kernel_params(kernel, n, cfg["kernel"])
                        anti_concentration_summary = _summarize_anti_concentration(
                            enabled=anti_concentration_enabled,
                            family=family,
                            init_scheme=init_scheme,
                            kernel=kernel,
                            n=n,
                            G=G,
                            theta_list=theta_list,
                            max_n=anti_concentration_max_n,
                            alphas=anti_concentration_alphas,
                            primary_alpha=anti_concentration_primary_alpha,
                            beta_min=anti_concentration_beta_min,
                            second_moment_threshold=anti_concentration_second_moment_threshold,
                            export_checkpoint=export_checkpoint,
                            checkpoints_dir=checkpoints_dir,
                        )

                        for param_idx in range(min(5, actual_m)):  # sample 5 params per setting
                            rng_est = np.random.default_rng(base_seed + param_idx)
                            stats = estimate_gradient_variance(
                                G=G, data=data, param_idx=param_idx,
                                theta_seeds=theta_list, kernel=kernel,
                                num_a_samples=num_a, num_z_samples=num_z,
                                rng=rng_est, **kernel_params,
                            )
                            record = {
                                "family": family,
                                "kernel": kernel,
                                "init": init_scheme,
                                "n": n,
                                "m": actual_m,
                                "param_idx": param_idx,
                                **stats,
                                **kernel_params,
                                **anti_concentration_summary,
                            }
                            fout.write(json.dumps(record) + "\n")

                        # TODO: Weeks 3-4 (D4.2/D4.3) fit polynomial vs exponential
                        # scaling here and emit the summary artifacts used in the interim memo.
                        # Read first: scipy.optimize.curve_fit
                        # https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.optimize.curve_fit.html ;
                        # json https://docs.python.org/3/library/json.html
                        done += 1
                        log.info(f"[{done}/{total}] family={family} kernel={kernel} init={init_scheme} n={n}")


def _as_list(val):
    return val if isinstance(val, list) else [val]


def _compute_m(n: int, formula: str) -> int:
    import math
    if isinstance(formula, int):
        return formula
    return max(1, int(eval(formula, {"n": n, "log": math.log})))


def _make_G(family, n, m, circuit_cfg, rng):
    kwargs = {}
    if family == "bounded_degree":
        kwargs = circuit_cfg.get("bounded_degree", {})
    elif family == "erdos_renyi":
        p = circuit_cfg.get("erdos_renyi", {}).get("p_edge", 0.1)
        kwargs = {"p_edge": p[0] if isinstance(p, list) else p}
    elif family == "lattice":
        kwargs = {
            "dimension": circuit_cfg.get("lattice", {}).get("dimension", 1),
            "range_": circuit_cfg.get("lattice", {}).get("range", 1),
        }
    elif family == "dense":
        kwargs = {"expected_weight": circuit_cfg.get("dense", {}).get("expected_weight", 0.5)}
    elif family == "community":
        kwargs = circuit_cfg.get("community", {})
    elif family == "symmetric":
        kwargs = {"parity": circuit_cfg.get("symmetric", {}).get("parity", "even")}
    return make_hypergraph(family=family, n=n, m=m, rng=rng, **kwargs)


def _make_data(dataset_cfg, n, seed):
    dtype = dataset_cfg.get("type", "product_bernoulli")
    N = dataset_cfg.get("n_samples", 1000)
    rng = np.random.default_rng(seed + 999)
    if dtype == "product_bernoulli":
        return rng.integers(0, 2, size=(N, n), dtype=np.uint8)
    # TODO: Weeks 3-4 (D4.1) implement the Ising-like synthetic target and the
    # structured real/binary-mixture target from the SMART dataset plan.
    # Read first: NumPy Generator https://numpy.org/doc/stable/reference/random/generator.html
    raise NotImplementedError(f"Dataset type {dtype!r} not yet implemented")


def _make_theta(scheme, G, data, init_cfg, seed):
    m = G.shape[0]
    rng = np.random.default_rng(seed)
    if scheme == "uniform":
        lo = init_cfg.get("uniform", {}).get("low", -np.pi)
        hi = init_cfg.get("uniform", {}).get("high", np.pi)
        return rng.uniform(lo, hi, size=m)
    elif scheme == "small_angle":
        std = init_cfg.get("small_angle", {}).get("std", [0.1])
        s = std[0] if isinstance(std, list) else std
        return rng.normal(0, s, size=m)
    elif scheme == "data_dependent":
        dd_cfg = init_cfg.get("data_dependent", {})
        scale = dd_cfg.get("scale", 0.1)
        return scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
    raise ValueError(f"Unknown init scheme {scheme!r}")


def _get_kernel_params(kernel, n, kernel_cfg):
    bw = kernel_cfg.get("bandwidth", [1.0])
    sigma = bw[0] if isinstance(bw, list) else bw
    if kernel == "gaussian":
        return {"sigma": sigma}
    elif kernel == "laplacian":
        return {"sigma": sigma}
    elif kernel == "multi_scale_gaussian":
        # TODO: Week 6 (D8.1) promote the multi-scale Gaussian config into the
        # phase-2 sweep once the Gaussian baseline is fully characterized.
        # Read first: itertools.product https://docs.python.org/3/library/itertools.html#itertools.product
        msg = kernel_cfg.get("multi_scale_gaussian", {})
        return {"sigmas": msg.get("sigmas", [sigma]), "weights": msg.get("weights")}
    elif kernel == "polynomial":
        poly = kernel_cfg.get("polynomial", {})
        return {"degree": poly.get("degree", 2), "constant": poly.get("constant", 1.0)}
    elif kernel == "linear":
        return {}
    return {}


def _summarize_anti_concentration(
    *,
    enabled: bool,
    family: str,
    init_scheme: str,
    kernel: str,
    n: int,
    G: np.ndarray,
    theta_list: list[np.ndarray],
    max_n: int,
    alphas: tuple[float, ...],
    primary_alpha: float,
    beta_min: float,
    second_moment_threshold: float,
    export_checkpoint: bool,
    checkpoints_dir: Path,
) -> dict[str, Any]:
    """Return compact anti-concentration fields for one scaling setting."""
    if not enabled:
        return {
            "anti_concentration_available": False,
            "anti_concentration_reason": "disabled",
        }

    if n > max_n:
        return {
            "anti_concentration_available": False,
            "anti_concentration_reason": f"n_exceeds_max_n:{n}>{max_n}",
        }

    if not theta_list:
        return {
            "anti_concentration_available": False,
            "anti_concentration_reason": "missing_theta_seed",
        }

    model = IQPModel(G=G, theta=theta_list[0])
    result = evaluate_anti_concentration_from_model(
        model,
        provenance={
            "source": "scaling_seed",
            "family": family,
            "init": init_scheme,
            "kernel": kernel,
            "n": int(n),
            "theta_seed_index": 0,
        },
        max_qubits=max_n,
        alphas=alphas,
        primary_alpha=primary_alpha,
        beta_min=beta_min,
        second_moment_threshold=second_moment_threshold,
    )
    checkpoint_path = None
    if export_checkpoint:
        checkpoint_path = save_iqp_checkpoint(
            model,
            checkpoints_dir / _checkpoint_name(family=family, init_scheme=init_scheme, kernel=kernel, n=n),
            metadata={
                "family": family,
                "init": init_scheme,
                "kernel": kernel,
                "n": int(n),
                "theta_seed_index": 0,
                "source": "run_scaling",
            },
        )
    threshold_beta_by_alpha = {
        str(entry["alpha"]): float(entry["beta_hat"])
        for entry in result["threshold_checks"]
    }
    summary = {
        "anti_concentration_available": True,
        "anti_concentration_reason": "exact_small_n",
        "ac_theta_seed_index": 0,
        "ac_mode": result["mode"],
        "ac_primary_alpha": float(result["primary_alpha"]),
        "ac_primary_beta_hat": float(result["primary_beta_hat"]),
        "ac_passes_primary_threshold": bool(result["passes_primary_threshold"]),
        "ac_scaled_second_moment": float(result["scaled_second_moment"]),
        "ac_passes_second_moment_threshold": bool(
            result["passes_second_moment_threshold"]
        ),
        "ac_max_probability_scaled": float(result["max_probability_scaled"]),
        "ac_beta_hat_by_alpha": threshold_beta_by_alpha,
    }
    if checkpoint_path is not None:
        summary["ac_checkpoint_path"] = str(checkpoint_path)
    return summary


def _checkpoint_name(*, family: str, init_scheme: str, kernel: str, n: int) -> str:
    """Build a stable filename for a saved anti-concentration checkpoint."""
    return f"{family}_n{n}_{kernel}_{init_scheme}_seed0.npz"
