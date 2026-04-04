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

from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.mmd.gradients import estimate_gradient_variance
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

                        data = _make_data(dataset_cfg, n, base_seed)

                        theta_list = [
                            _make_theta(init_scheme, m, cfg["init"], seed)
                            for seed in theta_seeds_raw[:num_seeds]
                        ]

                        kernel_params = _get_kernel_params(kernel, n, cfg["kernel"])

                        for param_idx in range(min(5, m)):  # sample 5 params per setting
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
                                "m": m,
                                "param_idx": param_idx,
                                **stats,
                                **kernel_params,
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


def _make_theta(scheme, m, init_cfg, seed):
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
        # TODO: Weeks 3-4 (D4.1) replace this stub with the covariance-informed
        # data-dependent initializer required by the SMART comparison study.
        # Read first: JAX random https://docs.jax.dev/en/latest/jax.random.html ;
        # NumPy Generator https://numpy.org/doc/stable/reference/random/generator.html
        return rng.normal(0, 0.01, size=m)  # stub
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
