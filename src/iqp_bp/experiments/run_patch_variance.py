"""Patch variance experiment runner.

Sweeps over expanding radii (r) around a central initialization point
to map the local variance of the loss landscape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.config import persist_experiment_manifest, resolve_experiment_grid
from iqp_bp.experiments.data_factory import make_dataset
from iqp_bp.iqp.model import IQPModel
from iqp_bp.mmd.gradients import estimate_gradient_variance
from iqp_bp.rng import STREAM_ESTIMATION, STREAM_THETA, derive_seed, experiment_stream_bundle

from iqp_bp.experiments.run_scaling import (
    _compute_m,
    _make_model,
    _make_theta,
    _get_kernel_params,
    _record_setting_fields,
    _setting_identity,
)

log = logging.getLogger(__name__)

def _sample_patch(theta_center: np.ndarray, r: float, num_samples: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Sample points uniformly from a hypercube of radius r around theta_center."""
    m = len(theta_center)
    # Draw uniform noise in the range [-r, r] for all parameters
    noise = rng.uniform(-r, r, size=(num_samples, m))
    
    return [theta_center + n for n in noise]

def run(cfg: dict[str, Any]) -> None:
    """Entry point called by CLI with loaded config dict."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "patch_variance_results.jsonl"

    # For the patch experiment, num_seeds becomes the number of points in the patch
    num_samples_per_patch = cfg["estimation"].get("num_seeds", 1000) 
    num_a = cfg["estimation"]["num_a_samples"]
    num_z = cfg["estimation"]["num_z_samples"]
    base_seed = cfg["experiment"]["seed"]
    dataset_cfg = cfg["dataset"]

    radii_grid = [2**(-i) for i in range(8, -1, -1)] 

    settings = resolve_experiment_grid(cfg)
    config_path, manifest_path = persist_experiment_manifest(cfg, settings)
    
    total = len(settings)
    done = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for setting in settings:
            setting_key = _setting_identity(setting)
            streams = experiment_stream_bundle(base_seed, "run_patch_variance", setting_key)
            n = int(setting["n"])
            family = str(setting["family"])
            kernel = str(setting["kernel"])
            
            init_scheme = str(setting["init_scheme"])

            m_requested = _compute_m(n, cfg["circuit"]["n_generators"])
            circuit_rng_seed = streams["circuit"]
            base_model = _make_model(
                family=family, n=n, m=m_requested, circuit_cfg=cfg["circuit"],
                rng=np.random.default_rng(circuit_rng_seed), rng_seed=circuit_rng_seed,
                er_p_edge=setting.get("er_p_edge"),
            )
            G = base_model.G
            actual_m = G.shape[0]

            data, dataset_metadata = make_dataset(dataset_cfg, n=n, seed=streams["data"])

            # Compute the central point (theta*) using data-dependent init
            # theta_center = _make_theta(
            #     init_scheme=init_scheme,
            #     G=G,
            #     data=data,
            #     init_cfg=cfg["init"],
            #     seed=derive_seed(base_seed, "run_patch_variance", setting_key, STREAM_THETA, 0),
            # )

            theta_center = _make_theta(
                init_scheme=init_scheme,
                G=G,
                data=data,
                init_cfg=cfg["init"],
                seed=derive_seed(base_seed, "run_patch_variance", setting_key, STREAM_THETA, 0),
                param_init_file=cfg.get("param_init_file"), 
            )

            kernel_params = _get_kernel_params(kernel=kernel, kernel_cfg=cfg["kernel"], bandwidth=setting.get("bandwidth"))

            param_idx = 0 
            
            rng_est = np.random.default_rng(
                derive_seed(base_seed, "run_patch_variance", setting_key, STREAM_ESTIMATION, param_idx)
            )

            for r in radii_grid:
                theta_patch_list = _sample_patch(theta_center, r, num_samples_per_patch, rng_est)

                stats = estimate_gradient_variance(
                    G=G,
                    data=data,
                    param_idx=param_idx,
                    theta_seeds=theta_patch_list,
                    kernel=kernel,
                    num_a_samples=num_a,
                    num_z_samples=num_z,
                    rng=rng_est,
                    **kernel_params,
                )

                record = {
                    **_record_setting_fields(setting),
                    "m": actual_m,
                    "param_idx": param_idx,
                    "radius_r": r,
                    "num_samples": num_samples_per_patch,
                    "dataset_metadata": dataset_metadata,
                    **stats,
                    **kernel_params,
                    "manifest_path": str(manifest_path),
                }
                fout.write(json.dumps(record) + "\n")

            done += 1
            log.info("[%s/%s] family=%s kernel=%s init=%s n=%s (Patch Sweep Complete)", done, total, family, kernel, init_scheme, n)

if __name__ == "__main__":
    import argparse
    import yaml
    from iqp_bp.experiments.run_patch_variance import run

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparams", type=str, required=True)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()

    with open(args.hyperparams, "r") as f:
        config = yaml.safe_load(f)
    
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir

    run(config)