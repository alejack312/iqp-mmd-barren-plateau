"""Scaling experiment runner.

Sweeps over the explicit scaling grid and estimates gradient variance.
Outputs one JSONL record per resolved scalar setting and parameter index.
"""

from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.config import persist_experiment_manifest, resolve_experiment_grid
from iqp_bp.experiments.data_factory import make_dataset
from iqp_bp.experiments.run_validation import (
    evaluate_anti_concentration_from_model,
    save_iqp_checkpoint,
    write_anti_concentration_artifacts,
)
from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.iqp.model import IQPModel
from iqp_bp.mmd.gradients import estimate_gradient_variance
from iqp_bp.mmd.mixture import dataset_expectations_batch
from iqp_bp.rng import (
    STREAM_ESTIMATION,
    STREAM_THETA,
    derive_seed,
    experiment_stream_bundle,
    named_seed_streams,
)

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> None:
    """Entry point called by CLI with loaded config dict."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.jsonl"

    num_seeds = cfg["estimation"]["num_seeds"]
    num_a = cfg["estimation"]["num_a_samples"]
    num_z = cfg["estimation"]["num_z_samples"]
    base_seed = cfg["experiment"]["seed"]
    dataset_cfg = cfg["dataset"]

    anti_concentration_cfg = cfg.get("anti_concentration", {})
    anti_concentration_enabled = bool(anti_concentration_cfg.get("enabled", True))
    anti_concentration_max_n = int(anti_concentration_cfg.get("max_n", 16))
    anti_concentration_alphas = tuple(
        float(alpha) for alpha in anti_concentration_cfg.get("alphas", (0.5, 1.0, 2.0))
    )
    anti_concentration_primary_alpha = float(anti_concentration_cfg.get("primary_alpha", 1.0))
    anti_concentration_beta_min = float(anti_concentration_cfg.get("beta_min", 0.25))
    anti_concentration_second_moment_threshold = float(
        anti_concentration_cfg.get("second_moment_threshold", 1.0)
    )
    export_checkpoint = bool(anti_concentration_cfg.get("export_checkpoint", False))
    checkpoints_dir = output_dir / anti_concentration_cfg.get("checkpoint_dir", "checkpoints")
    anti_concentration_dir = output_dir / anti_concentration_cfg.get(
        "artifact_dir", "anti_concentration"
    )

    settings = resolve_experiment_grid(cfg)
    config_path, manifest_path = persist_experiment_manifest(cfg, settings)
    total = len(settings)
    done = 0

    # Design decision: per-observable MMD² diagnostic arrays (a_samples, exp_p,
    # exp_q, contributions) are kept in-memory only if mmd2(return_details=True)
    # is ever called from this loop.  They are NOT written to results.jsonl or any
    # sidecar file because each array has shape (num_a_samples,) per estimate and
    # would dwarf the compact scalar records we persist here.  If sidecar storage
    # is needed in the future, write one JSON file per setting and store only a
    # stable pointer (e.g. the file path) as a field in results.jsonl.
    with open(out_path, "w", encoding="utf-8") as fout:
        for setting in settings:
            setting_key = _setting_identity(setting)
            streams = experiment_stream_bundle(base_seed, "run_scaling", setting_key)
            n = int(setting["n"])
            family = str(setting["family"])
            kernel = str(setting["kernel"])
            init_scheme = str(setting["init_scheme"])

            m_requested = _compute_m(n, cfg["circuit"]["n_generators"])
            circuit_rng_seed = streams["circuit"]
            base_model = _make_model(
                family=family,
                n=n,
                m=m_requested,
                circuit_cfg=cfg["circuit"],
                rng=np.random.default_rng(circuit_rng_seed),
                rng_seed=circuit_rng_seed,
                er_p_edge=setting.get("er_p_edge"),
            )
            G = base_model.G
            actual_m = G.shape[0]

            data, dataset_metadata = make_dataset(
                dataset_cfg,
                n=n,
                seed=streams["data"],
            )

            theta_list = [
                _make_theta(
                    init_scheme=init_scheme,
                    G=G,
                    data=data,
                    init_cfg=cfg["init"],
                    seed=derive_seed(base_seed, "run_scaling", setting_key, STREAM_THETA, idx),
                    small_angle_std=setting.get("small_angle_std"),
                )
                for idx in range(num_seeds)
            ]

            kernel_params = _get_kernel_params(
                kernel=kernel,
                kernel_cfg=cfg["kernel"],
                bandwidth=setting.get("bandwidth"),
            )

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
                artifact_dir=anti_concentration_dir,
                artifact_stem=_setting_stem(setting),
                provenance={
                    **_record_setting_fields(setting),
                    "m": int(actual_m),
                    "dataset_metadata": dataset_metadata,
                },
                model_provenance=base_model.provenance,
            )

            for param_idx in range(min(5, actual_m)):
                rng_est = np.random.default_rng(
                    derive_seed(base_seed, "run_scaling", setting_key, STREAM_ESTIMATION, param_idx)
                )
                stats = estimate_gradient_variance(
                    G=G,
                    data=data,
                    param_idx=param_idx,
                    theta_seeds=theta_list,
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
                    "dataset_metadata": dataset_metadata,
                    **stats,
                    **kernel_params,
                    **anti_concentration_summary,
                    "manifest_path": str(manifest_path),
                }
                fout.write(json.dumps(record) + "\n")

            done += 1
            log.info(
                "[%s/%s] family=%s kernel=%s init=%s n=%s",
                done,
                total,
                family,
                kernel,
                init_scheme,
                n,
            )


# Backwards-compatible alias: resolve_scaling_settings was moved to
# config.resolve_experiment_grid. Re-export it here so existing callers and
# tests continue to work without change.
resolve_scaling_settings = resolve_experiment_grid


def _compute_m(n: int, formula: str) -> int:
    import math

    if isinstance(formula, int):
        return formula
    return max(1, int(eval(formula, {"n": n, "log": math.log})))


def _extract_family_kwargs(
    family: str,
    circuit_cfg: dict,
    *,
    er_p_edge: float | None = None,
) -> dict:
    """Return family-specific constructor kwargs extracted from the circuit config."""
    if family == "bounded_degree":
        return circuit_cfg.get("bounded_degree", {})
    if family == "erdos_renyi":
        default_p = circuit_cfg.get("erdos_renyi", {}).get("p_edge", 0.1)
        if isinstance(default_p, list):
            default_p = default_p[0]
        return {"p_edge": er_p_edge if er_p_edge is not None else default_p}
    if family == "lattice":
        return {
            "dimension": circuit_cfg.get("lattice", {}).get("dimension", 1),
            "range_": circuit_cfg.get("lattice", {}).get("range", 1),
        }
    if family == "dense":
        return {"expected_weight": circuit_cfg.get("dense", {}).get("expected_weight", 0.5)}
    if family == "community":
        return circuit_cfg.get("community", {})
    if family == "symmetric":
        return {"parity": circuit_cfg.get("symmetric", {}).get("parity", "even")}
    return {}


def _make_G(
    family,
    n,
    m,
    circuit_cfg,
    rng,
    *,
    er_p_edge: float | None = None,
):
    kwargs = _extract_family_kwargs(family, circuit_cfg, er_p_edge=er_p_edge)
    return make_hypergraph(family=family, n=n, m=m, rng=rng, **kwargs)


def _make_model(
    family: str,
    n: int,
    m: int,
    circuit_cfg: dict,
    rng,
    rng_seed: int | None = None,
    *,
    er_p_edge: float | None = None,
) -> IQPModel:
    """Build an :class:`IQPModel` via :meth:`IQPModel.from_family`, preserving provenance."""
    kwargs = _extract_family_kwargs(family, circuit_cfg, er_p_edge=er_p_edge)
    return IQPModel.from_family(
        family=family, n=n, m=m, rng=rng, rng_seed=rng_seed, **kwargs
    )



def make_hybrid_exact_theta(G, data, scale=0.1):
    """
    Recipe A: Initializes weight-1 generators to match single-qubit data averages exactly.
    Uses Recipe B (parity scaling) for all other complex generators.
    """
    import numpy as np
    from iqp_bp.mmd.mixture import dataset_expectations_batch
    
    m, n = G.shape
    theta = np.zeros(m, dtype=np.float64)
    
    # Calculate the standard lightweight recipe (Recipe B) as a baseline for complex gates
    base_theta = scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
    
    # Calculate the exact feature means across the dataset
    feature_means = data.mean(axis=0)
    
    for i in range(m):
        row = G[i]
        
        # Check if the generator is a 1-qubit gate (Hamming weight == 1)
        if np.sum(row) == 1:
            qubit_index = np.argmax(row)
            target_mean = np.clip(feature_means[qubit_index], 0.0, 1.0) 
            
            # Recipe A math: arcsin(sqrt(mean))
            theta[i] = np.arcsin(np.sqrt(target_mean))
        else:
            # Fall back to Recipe B for 2-qubit or higher gates
            theta[i] = base_theta[i]
            
    return theta



def make_layer_wise_theta(G, data, num_layers, scheme="data_dependent", scale=0.1):
    m = G.shape[0]
    total_params = m * num_layers
    theta = np.zeros(total_params, dtype=np.float64)
    
    # if scheme == "data_dependent_exact":
    #     layer_1_init = make_hybrid_exact_theta(G, data, scale)
    # else:
    from iqp_bp.mmd.mixture import dataset_expectations_batch
    layer_1_init = scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
    
    theta[0:m] = layer_1_init
    return theta


###### make_theta function used for layer-wise training

def _make_theta(
    *,
    init_scheme: str,
    G: np.ndarray,
    data: np.ndarray,
    init_cfg: dict[str, Any],
    seed: int,
    small_angle_std: float | None = None,
    param_init_file: str | None = None,  
) -> np.ndarray:
    from iqp_bp.mmd.mixture import dataset_expectations_batch
    import numpy as np

    m = G.shape[0]
    
    # --- Layer by layer LOGIC ---
    if param_init_file is not None:
        print(f"SYSTEM: Loading file {param_init_file} for Variance Check")
        data_archive = np.load(param_init_file)
        
        if "theta" in data_archive.files:
            loaded_params = data_archive["theta"]
        else:
            print(f"GREEDY SYSTEM WARNING: Keys found -> {data_archive.files}")
            for key in data_archive.files:
                if len(data_archive[key].shape) == 1:
                    loaded_params = data_archive[key]
                    break
                    
        if loaded_params.size < m:
            padding = np.zeros(m - loaded_params.size)
            padded_theta = np.concatenate([loaded_params, padding])
            print(f"GREEDY SYSTEM: Padded array from {loaded_params.size} to {m} parameters.")
            return padded_theta
        return loaded_params
    # ----------------------------

    rng = np.random.default_rng(seed)
    if init_scheme == "uniform":
        low = init_cfg.get("uniform", {}).get("low", -np.pi)
        high = init_cfg.get("uniform", {}).get("high", np.pi)
        return rng.uniform(low, high, size=m)
    if init_scheme == "identity":
         return np.zeros(m)
    if init_scheme == "small_angle":
        std = small_angle_std
        if std is None:
            std = init_cfg.get("small_angle", {}).get("std", [0.1])
            std = std[0] if isinstance(std, list) else std
        return rng.normal(0.0, float(std), size=m)
    if init_scheme == "data_dependent":
        scale = float(init_cfg.get("data_dependent", {}).get("scale", 0.1))
        return scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
    raise ValueError(f"unknown init scheme {init_scheme!r}")


# def _make_theta(
#     *,
#     init_scheme,
#     G,
#     data,
#     init_cfg,
#     seed,
#     small_angle_std: float | None = None,
# ):
#     m = G.shape[0]
#     rng = np.random.default_rng(seed)
    
#     if init_scheme == "uniform":
#         lo = init_cfg.get("uniform", {}).get("low", -np.pi)
#         hi = init_cfg.get("uniform", {}).get("high", np.pi)
#         return rng.uniform(lo, hi, size=m)
        
#     if init_scheme == "small_angle":
#         std = small_angle_std
#         if std is None:
#             std = init_cfg.get("small_angle", {}).get("std", [0.1])
#             std = std[0] if isinstance(std, list) else std
#         return rng.normal(0, float(std), size=m)
    
#     if init_scheme == "identity":
#         return np.zeros(m)
        
#     if init_scheme == "data_dependent":
#         dd_cfg = init_cfg.get("data_dependent", {})
#         scale = dd_cfg.get("scale", 0.1)
#         from iqp_bp.mmd.mixture import dataset_expectations_batch
#         return scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)

#     if init_scheme == "layer_wise":
#         lw_cfg = init_cfg.get("layer_wise", {})
#         scale = lw_cfg.get("scale", 0.1)
#         num_layers = lw_cfg.get("num_layers", 2)
#         return make_layer_wise_theta(G, data, num_layers, scale)
    
#     # if init_scheme == "data_dependent_exact":
#     #     dd_exact_cfg = init_cfg.get("data_dependent_exact", {})
#     #     scale = dd_exact_cfg.get("scale", 0.1)
#     #     return make_hybrid_exact_theta(G, data, scale)

#     raise ValueError(f"Unknown init scheme {init_scheme!r}")


def _get_kernel_params(kernel, kernel_cfg, *, bandwidth: float | None = None):
    sigma = bandwidth
    if sigma is None:
        bw = kernel_cfg.get("bandwidth", [1.0])
        sigma = bw[0] if isinstance(bw, list) else bw
    sigma = float(sigma)
    if kernel == "gaussian":
        return {"sigma": sigma}
    if kernel == "laplacian":
        return {"sigma": sigma}
    if kernel == "multi_scale_gaussian":
        msg = kernel_cfg.get("multi_scale_gaussian", {})
        return {"sigmas": msg.get("sigmas", [sigma]), "weights": msg.get("weights")}
    if kernel == "polynomial":
        poly = kernel_cfg.get("polynomial", {})
        return {"degree": poly.get("degree", 2), "constant": poly.get("constant", 1.0)}
    if kernel == "linear":
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
    artifact_dir: Path,
    artifact_stem: str,
    provenance: dict[str, Any],
    model_provenance: dict[str, Any] | None = None,
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

    model = IQPModel(G=G, theta=theta_list[0], provenance=model_provenance or {})
    result = evaluate_anti_concentration_from_model(
        model,
        provenance={
            "source": "scaling_seed",
            "theta_seed_index": 0,
            **provenance,
        },
        max_qubits=max_n,
        alphas=alphas,
        primary_alpha=primary_alpha,
        beta_min=beta_min,
        second_moment_threshold=second_moment_threshold,
    )

    artifact_paths = write_anti_concentration_artifacts(
        result,
        output_dir=artifact_dir,
        stem=artifact_stem,
    )

    checkpoint_path = None
    if export_checkpoint:
        # model.provenance (family, n, m_requested, m_generated, rng_seed, …) is
        # serialized automatically by save_iqp_checkpoint as provenance_json.
        # We only add experiment-level fields not already captured in provenance.
        checkpoint_path = save_iqp_checkpoint(
            model,
            checkpoints_dir / _checkpoint_name(family=family, init_scheme=init_scheme, kernel=kernel, n=n),
            metadata={
                "theta_seed_index": 0,
                "source": "run_scaling",
                "kernel": kernel,
                "init_scheme": init_scheme,
                "dataset_metadata_json": json.dumps(
                    provenance.get("dataset_metadata", {}), sort_keys=True
                ),
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
        "ac_passes_second_moment_threshold": bool(result["passes_second_moment_threshold"]),
        "ac_max_probability_scaled": float(result["max_probability_scaled"]),
        "ac_beta_hat_by_alpha": threshold_beta_by_alpha,
        "ac_summary_path": str(artifact_paths["summary_path"]),
        "ac_thresholds_path": str(artifact_paths["thresholds_path"]),
        "ac_threshold_plot_path": str(artifact_paths["threshold_plot_path"]),
        "ac_diagnostics_plot_path": str(artifact_paths["diagnostics_plot_path"]),
    }
    if checkpoint_path is not None:
        summary["ac_checkpoint_path"] = str(checkpoint_path)
    return summary


def _checkpoint_name(*, family: str, init_scheme: str, kernel: str, n: int) -> str:
    """Build a stable filename for a saved anti-concentration checkpoint."""
    return f"{family}_n{n}_{kernel}_{init_scheme}_seed0.npz"




def _record_setting_fields(setting: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "family": setting["family"],
        "kernel": setting["kernel"],
        "init": setting["init_scheme"],
        "n": int(setting["n"]),
        "dataset_type": setting["dataset_type"],
    }
    if setting.get("bandwidth") is not None:
        fields["bandwidth"] = float(setting["bandwidth"])
    if setting.get("small_angle_std") is not None:
        fields["small_angle_std"] = float(setting["small_angle_std"])
    if setting.get("er_p_edge") is not None:
        fields["er_p_edge"] = float(setting["er_p_edge"])
    return fields


def _setting_identity(setting: dict[str, Any]) -> dict[str, Any]:
    return _record_setting_fields(setting)


def _setting_stem(setting: dict[str, Any]) -> str:
    parts = [
        str(setting["family"]),
        f"n{int(setting['n'])}",
        str(setting["kernel"]),
        str(setting["init_scheme"]),
        str(setting["dataset_type"]),
    ]
    if setting.get("bandwidth") is not None:
        parts.append(f"sigma{_format_scalar(setting['bandwidth'])}")
    if setting.get("small_angle_std") is not None:
        parts.append(f"theta{_format_scalar(setting['small_angle_std'])}")
    if setting.get("er_p_edge") is not None:
        parts.append(f"er{_format_scalar(setting['er_p_edge'])}")
    return "__".join(parts)


def _format_scalar(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")
