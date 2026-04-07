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
from iqp_bp.rng import derive_seed, named_seed_streams

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

    settings = resolve_scaling_settings(cfg)
    total = len(settings)
    done = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for setting in settings:
            setting_key = _setting_identity(setting)
            streams = named_seed_streams(
                base_seed,
                ("circuit", "data"),
                "run_scaling",
                setting_key,
            )
            n = int(setting["n"])
            family = str(setting["family"])
            kernel = str(setting["kernel"])
            init_scheme = str(setting["init_scheme"])

            m = _compute_m(n, cfg["circuit"]["n_generators"])
            G = _make_G(
                family=family,
                n=n,
                m=m,
                circuit_cfg=cfg["circuit"],
                rng=np.random.default_rng(streams["circuit"]),
                er_p_edge=setting.get("er_p_edge"),
            )
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
                    seed=derive_seed(base_seed, "run_scaling", setting_key, "theta", idx),
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
            )

            for param_idx in range(min(5, actual_m)):
                rng_est = np.random.default_rng(
                    derive_seed(base_seed, "run_scaling", setting_key, "estimation", param_idx)
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


def resolve_scaling_settings(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the explicit scalar experiment grid for the scaling runner."""
    families = _as_list(cfg["circuit"]["family"])
    n_qubits_list = _as_list(cfg["circuit"]["n_qubits"])
    kernels = _as_list(cfg["kernel"]["type"])
    init_schemes = _as_list(cfg["init"]["scheme"])

    settings: list[dict[str, Any]] = []
    for family, kernel, init_scheme, n in product(
        families,
        kernels,
        init_schemes,
        n_qubits_list,
    ):
        for bandwidth, er_p_edge, small_angle_std in product(
            _bandwidth_values_for_kernel(kernel, cfg["kernel"]),
            _erdos_renyi_values_for_family(family, cfg["circuit"]),
            _small_angle_values_for_init(init_scheme, cfg["init"]),
        ):
            settings.append(
                {
                    "family": family,
                    "kernel": kernel,
                    "init_scheme": init_scheme,
                    "n": int(n),
                    "bandwidth": bandwidth,
                    "er_p_edge": er_p_edge,
                    "small_angle_std": small_angle_std,
                    "dataset_type": str(cfg["dataset"].get("type", "product_bernoulli")),
                }
            )
    return settings


def _as_list(val):
    return val if isinstance(val, list) else [val]


def _compute_m(n: int, formula: str) -> int:
    import math

    if isinstance(formula, int):
        return formula
    return max(1, int(eval(formula, {"n": n, "log": math.log})))


def _make_G(
    family,
    n,
    m,
    circuit_cfg,
    rng,
    *,
    er_p_edge: float | None = None,
):
    kwargs = {}
    if family == "bounded_degree":
        kwargs = circuit_cfg.get("bounded_degree", {})
    elif family == "erdos_renyi":
        default_p = circuit_cfg.get("erdos_renyi", {}).get("p_edge", 0.1)
        if isinstance(default_p, list):
            default_p = default_p[0]
        kwargs = {"p_edge": er_p_edge if er_p_edge is not None else default_p}
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


def _make_theta(
    *,
    init_scheme,
    G,
    data,
    init_cfg,
    seed,
    small_angle_std: float | None = None,
):
    m = G.shape[0]
    rng = np.random.default_rng(seed)
    if init_scheme == "uniform":
        lo = init_cfg.get("uniform", {}).get("low", -np.pi)
        hi = init_cfg.get("uniform", {}).get("high", np.pi)
        return rng.uniform(lo, hi, size=m)
    if init_scheme == "small_angle":
        std = small_angle_std
        if std is None:
            std = init_cfg.get("small_angle", {}).get("std", [0.1])
            std = std[0] if isinstance(std, list) else std
        return rng.normal(0, float(std), size=m)
    if init_scheme == "data_dependent":
        dd_cfg = init_cfg.get("data_dependent", {})
        scale = dd_cfg.get("scale", 0.1)
        return scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
    raise ValueError(f"Unknown init scheme {init_scheme!r}")


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
        checkpoint_metadata = {
            key: value
            for key, value in provenance.items()
            if key != "dataset_metadata"
        }
        if "dataset_metadata" in provenance:
            checkpoint_metadata["dataset_metadata_json"] = json.dumps(
                provenance["dataset_metadata"],
                sort_keys=True,
            )
        checkpoint_path = save_iqp_checkpoint(
            model,
            checkpoints_dir / _checkpoint_name(family=family, init_scheme=init_scheme, kernel=kernel, n=n),
            metadata={
                "theta_seed_index": 0,
                "source": "run_scaling",
                **checkpoint_metadata,
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


def _bandwidth_values_for_kernel(kernel: str, kernel_cfg: dict[str, Any]) -> list[float | None]:
    if kernel in {"gaussian", "laplacian"}:
        return [float(value) for value in _as_list(kernel_cfg.get("bandwidth", [1.0]))]
    return [None]


def _erdos_renyi_values_for_family(
    family: str,
    circuit_cfg: dict[str, Any],
) -> list[float | None]:
    if family == "erdos_renyi":
        return [
            float(value)
            for value in _as_list(circuit_cfg.get("erdos_renyi", {}).get("p_edge", [0.1]))
        ]
    return [None]


def _small_angle_values_for_init(
    init_scheme: str,
    init_cfg: dict[str, Any],
) -> list[float | None]:
    if init_scheme == "small_angle":
        return [float(value) for value in _as_list(init_cfg.get("small_angle", {}).get("std", [0.1]))]
    return [None]


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
