"""Training runner for learned-distribution anti-concentration experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.config import persist_experiment_manifest, resolve_experiment_grid
from iqp_bp.distributions import draw_samples_from_probability_vector, summarize_by_order
from iqp_bp.experiments.data_factory import make_dataset
from iqp_bp.experiments.marginal_artifacts import write_marginal_summary
from iqp_bp.experiments.run_validation import (
    evaluate_anti_concentration_from_probabilities,
    evaluate_anti_concentration_from_samples,
)
from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.iqp.model import IQPModel
from iqp_bp.rng import STREAM_THETA, derive_seed, experiment_stream_bundle
from iqp_bp.training import Trainer

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the learned-distribution training sweep defined by the config."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = resolve_experiment_grid(cfg)
    _, manifest_path = persist_experiment_manifest(cfg, settings)
    summary_path = output_dir / "results.jsonl"
    training_cfg = cfg.get("training", {})

    summaries: list[dict[str, Any]] = []
    with open(summary_path, "w", encoding="utf-8") as handle:
        for setting in settings:
            setting_key = _record_setting_fields(setting)
            setting_stem = _setting_stem(setting)
            run_dir = output_dir / "runs" / setting_stem
            run_dir.mkdir(parents=True, exist_ok=True)

            streams = experiment_stream_bundle(
                int(cfg["experiment"]["seed"]),
                "run_training",
                setting_key,
            )
            n = int(setting["n"])
            m_requested = _compute_m(n, cfg["circuit"]["n_generators"])
            model = _make_model(
                family=str(setting["family"]),
                n=n,
                m=m_requested,
                circuit_cfg=cfg["circuit"],
                rng=np.random.default_rng(streams["circuit"]),
                rng_seed=streams["circuit"],
                er_p_edge=setting.get("er_p_edge"),
            )
            data, dataset_metadata = make_dataset(
                cfg["dataset"],
                n=n,
                seed=streams["data"],
            )
            model.theta = _make_theta(
                init_scheme=str(setting["init_scheme"]),
                G=model.G,
                data=data,
                init_cfg=cfg["init"],
                seed=derive_seed(streams[STREAM_THETA], "init"),
                small_angle_std=setting.get("small_angle_std"),
            )

            kernel_params = _get_kernel_params(
                kernel=str(setting["kernel"]),
                kernel_cfg=cfg["kernel"],
                bandwidth=setting.get("bandwidth"),
            )

            callback = _make_checkpoint_callback(
                run_dir=run_dir,
                kernel=str(setting["kernel"]),
                kernel_params=kernel_params,
                training_cfg=training_cfg,
                provenance={
                    **setting_key,
                    "dataset_metadata": dataset_metadata,
                    "manifest_path": str(manifest_path),
                },
            )
            trainer = Trainer(
                model=model,
                data=data,
                output_dir=run_dir,
                kernel=str(setting["kernel"]),
                kernel_params=kernel_params,
                optimizer=str(training_cfg.get("optimizer", "adam")),
                lr=float(training_cfg.get("lr", 0.05)),
                num_steps=int(training_cfg.get("num_steps", 100)),
                checkpoint_every=int(training_cfg.get("checkpoint_every", 10)),
                num_a_samples=int(cfg["estimation"]["num_a_samples"]),
                num_z_samples=int(cfg["estimation"]["num_z_samples"]),
                batch_size=cfg["estimation"].get("batch_size"),
                stream_seeds={
                    "callback": derive_seed(int(cfg["experiment"]["seed"]), "run_training", setting_key, "callback"),
                    "kernel": streams["kernel"],
                    "estimation": streams["estimation"],
                },
                loss_mode=str(training_cfg.get("loss_mode", "auto")),
                exact_loss_max_n=int(training_cfg.get("exact_loss_max_n", 12)),
                checkpoint_callback=callback,
            )
            result = trainer.run()
            summary = {
                **setting_key,
                "run_dir": str(run_dir),
                "trajectory_path": result["trajectory_path"],
                "checkpoint_dir": result["checkpoint_dir"],
                "final_loss": result["final_loss"],
                "written_steps": result["written_steps"],
                "dataset_metadata": dataset_metadata,
                "manifest_path": str(manifest_path),
            }
            handle.write(json.dumps(summary) + "\n")
            summaries.append(summary)
            log.info(
                "Training complete: family=%s kernel=%s init=%s n=%s trajectory=%s",
                setting["family"],
                setting["kernel"],
                setting["init_scheme"],
                setting["n"],
                result["trajectory_path"],
            )

    return summaries


def _make_checkpoint_callback(
    *,
    run_dir: Path,
    kernel: str,
    kernel_params: dict[str, Any],
    training_cfg: dict[str, Any],
    provenance: dict[str, Any],
):
    diagnostics_cfg = training_cfg.get("diagnostics", {})
    enabled = bool(diagnostics_cfg.get("enabled", True))
    distribution_mode = str(diagnostics_cfg.get("distribution_mode", "auto"))
    exact_probability_max_n = int(diagnostics_cfg.get("exact_probability_max_n", 12))
    sample_from_exact_probability_max_n = int(
        diagnostics_cfg.get("sample_from_exact_probability_max_n", 20)
    )
    sample_count = int(diagnostics_cfg.get("sample_count", 4096))
    max_order = diagnostics_cfg.get("max_order")
    max_subsets_per_order = diagnostics_cfg.get("max_subsets_per_order")
    ac_cfg = diagnostics_cfg.get("anti_concentration", {})
    alphas = tuple(float(alpha) for alpha in ac_cfg.get("alphas", (0.5, 1.0, 2.0)))
    primary_alpha = float(ac_cfg.get("primary_alpha", 1.0))
    beta_min = float(ac_cfg.get("beta_min", 0.25))
    second_moment_threshold = float(ac_cfg.get("second_moment_threshold", 1.0))

    def callback(
        step: int,
        model: IQPModel,
        data: np.ndarray,
        loss_details: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if not enabled:
            return {"diagnostics_available": False, "diagnostics_reason": "disabled"}

        n = model.n
        orders = list(range(1, (n if max_order is None else min(int(max_order), n)) + 1))
        row: dict[str, Any]
        sigma = float(kernel_params["sigma"]) if kernel == "gaussian" and "sigma" in kernel_params else None

        if distribution_mode in {"auto", "exact"} and n <= exact_probability_max_n:
            probabilities = model.probability_vector_exact(max_qubits=exact_probability_max_n)
            ac_result = evaluate_anti_concentration_from_probabilities(
                probabilities,
                provenance={"source": "training_exact", **provenance, "step": int(step)},
                mode="exact_probabilities",
                alphas=alphas,
                primary_alpha=primary_alpha,
                beta_min=beta_min,
                second_moment_threshold=second_moment_threshold,
            )
            marginal_summary = summarize_by_order(
                data,
                probabilities,
                n=n,
                orders=orders,
                max_subsets_per_order=max_subsets_per_order,
                rng=rng,
                sigma=sigma,
            )
            row = {
                "diagnostics_available": True,
                "distribution_mode": "exact",
                **_flatten_ac(ac_result),
            }
        elif distribution_mode in {"auto", "sample"} and n <= sample_from_exact_probability_max_n:
            probabilities = model.probability_vector_exact(
                max_qubits=sample_from_exact_probability_max_n
            )
            model_samples = draw_samples_from_probability_vector(
                probabilities,
                sample_count,
                rng=rng,
            )
            ac_result = evaluate_anti_concentration_from_samples(
                model_samples,
                provenance={"source": "training_sampled", **provenance, "step": int(step)},
                alphas=alphas,
                primary_alpha=primary_alpha,
                beta_min=beta_min,
                second_moment_threshold=second_moment_threshold,
            )
            marginal_summary = summarize_by_order(
                data,
                model_samples,
                n=n,
                orders=orders,
                max_subsets_per_order=max_subsets_per_order,
                rng=rng,
                sigma=sigma,
            )
            row = {
                "diagnostics_available": True,
                "distribution_mode": "sample",
                "distribution_sample_count": sample_count,
                **_flatten_ac(ac_result),
            }
        else:
            return {
                "diagnostics_available": False,
                "diagnostics_reason": (
                    f"unsupported_mode:{distribution_mode}:n={n}"
                ),
            }

        summary_path = write_marginal_summary(
            marginal_summary,
            output_dir=run_dir / "marginals",
            stem=f"step_{step:04d}",
        )
        row.update(
            {
                "marginal_summary_path": str(summary_path),
                "marginal_weighted_mmd2_total": marginal_summary["weighted_mmd2_total"],
                "marginal_orders": marginal_summary["orders"],
            }
        )
        return row

    return callback


def _flatten_ac(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ac_mode": result["mode"],
        "ac_primary_alpha": float(result["primary_alpha"]),
        "ac_primary_beta_hat": float(result["primary_beta_hat"]),
        "ac_passes_primary_threshold": bool(result["passes_primary_threshold"]),
        "ac_scaled_second_moment": float(result["scaled_second_moment"]),
        "ac_passes_second_moment_threshold": bool(result["passes_second_moment_threshold"]),
        "ac_max_probability_scaled": float(result["max_probability_scaled"]),
        "ac_beta_hat_by_alpha": {
            str(entry["alpha"]): float(entry["beta_hat"]) for entry in result["threshold_checks"]
        },
    }


def _compute_m(n: int, formula: str | int) -> int:
    import math

    if isinstance(formula, int):
        return formula
    return max(1, int(eval(formula, {"n": n, "log": math.log})))


def _extract_family_kwargs(
    family: str,
    circuit_cfg: dict[str, Any],
    *,
    er_p_edge: float | None = None,
) -> dict[str, Any]:
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


def _make_model(
    family: str,
    n: int,
    m: int,
    circuit_cfg: dict[str, Any],
    rng: np.random.Generator,
    rng_seed: int | None = None,
    *,
    er_p_edge: float | None = None,
) -> IQPModel:
    kwargs = _extract_family_kwargs(family, circuit_cfg, er_p_edge=er_p_edge)
    G = make_hypergraph(family=family, n=n, m=m, rng=rng, **kwargs)
    provenance = {
        "family": family,
        "n": int(n),
        "m_requested": int(m),
        "m_generated": int(G.shape[0]),
        "family_kwargs": kwargs,
    }
    if rng_seed is not None:
        provenance["rng_seed"] = int(rng_seed)
    return IQPModel(G=G, provenance=provenance)


def _make_theta(
    *,
    init_scheme: str,
    G: np.ndarray,
    data: np.ndarray,
    init_cfg: dict[str, Any],
    seed: int,
    small_angle_std: float | None = None,
) -> np.ndarray:
    from iqp_bp.mmd.mixture import dataset_expectations_batch

    m = G.shape[0]
    rng = np.random.default_rng(seed)
    if init_scheme == "uniform":
        low = init_cfg.get("uniform", {}).get("low", -np.pi)
        high = init_cfg.get("uniform", {}).get("high", np.pi)
        return rng.uniform(low, high, size=m)
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


def _get_kernel_params(
    kernel: str,
    kernel_cfg: dict[str, Any],
    *,
    bandwidth: float | None = None,
) -> dict[str, Any]:
    sigma = bandwidth
    if sigma is None:
        default_bandwidth = kernel_cfg.get("bandwidth", [1.0])
        sigma = default_bandwidth[0] if isinstance(default_bandwidth, list) else default_bandwidth
    sigma = float(sigma)
    if kernel in {"gaussian", "laplacian"}:
        return {"sigma": sigma}
    if kernel == "multi_scale_gaussian":
        msg = kernel_cfg.get("multi_scale_gaussian", {})
        return {"sigmas": msg.get("sigmas", [sigma]), "weights": msg.get("weights")}
    if kernel == "polynomial":
        poly = kernel_cfg.get("polynomial", {})
        return {"degree": poly.get("degree", 2), "constant": poly.get("constant", 1.0)}
    return {}


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
