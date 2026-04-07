"""Deterministic validation helpers for output-distribution diagnostics.

This module now hosts the exact small-n anti-concentration checker and the
runner that turns exact probabilities, empirical bitstring samples, or a small
deterministic IQP checkpoint into serialized validation artifacts.

Why this exists:

- the trainability experiments already answer a gradient question
- anti-concentration is a distribution-shape question
- for small n we want exact probabilities when possible, and clearly labeled
  sample-histogram diagnostics otherwise
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.iqp.model import IQPModel

log = logging.getLogger(__name__)


DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_PRIMARY_ALPHA = 1.0
DEFAULT_BETA_MIN = 0.25
DEFAULT_SECOND_MOMENT_THRESHOLD = 1.0


def _infer_num_qubits(num_probabilities: int) -> int:
    """Infer n from a probability vector of length 2**n."""
    if num_probabilities <= 0:
        raise ValueError("Probability vector must be non-empty.")
    n = int(round(math.log2(num_probabilities)))
    if 2**n != num_probabilities:
        raise ValueError(
            "Probability vector length must be an exact power of two; "
            f"got length={num_probabilities}."
        )
    return n


def _coerce_probability_vector(
    probabilities: np.ndarray | list[float],
    *,
    atol: float,
) -> tuple[np.ndarray, int]:
    """Validate and normalize the exact probability vector input contract."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"Probability vector must be 1D; got shape={p.shape}.")
    if not np.all(np.isfinite(p)):
        raise ValueError("Probability vector must contain only finite values.")
    if np.any(p < -atol):
        raise ValueError("Probability vector cannot contain negative entries.")

    p = np.where(np.abs(p) <= atol, 0.0, p)
    n = _infer_num_qubits(len(p))
    total = float(p.sum())
    if not np.isclose(total, 1.0, atol=atol):
        raise ValueError(
            "Probability vector must sum to 1 within tolerance; "
            f"got total={total:.16f}, atol={atol}."
        )
    return p, n


def _beta_hat(probabilities: np.ndarray, alpha: float, n: int) -> float:
    """Compute the finite-n threshold statistic beta_hat(alpha)."""
    threshold = alpha * (2.0 ** (-n))
    return float(np.mean(probabilities >= threshold))


def check_anti_concentration(
    probabilities: np.ndarray | list[float],
    *,
    alphas: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    primary_alpha: float = DEFAULT_PRIMARY_ALPHA,
    beta_min: float = DEFAULT_BETA_MIN,
    second_moment_threshold: float = DEFAULT_SECOND_MOMENT_THRESHOLD,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate anti-concentration diagnostics on an exact probability vector.

    Args:
        probabilities:
            Full probability vector `p` over `{0,1}^n`, with length `2**n`.
        alphas:
            Fixed threshold multipliers used for the interpretable diagnostic
            `beta_hat(alpha) = 2^{-n} |{x : p(x) >= alpha 2^{-n}}|`.
        primary_alpha:
            The alpha value whose threshold boolean is surfaced as the default
            pass/fail decision.
        beta_min:
            Minimum acceptable `beta_hat(primary_alpha)` threshold.
        second_moment_threshold:
            Minimum acceptable scaled second moment `2^n sum_x p(x)^2`.
        atol:
            Tolerance for normalization and tiny negative roundoff.

    Returns:
        A JSON-serializable dictionary containing the exact scalar check,
        threshold diagnostics, and pass/fail booleans.
    """
    p, n = _coerce_probability_vector(probabilities, atol=atol)
    alpha_grid = tuple(float(alpha) for alpha in alphas)
    if any(alpha <= 0.0 for alpha in alpha_grid):
        raise ValueError(f"All alpha values must be positive; got {alpha_grid}.")
    if primary_alpha not in alpha_grid:
        alpha_grid = tuple(sorted(set(alpha_grid + (float(primary_alpha),))))

    num_outcomes = len(p)
    uniform_probability = 2.0 ** (-n)
    collision_probability = float(np.sum(p**2))
    scaled_second_moment = float((2.0**n) * collision_probability)
    max_probability = float(np.max(p))
    max_probability_scaled = float((2.0**n) * max_probability)
    effective_support = float(np.inf if collision_probability == 0.0 else 1.0 / collision_probability)
    effective_support_ratio = float(effective_support / num_outcomes)

    threshold_checks = []
    for alpha in alpha_grid:
        beta_hat = _beta_hat(p, alpha, n)
        threshold_checks.append(
            {
                "alpha": float(alpha),
                "threshold_probability": float(alpha * uniform_probability),
                "beta_hat": beta_hat,
                "passes_beta_threshold": bool(beta_hat >= beta_min - atol),
            }
        )

    threshold_by_alpha = {entry["alpha"]: entry for entry in threshold_checks}
    primary_check = threshold_by_alpha[float(primary_alpha)]

    return {
        "n": n,
        "num_outcomes": num_outcomes,
        "uniform_probability": float(uniform_probability),
        "collision_probability": collision_probability,
        "scaled_second_moment": scaled_second_moment,
        "second_moment_threshold": float(second_moment_threshold),
        "passes_second_moment_threshold": bool(
            scaled_second_moment >= second_moment_threshold - atol
        ),
        "max_probability": max_probability,
        "max_probability_scaled": max_probability_scaled,
        "effective_support": effective_support,
        "effective_support_ratio": effective_support_ratio,
        "beta_min": float(beta_min),
        "primary_alpha": float(primary_alpha),
        "primary_beta_hat": float(primary_check["beta_hat"]),
        "passes_primary_threshold": bool(primary_check["passes_beta_threshold"]),
        "threshold_checks": threshold_checks,
    }


def samples_to_probability_vector(samples: np.ndarray | list[list[int]]) -> np.ndarray:
    """Convert binary bitstring samples into an empirical probability vector.

    This is the secondary diagnostic path. It does not produce an exact circuit
    distribution, only the histogram induced by the provided bitstrings.
    """
    sample_array = np.asarray(samples, dtype=np.uint8)
    if sample_array.ndim != 2:
        raise ValueError(f"Samples must be a 2D array; got shape={sample_array.shape}.")
    if sample_array.shape[0] == 0:
        raise ValueError("Samples array must contain at least one bitstring.")
    if not np.all((sample_array == 0) | (sample_array == 1)):
        raise ValueError("Samples must be binary with entries in {0, 1}.")

    n = sample_array.shape[1]
    bit_weights = 2 ** np.arange(n - 1, -1, -1, dtype=np.uint64)
    bitstring_indices = sample_array.astype(np.uint64) @ bit_weights
    counts = np.bincount(bitstring_indices, minlength=2**n).astype(np.float64)
    probabilities = counts / counts.sum()
    return probabilities


def evaluate_anti_concentration_from_probabilities(
    probabilities: np.ndarray | list[float],
    *,
    provenance: dict[str, Any] | None = None,
    mode: str = "exact_probabilities",
    alphas: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    primary_alpha: float = DEFAULT_PRIMARY_ALPHA,
    beta_min: float = DEFAULT_BETA_MIN,
    second_moment_threshold: float = DEFAULT_SECOND_MOMENT_THRESHOLD,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Run anti-concentration checks on a probability vector and attach provenance."""
    result = check_anti_concentration(
        probabilities,
        alphas=alphas,
        primary_alpha=primary_alpha,
        beta_min=beta_min,
        second_moment_threshold=second_moment_threshold,
        atol=atol,
    )
    result["mode"] = mode
    result["provenance"] = provenance or {}
    return result


def evaluate_anti_concentration_from_samples(
    samples: np.ndarray | list[list[int]],
    *,
    provenance: dict[str, Any] | None = None,
    alphas: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    primary_alpha: float = DEFAULT_PRIMARY_ALPHA,
    beta_min: float = DEFAULT_BETA_MIN,
    second_moment_threshold: float = DEFAULT_SECOND_MOMENT_THRESHOLD,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Histogram samples first, then run the anti-concentration checker."""
    probabilities = samples_to_probability_vector(samples)
    return evaluate_anti_concentration_from_probabilities(
        probabilities,
        provenance=provenance,
        mode="empirical_histogram",
        alphas=alphas,
        primary_alpha=primary_alpha,
        beta_min=beta_min,
        second_moment_threshold=second_moment_threshold,
        atol=atol,
    )


def evaluate_anti_concentration_from_model(
    model: IQPModel,
    *,
    provenance: dict[str, Any] | None = None,
    max_qubits: int = 20,
    alphas: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    primary_alpha: float = DEFAULT_PRIMARY_ALPHA,
    beta_min: float = DEFAULT_BETA_MIN,
    second_moment_threshold: float = DEFAULT_SECOND_MOMENT_THRESHOLD,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Extract exact small-n probabilities from an IQP model, then validate them."""
    probabilities = model.probability_vector_exact(max_qubits=max_qubits)
    return evaluate_anti_concentration_from_probabilities(
        probabilities,
        provenance=provenance,
        mode="exact_iqp_model",
        alphas=alphas,
        primary_alpha=primary_alpha,
        beta_min=beta_min,
        second_moment_threshold=second_moment_threshold,
        atol=atol,
    )


def load_probability_vector(path: str | Path) -> np.ndarray:
    """Load a probability vector from `.npy`, `.json`, or delimited text."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return np.load(file_path)
    if suffix == ".json":
        with open(file_path, encoding="utf-8") as handle:
            return np.asarray(json.load(handle), dtype=np.float64)
    return np.loadtxt(file_path, delimiter=",", dtype=np.float64)


def load_bitstring_samples(path: str | Path) -> np.ndarray:
    """Load binary bitstring samples from `.npy` or delimited text."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return np.load(file_path)

    for delimiter in (",", None):
        try:
            return np.loadtxt(file_path, delimiter=delimiter, dtype=np.uint8)
        except ValueError:
            continue
    raise ValueError(f"Could not parse bitstring samples from {file_path}.")


def load_iqp_checkpoint(path: str | Path) -> tuple[IQPModel, dict[str, Any]]:
    """Load a deterministic-side IQP checkpoint containing `G` and `theta`.

    The supported checkpoint format is `.npz` with keys:

    - `G`: binary generator matrix
    - `theta`: parameter vector
    - optional metadata arrays such as `family`, `seed`, or `description`
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".npz":
        raise ValueError("IQP checkpoint loader currently supports `.npz` files only.")

    with np.load(file_path, allow_pickle=False) as checkpoint:
        if "G" not in checkpoint or "theta" not in checkpoint:
            raise ValueError("IQP checkpoint must contain `G` and `theta` arrays.")
        model = IQPModel(G=checkpoint["G"], theta=checkpoint["theta"])
        metadata: dict[str, Any] = {}
        for key in checkpoint.files:
            if key in {"G", "theta"}:
                continue
            value = checkpoint[key]
            metadata[key] = value.tolist() if hasattr(value, "tolist") else value
    return model, metadata


def save_iqp_checkpoint(
    model: IQPModel,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save an IQP model as a deterministic validation checkpoint.

    The checkpoint format is `.npz` with mandatory arrays:

    - `G`: binary generator matrix
    - `theta`: parameter vector

    and optional scalar/string metadata fields such as `family`, `seed`, or
    `description`.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "G": np.asarray(model.G, dtype=np.uint8),
        "theta": np.asarray(model.theta, dtype=np.float64),
    }
    for key, value in (metadata or {}).items():
        arrays[str(key)] = np.asarray(value)

    np.savez(checkpoint_path, **arrays)
    return checkpoint_path


def write_anti_concentration_artifacts(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
    stem: str = "anti_concentration",
) -> dict[str, Path]:
    """Write the validation summary to JSON and thresholds to CSV."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    summary_path = directory / f"{stem}.json"
    thresholds_path = directory / f"{stem}_thresholds.csv"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    with open(thresholds_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["alpha", "threshold_probability", "beta_hat", "passes_beta_threshold"],
        )
        writer.writeheader()
        for row in result["threshold_checks"]:
            writer.writerow(row)

    return {"summary_path": summary_path, "thresholds_path": thresholds_path}


def _theta_from_cfg(init_cfg: dict[str, Any], m: int, seed: int) -> np.ndarray:
    """Construct a deterministic theta vector from a validation config."""
    rng = np.random.default_rng(seed)
    scheme = init_cfg.get("scheme", "uniform")
    if scheme == "uniform":
        low = init_cfg.get("uniform", {}).get("low", -np.pi)
        high = init_cfg.get("uniform", {}).get("high", np.pi)
        return rng.uniform(low, high, size=m)
    if scheme == "small_angle":
        std = init_cfg.get("small_angle", {}).get("std", 0.1)
        if isinstance(std, list):
            std = std[0]
        return rng.normal(0.0, float(std), size=m)
    raise ValueError(f"Unsupported validation init scheme {scheme!r}.")


def _build_model_from_cfg(cfg: dict[str, Any]) -> tuple[IQPModel, dict[str, Any]]:
    """Build a small deterministic IQP model directly from config fields."""
    experiment_cfg = cfg.get("experiment", {})
    circuit_cfg = cfg.get("circuit", {})
    base_seed = int(experiment_cfg.get("seed", 0))

    family = circuit_cfg.get("family", "product_state")
    if isinstance(family, list):
        family = family[0]
    n_qubits = circuit_cfg.get("n_qubits", 4)
    if isinstance(n_qubits, list):
        n_qubits = n_qubits[0]
    n_generators = circuit_cfg.get("n_generators", n_qubits)
    if not isinstance(n_generators, int):
        n_generators = int(n_qubits)

    rng = np.random.default_rng(base_seed)
    family_kwargs: dict[str, Any] = {}
    if family == "lattice":
        family_kwargs = {
            "dimension": circuit_cfg.get("lattice", {}).get("dimension", 2),
            "range_": circuit_cfg.get("lattice", {}).get("range", 1),
        }
    elif family == "erdos_renyi":
        p_edge = circuit_cfg.get("erdos_renyi", {}).get("p_edge", 2.0)
        if isinstance(p_edge, list):
            p_edge = p_edge[0]
        family_kwargs = {"p_edge": p_edge}

    generator_matrix = make_hypergraph(
        family=family,
        n=int(n_qubits),
        m=int(n_generators),
        rng=rng,
        **family_kwargs,
    )
    theta = _theta_from_cfg(cfg.get("init", {}), generator_matrix.shape[0], base_seed)
    model = IQPModel(G=generator_matrix, theta=theta)
    provenance = {
        "source": "config_model",
        "family": family,
        "n": int(n_qubits),
        "seed": base_seed,
    }
    return model, provenance


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Entry point called by CLI.

    Supported validation sources:

    - `validation.checkpoint_path`: deterministic `.npz` checkpoint with `G` and `theta`
    - `validation.samples_path`: binary bitstring samples to be histogrammed
    - `validation.probabilities_path`: exact probability vector on disk
    - `validation.probabilities`: exact probability vector embedded in config
    - otherwise: build a small deterministic model directly from the config
    """
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_cfg = cfg.get("validation", {})
    alpha_grid = validation_cfg.get("alphas", list(DEFAULT_ALPHA_GRID))
    if not isinstance(alpha_grid, (list, tuple)):
        raise ValueError("validation.alphas must be a list or tuple of positive floats.")
    alpha_grid_tuple = tuple(float(alpha) for alpha in alpha_grid)

    primary_alpha = float(validation_cfg.get("primary_alpha", DEFAULT_PRIMARY_ALPHA))
    beta_min = float(validation_cfg.get("beta_min", DEFAULT_BETA_MIN))
    second_moment_threshold = float(
        validation_cfg.get("second_moment_threshold", DEFAULT_SECOND_MOMENT_THRESHOLD)
    )
    atol = float(validation_cfg.get("atol", 1e-12))
    stem = validation_cfg.get("output_stem", "anti_concentration")

    result: dict[str, Any]
    if "checkpoint_path" in validation_cfg:
        checkpoint_path = Path(validation_cfg["checkpoint_path"])
        model, checkpoint_metadata = load_iqp_checkpoint(checkpoint_path)
        provenance = {
            "source": "checkpoint",
            "checkpoint_path": str(checkpoint_path),
            **checkpoint_metadata,
        }
        result = evaluate_anti_concentration_from_model(
            model,
            provenance=provenance,
            max_qubits=int(validation_cfg.get("max_qubits", 20)),
            alphas=alpha_grid_tuple,
            primary_alpha=primary_alpha,
            beta_min=beta_min,
            second_moment_threshold=second_moment_threshold,
            atol=atol,
        )
    elif "samples_path" in validation_cfg:
        samples_path = Path(validation_cfg["samples_path"])
        samples = load_bitstring_samples(samples_path)
        provenance = {
            "source": "samples_path",
            "samples_path": str(samples_path),
        }
        result = evaluate_anti_concentration_from_samples(
            samples,
            provenance=provenance,
            alphas=alpha_grid_tuple,
            primary_alpha=primary_alpha,
            beta_min=beta_min,
            second_moment_threshold=second_moment_threshold,
            atol=atol,
        )
    elif "probabilities_path" in validation_cfg:
        probabilities_path = Path(validation_cfg["probabilities_path"])
        probabilities = load_probability_vector(probabilities_path)
        provenance = {
            "source": "probabilities_path",
            "probabilities_path": str(probabilities_path),
        }
        result = evaluate_anti_concentration_from_probabilities(
            probabilities,
            provenance=provenance,
            mode="exact_probabilities",
            alphas=alpha_grid_tuple,
            primary_alpha=primary_alpha,
            beta_min=beta_min,
            second_moment_threshold=second_moment_threshold,
            atol=atol,
        )
    elif "probabilities" in validation_cfg:
        result = evaluate_anti_concentration_from_probabilities(
            validation_cfg["probabilities"],
            provenance={"source": "inline_probabilities"},
            mode="exact_probabilities",
            alphas=alpha_grid_tuple,
            primary_alpha=primary_alpha,
            beta_min=beta_min,
            second_moment_threshold=second_moment_threshold,
            atol=atol,
        )
    else:
        model, provenance = _build_model_from_cfg(cfg)
        result = evaluate_anti_concentration_from_model(
            model,
            provenance=provenance,
            max_qubits=int(validation_cfg.get("max_qubits", 20)),
            alphas=alpha_grid_tuple,
            primary_alpha=primary_alpha,
            beta_min=beta_min,
            second_moment_threshold=second_moment_threshold,
            atol=atol,
        )

    artifact_paths = write_anti_concentration_artifacts(result, output_dir=output_dir, stem=stem)
    log.info(
        "Anti-concentration validation complete: mode=%s summary=%s thresholds=%s",
        result["mode"],
        artifact_paths["summary_path"],
        artifact_paths["thresholds_path"],
    )
    result["artifact_paths"] = {key: str(path) for key, path in artifact_paths.items()}
    return result
