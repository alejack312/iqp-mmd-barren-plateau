"""Config loading, validation, grid resolution, and manifest persistence."""

from __future__ import annotations

import json
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_CONFIG_PATH = _REPO_ROOT / "configs" / "base.yaml"
_SCHEMA_PATH = _REPO_ROOT / "configs" / "schema.yaml"

_REQUIRED_TOP_LEVEL_SECTIONS = (
    "experiment",
    "circuit",
    "kernel",
    "init",
    "dataset",
    "estimation",
)

_REQUIRED_FIELDS = (
    ("experiment", "name"),
    ("circuit", "family"),
    ("circuit", "n_qubits"),
    ("kernel", "type"),
    ("init", "scheme"),
    ("dataset", "type"),
)

_VALID_FAMILIES = {
    "product_state",
    "lattice",
    "erdos_renyi",
    "complete_graph",
    "bounded_degree",
    "dense",
    "community",
    "symmetric",
}
_VALID_KERNELS = {
    "gaussian",
    "laplacian",
    "multi_scale_gaussian",
    "polynomial",
    "linear",
}
_VALID_INIT_SCHEMES = {"uniform", "small_angle", "data_dependent"}
_VALID_DATASETS = {"product_bernoulli", "ising", "binary_mixture"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING"}
_VALID_QISKIT_BACKENDS = {"statevector", "aer_simulator"}
_VALID_QISKIT_NOISE_MODELS = {
    "depolarizing",
    "readout",
    "combined",
    "amplitude_damping",
    "phase_damping",
    "thermal_relaxation",
    "backend_preset",
}
_VALID_ISING_TOPOLOGIES = {"grid_2d", "erdos_renyi"}
_VALID_SYMMETRIC_PARITY = {"even", "odd"}
_VALID_OUTPUT_FORMATS = {"jsonl", "parquet"}
_VALID_FIGURE_FORMATS = {"pdf", "png", "svg"}
_VALID_TRAINING_OPTIMIZERS = {"sgd", "adam"}
_VALID_TRAINING_LOSS_MODES = {"auto", "sampled", "exact_small_n"}
_VALID_TRAINING_DISTRIBUTION_MODES = {"auto", "exact", "sample"}

_SWEEPABLE_ENUM_PATHS = {
    ("circuit", "family"): _VALID_FAMILIES,
    ("kernel", "type"): _VALID_KERNELS,
    ("init", "scheme"): _VALID_INIT_SCHEMES,
}

_SCALAR_ENUM_PATHS = {
    ("dataset", "type"): _VALID_DATASETS,
    ("experiment", "log_level"): _VALID_LOG_LEVELS,
    ("dataset", "ising", "topology"): _VALID_ISING_TOPOLOGIES,
    ("circuit", "symmetric", "parity"): _VALID_SYMMETRIC_PARITY,
    ("output", "format"): _VALID_OUTPUT_FORMATS,
    ("output", "figures", "format"): _VALID_FIGURE_FORMATS,
    ("qiskit", "noise", "model"): _VALID_QISKIT_NOISE_MODELS,
    ("training", "optimizer"): _VALID_TRAINING_OPTIMIZERS,
    ("training", "loss_mode"): _VALID_TRAINING_LOSS_MODES,
    ("training", "diagnostics", "distribution_mode"): _VALID_TRAINING_DISTRIBUTION_MODES,
}

_OPTIONAL_NULL_LIST_PATHS = {
    ("kernel", "multi_scale_gaussian", "weights"),
}


def load_config(
    path: str | Path,
    base_path: str | Path = _BASE_CONFIG_PATH,
) -> dict[str, Any]:
    """Load an experiment YAML config and deep-merge it over base defaults."""
    base = _load_yaml(base_path)
    override = _load_yaml(path)
    merged = _deep_merge(base, override)
    validate_config(merged)
    return merged


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    """Return the repo's YAML schema as a plain nested dict."""
    return _load_yaml(_SCHEMA_PATH)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` with override precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Validate the merged config against the repo schema and runner contract."""
    schema = _load_schema()
    _require_mapping("config", config)

    for section in _REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing required top-level section: '{section}'")
        _require_mapping(section, config[section])

    for path in _REQUIRED_FIELDS:
        if _get_path(config, path) is None:
            raise ValueError(f"{'.'.join(path)} is required")

    _validate_against_schema(config, schema)

    for path, allowed in _SWEEPABLE_ENUM_PATHS.items():
        _validate_enum(config, path, allowed, allow_lists=True)

    for path, allowed in _SCALAR_ENUM_PATHS.items():
        if _get_path(config, path) is not None:
            _validate_enum(config, path, allowed, allow_lists=False)

    backend = _get_path(config, ("qiskit", "backend"))
    if backend is not None:
        _validate_qiskit_backend(backend)

    weights = _get_path(config, ("kernel", "multi_scale_gaussian", "weights"))
    if weights is not None:
        total = float(sum(weights))
        if not abs(total - 1.0) <= 1e-9:
            raise ValueError(
                "kernel.multi_scale_gaussian.weights must sum to 1.0; "
                f"got {total:.12g}"
            )

    _validate_dataset_config(config)


def _require_positive_float(path: str, value: Any) -> None:
    """Raise if *value* is present and not strictly positive."""
    if value is None:
        return
    if not _is_float_like(value) or float(value) <= 0:
        raise ValueError(f"{path} must be a positive number, got {value!r}")


def _validate_dataset_config(config: dict[str, Any]) -> None:
    """Validate numeric bounds for dataset sub-configuration keys."""
    dataset = config.get("dataset", {})

    n_samples = dataset.get("n_samples")
    if n_samples is not None and int(n_samples) < 1:
        raise ValueError(f"dataset.n_samples must be >= 1, got {n_samples}")

    ising = dataset.get("ising") or {}
    if ising:
        _require_positive_float("dataset.ising.beta", ising.get("beta"))
        _require_positive_float("dataset.ising.coupling_std", ising.get("coupling_std"))

    bm = dataset.get("binary_mixture") or {}
    if bm:
        n_modes = bm.get("n_modes")
        if n_modes is not None and int(n_modes) < 1:
            raise ValueError(f"dataset.binary_mixture.n_modes must be >= 1, got {n_modes}")
        _require_positive_float("dataset.binary_mixture.noise", bm.get("noise"))


def _validate_against_schema(
    config: dict[str, Any],
    schema: dict[str, Any],
    path: tuple[str, ...] = (),
) -> None:
    """Validate present config keys against the schema's declared shapes."""
    for key, value in config.items():
        schema_value = schema.get(key)
        current_path = path + (key,)
        if schema_value is None:
            continue
        if isinstance(schema_value, dict):
            _require_mapping(".".join(current_path), value)
            _validate_against_schema(value, schema_value, current_path)
            continue
        _validate_schema_scalar(current_path, value, str(schema_value))


def _validate_schema_scalar(path: tuple[str, ...], value: Any, schema_type: str) -> None:
    dotted = ".".join(path)

    if schema_type == "str":
        if path in _SWEEPABLE_ENUM_PATHS:
            _ensure_str_or_str_list(dotted, value)
            return
        if path == ("circuit", "n_generators"):
            if isinstance(value, int) and not isinstance(value, bool):
                return
            if isinstance(value, str):
                return
            raise ValueError(f"{dotted} must be a string formula or integer literal")
        if not isinstance(value, str):
            raise ValueError(f"{dotted} must be a string")
        return

    if schema_type == "int":
        if not _is_int_like(value):
            raise ValueError(f"{dotted} must be an integer")
        return

    if schema_type == "float":
        if not _is_float_like(value):
            raise ValueError(f"{dotted} must be a float")
        return

    if schema_type == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{dotted} must be a boolean")
        return

    if schema_type.startswith("list["):
        if path in _OPTIONAL_NULL_LIST_PATHS and value is None:
            return
        if not isinstance(value, list):
            raise ValueError(f"{dotted} must be a list")
        item_type = schema_type[5:-1]
        for index, item in enumerate(value):
            _validate_list_item(path, index, item, item_type)
        return


def _validate_list_item(
    path: tuple[str, ...],
    index: int,
    item: Any,
    item_type: str,
) -> None:
    dotted = ".".join(path)
    item_label = f"{dotted}[{index}]"
    if item_type == "int":
        if not _is_int_like(item):
            raise ValueError(f"{item_label} must be an integer")
        return
    if item_type == "float":
        if not _is_float_like(item):
            raise ValueError(f"{item_label} must be a float")
        return
    if item_type == "str":
        if not isinstance(item, str):
            raise ValueError(f"{item_label} must be a string")
        return
    if item_type == "bool":
        if not isinstance(item, bool):
            raise ValueError(f"{item_label} must be a boolean")


def _validate_enum(
    config: dict[str, Any],
    path: tuple[str, ...],
    allowed: set[str],
    *,
    allow_lists: bool,
) -> None:
    value = _get_path(config, path)
    if value is None:
        return
    dotted = ".".join(path)

    if allow_lists and isinstance(value, list):
        if not value:
            raise ValueError(f"{dotted} must contain at least one value")
        bad = [entry for entry in value if entry not in allowed]
        if bad:
            raise ValueError(f"Unsupported {dotted}: {bad!r}")
        return

    if value not in allowed:
        raise ValueError(f"Unsupported {dotted}: {value!r}")


def _validate_qiskit_backend(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError("qiskit.backend must be a string")
    if value in _VALID_QISKIT_BACKENDS:
        return
    if value.startswith("ibm_"):
        return
    raise ValueError(
        "Unsupported qiskit.backend: "
        f"{value!r}. Use 'statevector', 'aer_simulator', or an 'ibm_*' device name."
    )


def _ensure_str_or_str_list(label: str, value: Any) -> None:
    if isinstance(value, str):
        return
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return
    raise ValueError(f"{label} must be a string or a non-empty list of strings")


def _get_path(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _require_mapping(label: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping/object")


def _is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float_like(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def resolve_experiment_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize the resolved scalar scaling grid from a merged config."""
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
            _bandwidth_values_for_kernel(kernel, cfg.get("kernel", {})),
            _erdos_renyi_values_for_family(family, cfg.get("circuit", {})),
            _small_angle_values_for_init(init_scheme, cfg.get("init", {})),
        ):
            setting = {
                "family": str(family),
                "kernel": str(kernel),
                "init_scheme": str(init_scheme),
                "n": int(n),
                "bandwidth": bandwidth,
                "er_p_edge": er_p_edge,
                "small_angle_std": small_angle_std,
                "dataset_type": str(cfg.get("dataset", {}).get("type", "product_bernoulli")),
            }
            setting.update(_resolved_kernel_metadata(str(kernel), cfg.get("kernel", {})))
            settings.append(setting)
    return settings


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _bandwidth_values_for_kernel(
    kernel: str,
    kernel_cfg: dict[str, Any],
) -> list[float | None]:
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
        return [
            float(value)
            for value in _as_list(init_cfg.get("small_angle", {}).get("std", [0.1]))
        ]
    return [None]


def _resolved_kernel_metadata(
    kernel: str,
    kernel_cfg: dict[str, Any],
) -> dict[str, Any]:
    if kernel == "multi_scale_gaussian":
        msg = kernel_cfg.get("multi_scale_gaussian", {})
        return {
            "multi_scale_sigmas": [float(value) for value in msg.get("sigmas", [])],
            "multi_scale_weights": (
                None
                if msg.get("weights") is None
                else [float(value) for value in msg.get("weights", [])]
            ),
        }
    if kernel == "polynomial":
        poly = kernel_cfg.get("polynomial", {})
        return {
            "polynomial_degree": int(poly.get("degree", 2)),
            "polynomial_constant": float(poly.get("constant", 1.0)),
        }
    return {}


def persist_experiment_manifest(
    config: dict[str, Any],
    grid: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write the merged config and resolved grid manifest to the output dir."""
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.json"

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(grid, handle, indent=2)

    return config_path, manifest_path
