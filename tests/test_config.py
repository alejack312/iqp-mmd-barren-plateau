"""Tests for config loading, validation, grid resolution, and manifest persistence."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from iqp_bp.config import (
    load_config,
    persist_experiment_manifest,
    resolve_experiment_grid,
    validate_config,
)


def _minimal_valid_cfg(**section_overrides) -> dict:
    """Return a minimal config dict that passes validate_config."""
    cfg = {
        "experiment": {"name": "test", "seed": 0, "output_dir": "results/"},
        "circuit": {
            "family": "product_state",
            "n_qubits": [4],
            "n_generators": "n",
            "erdos_renyi": {"p_edge": [2.0]},
            "lattice": {"dimension": 2, "range": 1},
        },
        "kernel": {"type": "gaussian", "bandwidth": [1.0]},
        "init": {
            "scheme": "uniform",
            "uniform": {"low": -3.14159265, "high": 3.14159265},
            "small_angle": {"std": [0.1]},
        },
        "dataset": {"type": "product_bernoulli", "n_samples": 32},
        "estimation": {"num_a_samples": 4, "num_z_samples": 16, "num_seeds": 1},
    }
    for key, val in section_overrides.items():
        cfg[key] = val
    return cfg


def _workspace_tmp_dir() -> Path:
    path = Path("tests") / "_tmp_config" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


# ---------------------------------------------------------------------------
# Valid merge and validation
# ---------------------------------------------------------------------------


def test_validate_config_accepts_minimal_valid_config():
    validate_config(_minimal_valid_cfg())  # must not raise


def test_load_config_accepts_repo_scaling_v1_sweep_axes():
    cfg = load_config("configs/experiments/scaling_v1.yaml")
    assert cfg["circuit"]["family"] == [
        "product_state",
        "lattice",
        "erdos_renyi",
        "complete_graph",
    ]
    assert cfg["kernel"]["type"] == ["gaussian"]
    assert cfg["init"]["scheme"] == ["uniform", "small_angle", "data_dependent"]


def test_load_config_merges_override_into_base():
    tmp_path = _workspace_tmp_dir()
    override = {"experiment": {"name": "my_test", "seed": 99}}
    yaml_path = tmp_path / "override.yaml"
    yaml_path.write_text(yaml.dump(override))
    cfg = load_config(yaml_path)
    assert cfg["experiment"]["name"] == "my_test"
    assert cfg["experiment"]["seed"] == 99
    # base default preserved in a sibling key
    assert cfg["kernel"]["type"] == "gaussian"


def test_load_config_deep_merges_nested_sections():
    tmp_path = _workspace_tmp_dir()
    override = {"circuit": {"family": "lattice", "n_qubits": [4, 9]}}
    yaml_path = tmp_path / "override.yaml"
    yaml_path.write_text(yaml.dump(override))
    cfg = load_config(yaml_path)
    assert cfg["circuit"]["family"] == "lattice"
    assert cfg["circuit"]["n_qubits"] == [4, 9]
    # base lattice sub-section still present after merge
    assert "lattice" in cfg["circuit"]
    assert cfg["circuit"]["lattice"]["dimension"] == 2


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_family", ["random_graph", "ising_chain", "hypergraph"])
def test_validate_config_rejects_invalid_family(bad_family):
    cfg = _minimal_valid_cfg()
    cfg["circuit"]["family"] = bad_family
    with pytest.raises(ValueError, match="circuit.family"):
        validate_config(cfg)


@pytest.mark.parametrize("bad_kernel", ["rbf", "cosine", "haversine"])
def test_validate_config_rejects_invalid_kernel_type(bad_kernel):
    cfg = _minimal_valid_cfg()
    cfg["kernel"]["type"] = bad_kernel
    with pytest.raises(ValueError, match="kernel.type"):
        validate_config(cfg)


@pytest.mark.parametrize("bad_scheme", ["random", "gaussian_noise", "zeros"])
def test_validate_config_rejects_invalid_init_scheme(bad_scheme):
    cfg = _minimal_valid_cfg()
    cfg["init"]["scheme"] = bad_scheme
    with pytest.raises(ValueError, match="init.scheme"):
        validate_config(cfg)


@pytest.mark.parametrize("bad_dataset", ["mnist", "cifar10", "uniform_bits"])
def test_validate_config_rejects_invalid_dataset_type(bad_dataset):
    cfg = _minimal_valid_cfg()
    cfg["dataset"]["type"] = bad_dataset
    with pytest.raises(ValueError, match="dataset.type"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_value_inside_family_sweep():
    cfg = _minimal_valid_cfg()
    cfg["circuit"]["family"] = ["product_state", "not_a_family"]
    with pytest.raises(ValueError, match="circuit.family"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Invalid scalar shapes (fields that must be lists)
# ---------------------------------------------------------------------------


def test_validate_config_rejects_scalar_n_qubits():
    cfg = _minimal_valid_cfg()
    cfg["circuit"]["n_qubits"] = 4
    with pytest.raises(ValueError, match="n_qubits must be a list"):
        validate_config(cfg)


def test_validate_config_rejects_scalar_bandwidth():
    cfg = _minimal_valid_cfg()
    cfg["kernel"]["bandwidth"] = 1.0
    with pytest.raises(ValueError, match="bandwidth must be a list"):
        validate_config(cfg)


def test_validate_config_rejects_scalar_p_edge():
    cfg = _minimal_valid_cfg()
    cfg["circuit"]["erdos_renyi"]["p_edge"] = 0.5
    with pytest.raises(ValueError, match="p_edge must be a list"):
        validate_config(cfg)


def test_validate_config_rejects_scalar_small_angle_std():
    cfg = _minimal_valid_cfg()
    cfg["init"]["small_angle"]["std"] = 0.1
    with pytest.raises(ValueError, match="init.small_angle.std must be a list"):
        validate_config(cfg)


def test_validate_config_rejects_multi_scale_weights_that_do_not_sum_to_one():
    cfg = _minimal_valid_cfg()
    cfg["kernel"]["type"] = "multi_scale_gaussian"
    cfg["kernel"]["multi_scale_gaussian"] = {
        "sigmas": [0.5, 1.0, 2.0],
        "weights": [0.2, 0.2, 0.2],
    }
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Persisted manifest contents
# ---------------------------------------------------------------------------


def test_persist_manifest_writes_config_json():
    tmp_path = _workspace_tmp_dir()
    cfg = _minimal_valid_cfg()
    cfg["experiment"]["output_dir"] = str(tmp_path)
    grid = resolve_experiment_grid(cfg)
    config_path, _ = persist_experiment_manifest(cfg, grid)
    assert config_path.exists()
    saved = json.loads(config_path.read_text())
    assert saved["experiment"]["name"] == "test"
    assert saved["circuit"]["family"] == "product_state"
    assert saved["kernel"]["type"] == "gaussian"


def test_persist_manifest_writes_manifest_json():
    tmp_path = _workspace_tmp_dir()
    cfg = _minimal_valid_cfg()
    cfg["experiment"]["output_dir"] = str(tmp_path)
    grid = resolve_experiment_grid(cfg)
    _, manifest_path = persist_experiment_manifest(cfg, grid)
    assert manifest_path.exists()
    saved_grid = json.loads(manifest_path.read_text())
    assert isinstance(saved_grid, list)
    assert len(saved_grid) == len(grid)
    entry = saved_grid[0]
    assert entry["family"] == "product_state"
    assert entry["n"] == 4
    assert entry["kernel"] == "gaussian"
    assert "dataset_type" in entry
    assert "init_scheme" in entry


def test_persist_manifest_grid_covers_cartesian_axes():
    tmp_path = _workspace_tmp_dir()
    # Use resolve_experiment_grid directly — bypass validate_config so that
    # multi-value family lists are allowed without schema friction.
    cfg = _minimal_valid_cfg()
    cfg["experiment"]["output_dir"] = str(tmp_path)
    cfg["circuit"]["family"] = ["product_state", "complete_graph"]
    cfg["circuit"]["n_qubits"] = [4, 9]
    cfg["kernel"]["bandwidth"] = [0.5, 1.0]
    grid = resolve_experiment_grid(cfg)
    _, manifest_path = persist_experiment_manifest(cfg, grid)
    # 2 families × 2 n × 2 bandwidths × 1 kernel × 1 init = 8 settings
    assert len(grid) == 8
    saved_grid = json.loads(manifest_path.read_text())
    assert len(saved_grid) == 8
    assert {s["family"] for s in saved_grid} == {"product_state", "complete_graph"}
    assert {s["n"] for s in saved_grid} == {4, 9}
    assert {s["bandwidth"] for s in saved_grid} == {0.5, 1.0}
