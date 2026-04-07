from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from iqp_bp.experiments.run_validation import load_iqp_checkpoint
from iqp_bp.experiments.run_scaling import run


def _make_output_dir() -> Path:
    root = Path("tests") / "_tmp_scaling" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_run_scaling_writes_anti_concentration_fields_for_small_n():
    output_dir = _make_output_dir()
    cfg = {
        "experiment": {
            "seed": 5,
            "output_dir": str(output_dir),
        },
        "circuit": {
            "family": ["complete_graph"],
            "n_qubits": [4],
            "n_generators": "n",
            "lattice": {"dimension": 2, "range": 1},
            "erdos_renyi": {"p_edge": [2.0]},
        },
        "kernel": {
            "type": ["gaussian"],
            "bandwidth": [1.0],
        },
        "init": {
            "scheme": ["uniform"],
            "uniform": {"low": -3.14159265, "high": 3.14159265},
            "small_angle": {"std": [0.1]},
            "data_dependent": {"dataset": "product_bernoulli"},
        },
        "dataset": {
            "type": "product_bernoulli",
            "n_samples": 32,
        },
        "estimation": {
            "num_a_samples": 4,
            "num_z_samples": 16,
            "num_seeds": 1,
        },
        "anti_concentration": {
            "enabled": True,
            "max_n": 8,
            "alphas": [0.5, 1.0],
            "primary_alpha": 1.0,
            "beta_min": 0.2,
            "second_moment_threshold": 1.0,
            "export_checkpoint": True,
        },
    }

    try:
        run(cfg)
        records_path = output_dir / "results.jsonl"
        lines = records_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        record = json.loads(lines[0])
        assert record["anti_concentration_available"] is True
        assert record["anti_concentration_reason"] == "exact_small_n"
        assert record["ac_theta_seed_index"] == 0
        assert isinstance(record["ac_scaled_second_moment"], float)
        assert isinstance(record["ac_primary_beta_hat"], float)
        assert isinstance(record["ac_passes_primary_threshold"], bool)
        assert isinstance(record["ac_passes_second_moment_threshold"], bool)
        assert set(record["ac_beta_hat_by_alpha"].keys()) == {"0.5", "1.0"}
        checkpoint_path = Path(record["ac_checkpoint_path"])
        assert checkpoint_path.exists()
        model, metadata = load_iqp_checkpoint(checkpoint_path)
        assert model.n == 4
        assert metadata["family"] == "complete_graph"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_run_scaling_marks_anti_concentration_unavailable_above_cap():
    output_dir = _make_output_dir()
    cfg = {
        "experiment": {
            "seed": 9,
            "output_dir": str(output_dir),
        },
        "circuit": {
            "family": ["product_state"],
            "n_qubits": [5],
            "n_generators": "n",
            "lattice": {"dimension": 2, "range": 1},
            "erdos_renyi": {"p_edge": [2.0]},
        },
        "kernel": {
            "type": ["gaussian"],
            "bandwidth": [1.0],
        },
        "init": {
            "scheme": ["small_angle"],
            "uniform": {"low": -3.14159265, "high": 3.14159265},
            "small_angle": {"std": [0.1]},
            "data_dependent": {"dataset": "product_bernoulli"},
        },
        "dataset": {
            "type": "product_bernoulli",
            "n_samples": 32,
        },
        "estimation": {
            "num_a_samples": 4,
            "num_z_samples": 16,
            "num_seeds": 1,
        },
        "anti_concentration": {
            "enabled": True,
            "max_n": 4,
        },
    }

    try:
        run(cfg)
        records_path = output_dir / "results.jsonl"
        lines = records_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        record = json.loads(lines[0])
        assert record["anti_concentration_available"] is False
        assert record["anti_concentration_reason"] == "n_exceeds_max_n:5>4"
        assert "ac_scaled_second_moment" not in record
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
