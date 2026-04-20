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
        assert record["bandwidth"] == 1.0
        assert record["dataset_type"] == "product_bernoulli"
        assert Path(record["ac_summary_path"]).exists()
        assert Path(record["ac_thresholds_path"]).exists()
        assert Path(record["ac_threshold_plot_path"]).exists()
        assert Path(record["ac_diagnostics_plot_path"]).exists()
        checkpoint_path = Path(record["ac_checkpoint_path"])
        assert checkpoint_path.exists()
        model, metadata = load_iqp_checkpoint(checkpoint_path)
        assert model.n == 4
        # Provenance is stored as provenance_json and parsed onto model.provenance.
        assert model.provenance["family"] == "complete_graph"
        assert model.provenance["n"] == 4
        assert "m_generated" in model.provenance
        assert model.provenance["m_generated"] == model.m
        assert "m_requested" in model.provenance
        assert "rng_seed" in model.provenance
        # Experiment-level metadata fields (not part of circuit provenance)
        assert metadata["source"] == "run_scaling"
        assert metadata["theta_seed_index"] == 0
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_run_scaling_writes_resolved_grid_manifest_before_records():
    """config.json and manifest.json must be written before any JSONL records.

    persist_experiment_manifest is called before the settings loop, so by the
    time the first record is flushed both files are guaranteed to exist.  The
    test verifies existence, well-formedness, and that every record carries a
    manifest_path pointer back to the same manifest file.
    """
    output_dir = _make_output_dir()
    cfg = {
        "experiment": {
            "seed": 3,
            "output_dir": str(output_dir),
        },
        "circuit": {
            "family": ["product_state"],
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
            "enabled": False,
        },
    }

    try:
        run(cfg)

        config_json = output_dir / "config.json"
        manifest_json = output_dir / "manifest.json"
        assert config_json.exists(), "config.json not written"
        assert manifest_json.exists(), "manifest.json not written"

        # Config round-trips the experiment section
        saved_cfg = json.loads(config_json.read_text(encoding="utf-8"))
        assert saved_cfg["experiment"]["seed"] == 3
        assert saved_cfg["circuit"]["family"] == ["product_state"]

        # Manifest is a non-empty list of resolved scalar settings
        saved_grid = json.loads(manifest_json.read_text(encoding="utf-8"))
        assert isinstance(saved_grid, list)
        assert len(saved_grid) >= 1
        entry = saved_grid[0]
        assert entry["family"] == "product_state"
        assert entry["n"] == 4
        assert entry["kernel"] == "gaussian"
        assert "dataset_type" in entry
        assert "init_scheme" in entry

        # Every JSONL record carries manifest_path pointing at the manifest file
        lines = (output_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert lines
        for line in lines:
            record = json.loads(line)
            assert "manifest_path" in record
            assert Path(record["manifest_path"]).exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_run_scaling_rerun_reproducibility():
    """Two independent runs with the same config+seed produce identical gradient records.

    Verifies that the named-stream seeding policy makes results stable: numeric
    fields (mean, var, std, median, n_seeds) and structural fields must match
    exactly.  Path fields (manifest_path, ac_*_path) are excluded because the
    output directories differ between runs.
    """
    base_cfg = {
        "experiment": {
            "seed": 77,
            "output_dir": None,  # filled per run below
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
            "num_a_samples": 8,
            "num_z_samples": 32,
            "num_seeds": 3,
        },
        "anti_concentration": {
            "enabled": False,
        },
    }

    out1 = _make_output_dir()
    out2 = _make_output_dir()

    import copy
    cfg1 = copy.deepcopy(base_cfg)
    cfg2 = copy.deepcopy(base_cfg)
    cfg1["experiment"]["output_dir"] = str(out1)
    cfg2["experiment"]["output_dir"] = str(out2)

    _NUMERIC_KEYS = ("mean", "var", "std", "median", "n_seeds")
    _STRUCTURAL_KEYS = ("family", "kernel", "init", "n", "dataset_type", "bandwidth", "param_idx", "m")

    try:
        run(cfg1)
        run(cfg2)

        lines1 = (out1 / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        lines2 = (out2 / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()

        assert len(lines1) == len(lines2), (
            f"Run 1 produced {len(lines1)} records, run 2 produced {len(lines2)}"
        )
        assert len(lines1) > 0

        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            r1, r2 = json.loads(l1), json.loads(l2)
            for key in _STRUCTURAL_KEYS:
                if key in r1:
                    assert r1[key] == r2[key], (
                        f"Record {i}: structural field {key!r} differs: {r1[key]!r} vs {r2[key]!r}"
                    )
            for key in _NUMERIC_KEYS:
                assert r1[key] == r2[key], (
                    f"Record {i}: numeric field {key!r} differs: {r1[key]} vs {r2[key]}"
                )
    finally:
        shutil.rmtree(out1, ignore_errors=True)
        shutil.rmtree(out2, ignore_errors=True)


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


def test_run_scaling_dataset_ising_records_dataset_metadata():
    output_dir = _make_output_dir()
    cfg = {
        "experiment": {
            "seed": 11,
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
            "type": "ising",
            "n_samples": 32,
            "ising": {
                "topology": "grid_2d",
                "beta": 1.0,
                "burn_in_sweeps": 10,
                "thinning": 1,
                "num_chains": 1,
            },
        },
        "estimation": {
            "num_a_samples": 4,
            "num_z_samples": 16,
            "num_seeds": 1,
        },
        "anti_concentration": {
            "enabled": False,
        },
    }

    try:
        run(cfg)
        records_path = output_dir / "results.jsonl"
        lines = records_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        record = json.loads(lines[0])
        dm = record["dataset_metadata"]
        assert dm["type"] == "ising"
        assert dm["topology"] == "grid_2d"
        assert dm["grid_side"] == 2
        assert "beta" in dm
        assert "seed" in dm
        assert "num_edges" in dm
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_run_scaling_dataset_binary_mixture_records_dataset_metadata():
    output_dir = _make_output_dir()
    cfg = {
        "experiment": {
            "seed": 13,
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
            "type": "binary_mixture",
            "n_samples": 32,
            "binary_mixture": {
                "n_modes": 3,
                "noise": 0.2,
            },
        },
        "estimation": {
            "num_a_samples": 4,
            "num_z_samples": 16,
            "num_seeds": 1,
        },
        "anti_concentration": {
            "enabled": False,
        },
    }

    try:
        run(cfg)
        records_path = output_dir / "results.jsonl"
        lines = records_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        record = json.loads(lines[0])
        dm = record["dataset_metadata"]
        assert dm["type"] == "binary_mixture"
        assert dm["n_modes"] == 3
        assert dm["latent_center_generation_policy"] == "standard_normal"
        assert "seed" in dm
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
