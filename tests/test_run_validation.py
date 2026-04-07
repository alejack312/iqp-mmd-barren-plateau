"""Tests for the deterministic anti-concentration validation runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from iqp_bp.experiments.run_validation import (
    evaluate_anti_concentration_from_model,
    run,
    samples_to_probability_vector,
)
from iqp_bp.iqp.model import IQPModel


def _workspace_tmp_dir() -> Path:
    """Create a writable scratch directory inside the repo workspace."""
    path = Path("tests") / "_tmp_validation" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_samples_to_probability_vector_matches_empirical_histogram():
    """Bitstring samples should map to the expected empirical probability vector."""
    samples = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.uint8,
    )

    probabilities = samples_to_probability_vector(samples)

    expected = np.array([0.5, 0.25, 0.0, 0.25], dtype=np.float64)
    assert np.allclose(probabilities, expected)


def test_evaluate_anti_concentration_from_model_attaches_mode_and_provenance():
    """Model-based validation should report the exact-IQP mode and preserve provenance."""
    model = IQPModel(G=np.eye(3, dtype=np.uint8), theta=np.zeros(3))

    result = evaluate_anti_concentration_from_model(
        model,
        provenance={"family": "product_state", "seed": 7},
        max_qubits=4,
    )

    assert result["mode"] == "exact_iqp_model"
    assert result["provenance"]["family"] == "product_state"
    assert result["provenance"]["seed"] == 7
    assert result["passes_primary_threshold"] is False


def test_run_validation_from_checkpoint_writes_summary_and_thresholds():
    """Checkpoint-driven validation should serialize JSON and CSV artifacts."""
    tmp_path = _workspace_tmp_dir()
    checkpoint_path = tmp_path / "toy_checkpoint.npz"
    np.savez(
        checkpoint_path,
        G=np.eye(2, dtype=np.uint8),
        theta=np.zeros(2, dtype=np.float64),
        family=np.array("product_state"),
        seed=np.array(123),
    )

    output_dir = tmp_path / "results"
    cfg = {
        "experiment": {
            "output_dir": str(output_dir),
            "seed": 123,
        },
        "validation": {
            "checkpoint_path": str(checkpoint_path),
            "output_stem": "checkpoint_validation",
        },
    }

    result = run(cfg)

    assert result["mode"] == "exact_iqp_model"
    assert result["provenance"]["source"] == "checkpoint"
    summary_path = output_dir / "checkpoint_validation.json"
    thresholds_path = output_dir / "checkpoint_validation_thresholds.csv"
    assert summary_path.exists()
    assert thresholds_path.exists()

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["mode"] == "exact_iqp_model"
    assert summary["provenance"]["checkpoint_path"] == str(checkpoint_path)

    with open(thresholds_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 3
    assert {"alpha", "threshold_probability", "beta_hat", "passes_beta_threshold"} == set(rows[0].keys())


def test_run_validation_from_samples_path_uses_empirical_histogram_mode():
    """Sample-driven validation should label the result as an empirical histogram."""
    tmp_path = _workspace_tmp_dir()
    samples_path = tmp_path / "samples.csv"
    np.savetxt(samples_path, np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.uint8), fmt="%d", delimiter=",")

    output_dir = tmp_path / "results"
    cfg = {
        "experiment": {
            "output_dir": str(output_dir),
            "seed": 321,
        },
        "validation": {
            "samples_path": str(samples_path),
            "output_stem": "sample_validation",
        },
    }

    result = run(cfg)

    assert result["mode"] == "empirical_histogram"
    assert result["provenance"]["source"] == "samples_path"
    summary_path = output_dir / "sample_validation.json"
    assert summary_path.exists()
