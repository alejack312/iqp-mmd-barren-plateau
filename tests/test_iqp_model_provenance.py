"""Tests for IQPModel provenance metadata and checkpoint round-trips."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from iqp_bp.experiments.run_validation import load_iqp_checkpoint, save_iqp_checkpoint
from iqp_bp.iqp.model import IQPModel


def _tmp_dir() -> Path:
    root = Path("tests") / "_tmp_provenance" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# from_family provenance population
# ---------------------------------------------------------------------------


def test_from_family_provenance_contains_required_fields():
    model = IQPModel.from_family("complete_graph", n=4)
    prov = model.provenance
    assert prov["family"] == "complete_graph"
    assert prov["n"] == 4
    assert "m_requested" in prov
    assert "m_generated" in prov
    assert "family_kwargs" in prov


def test_from_family_provenance_m_requested_defaults_to_n():
    model = IQPModel.from_family("product_state", n=5)
    prov = model.provenance
    assert prov["m_requested"] == 5
    assert prov["m_generated"] == model.m


def test_from_family_provenance_m_requested_explicit():
    model = IQPModel.from_family("complete_graph", n=4, m=6)
    prov = model.provenance
    assert prov["m_requested"] == 6
    assert prov["n"] == 4
    assert prov["m_generated"] == model.m


def test_from_family_provenance_rng_seed_stored():
    model = IQPModel.from_family("complete_graph", n=4, rng_seed=42)
    assert model.provenance["rng_seed"] == 42


def test_from_family_provenance_rng_seed_absent_when_not_provided():
    model = IQPModel.from_family("product_state", n=3)
    assert "rng_seed" not in model.provenance


def test_from_family_provenance_family_kwargs_stored():
    model = IQPModel.from_family(
        "erdos_renyi", n=5, rng=np.random.default_rng(1), p_edge=0.3
    )
    assert model.provenance["family_kwargs"]["p_edge"] == 0.3


def test_from_family_provenance_is_json_serializable():
    model = IQPModel.from_family("complete_graph", n=4, rng_seed=7)
    serialized = json.dumps(model.provenance)
    recovered = json.loads(serialized)
    assert recovered["family"] == "complete_graph"
    assert recovered["rng_seed"] == 7


def test_from_family_provenance_values_are_python_native_types():
    model = IQPModel.from_family("complete_graph", n=4)
    prov = model.provenance
    assert isinstance(prov["n"], int)
    assert isinstance(prov["m_requested"], int)
    assert isinstance(prov["m_generated"], int)
    assert isinstance(prov["family"], str)


def test_init_with_no_provenance_yields_empty_dict():
    G = np.eye(3, dtype=np.uint8)
    model = IQPModel(G=G)
    assert model.provenance == {}


def test_init_with_explicit_provenance():
    G = np.eye(3, dtype=np.uint8)
    prov = {"family": "manual", "n": 3}
    model = IQPModel(G=G, provenance=prov)
    assert model.provenance["family"] == "manual"


# ---------------------------------------------------------------------------
# Checkpoint round-trips
# ---------------------------------------------------------------------------


def test_save_and_load_checkpoint_preserves_provenance():
    tmp = _tmp_dir()
    try:
        model = IQPModel.from_family("complete_graph", n=4, rng_seed=99)
        model.theta = np.ones(model.m) * 0.5
        checkpoint_path = save_iqp_checkpoint(model, tmp / "model.npz")

        loaded_model, metadata = load_iqp_checkpoint(checkpoint_path)
        assert loaded_model.provenance["family"] == "complete_graph"
        assert loaded_model.provenance["n"] == 4
        assert loaded_model.provenance["rng_seed"] == 99
        assert loaded_model.provenance["m_generated"] == model.m
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_and_load_checkpoint_provenance_json_in_metadata():
    """provenance_json key should appear in raw metadata dict after load."""
    tmp = _tmp_dir()
    try:
        model = IQPModel.from_family("product_state", n=3)
        save_iqp_checkpoint(model, tmp / "m.npz")
        _, metadata = load_iqp_checkpoint(tmp / "m.npz")
        assert "provenance_json" in metadata
        parsed = json.loads(metadata["provenance_json"])
        assert parsed["family"] == "product_state"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_without_provenance_json_loads_clean():
    """Checkpoints written before provenance support should load without error."""
    tmp = _tmp_dir()
    try:
        checkpoint_path = tmp / "legacy.npz"
        np.savez(
            checkpoint_path,
            G=np.eye(2, dtype=np.uint8),
            theta=np.zeros(2, dtype=np.float64),
            family=np.asarray("product_state"),
        )
        model, metadata = load_iqp_checkpoint(checkpoint_path)
        # No provenance_json — provenance should be empty, not raise.
        assert model.provenance == {}
        assert metadata["family"] == "product_state"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_explicit_metadata_and_provenance_coexist():
    """Explicit metadata fields and provenance_json are both written and readable."""
    tmp = _tmp_dir()
    try:
        model = IQPModel.from_family("complete_graph", n=4, rng_seed=3)
        save_iqp_checkpoint(
            model,
            tmp / "full.npz",
            metadata={"theta_seed_index": 0, "source": "test"},
        )
        loaded_model, metadata = load_iqp_checkpoint(tmp / "full.npz")
        assert loaded_model.provenance["family"] == "complete_graph"
        assert metadata["theta_seed_index"] == 0
        assert metadata["source"] == "test"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_round_trip_preserves_family_kwargs():
    tmp = _tmp_dir()
    try:
        model = IQPModel.from_family(
            "erdos_renyi", n=5, rng=np.random.default_rng(0), p_edge=0.4
        )
        save_iqp_checkpoint(model, tmp / "er.npz")
        loaded_model, _ = load_iqp_checkpoint(tmp / "er.npz")
        assert loaded_model.provenance["family_kwargs"]["p_edge"] == 0.4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
