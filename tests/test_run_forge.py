from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import pytest

from iqp_bp.experiments.run_forge import run
from iqp_bp.forge.parser import (
    detect_plateau_agreement,
    detect_status,
    extract_witness_block,
)
from iqp_bp.forge.runner import ForgeResult


def _make_local_tmp_dir() -> Path:
    base_dir = Path(tempfile.gettempdir()) / "iqp_bp_tmp_forge"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / str(uuid.uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _small_forge_cfg(tmp_path: Path) -> dict:
    return {
        "experiment": {"output_dir": str(tmp_path / "results"), "seed": 42},
        "circuit": {
            "family": ["product_state"],
            "n_qubits": [4],
            "n_generators": "n",
            "lattice": {"dimension": 2, "range": 1},
            "erdos_renyi": {"p_edge": [2.0]},
        },
        "kernel": {"type": "gaussian", "bandwidth": [1.0]},
        "init": {"scheme": "uniform", "uniform": {"low": -3.14, "high": 3.14}},
        "dataset": {"type": "product_bernoulli", "n_samples": 100},
        "estimation": {"num_a_samples": 8, "num_z_samples": 8, "num_seeds": 1},
        "forge": {
            "max_n": 12,
            "timeout_sec": 5,
            "mode": "template_search",
            "query_template": "plateau_inducing_bounded",
            "thresholds": {"max_weight": 3, "overlap_threshold": 1},
            "export_instances": True,
            "plateau_agreement": {
                "label_source": str(tmp_path / "labels.jsonl"),
                "ac_field": "ac_passes_primary_threshold",
            },
        },
    }


def _small_plateau_cfg(tmp_path: Path, label_source: Path) -> dict:
    cfg = _small_forge_cfg(tmp_path)
    cfg["forge"]["mode"] = "plateau_agreement"
    cfg["forge"]["query_template"] = "plateau_agreement"
    cfg["forge"]["thresholds"] = {"max_weight": 1, "overlap_threshold": 0}
    cfg["forge"]["plateau_agreement"]["label_source"] = str(label_source)
    return cfg


def _write_checkpoint(path: Path, G: np.ndarray) -> None:
    np.savez(path, G=G.astype(np.uint8), theta=np.zeros(G.shape[0], dtype=np.float64))


def _write_plateau_fixture(tmp_path: Path) -> Path:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "plateau_row.npz"
    _write_checkpoint(checkpoint_path, np.array([[1, 0], [1, 0]], dtype=np.uint8))

    rows_path = tmp_path / "labels.jsonl"
    records = [
        {
            "family": "product_state",
            "kernel": "gaussian",
            "init": "small_angle",
            "n": 2,
            "m": 2,
            "param_idx": 0,
            "ac_passes_primary_threshold": False,
            "ac_checkpoint_path": str(checkpoint_path),
        },
        {
            "family": "product_state",
            "kernel": "gaussian",
            "init": "small_angle",
            "n": 2,
            "m": 2,
            "param_idx": 1,
            "ac_passes_primary_threshold": True,
            "ac_checkpoint_path": str(checkpoint_path),
        },
    ]
    rows_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return rows_path


def _load_rows(results_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (results_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_detect_status_and_extract_witness_block():
    stdout = (
        "Failed test candidate. Expected unsat, got sat. "
        "Found instance #(struct:Sat witness)\n"
    )

    assert detect_status(stdout) == "sat"
    assert extract_witness_block(stdout) == "#(struct:Sat witness)"
    assert detect_status("Test passed: candidate") == "unsat"
    assert detect_status("solver output unavailable") == "unknown"
    assert detect_plateau_agreement("Test passed: candidate") == "agree"
    assert (
        detect_plateau_agreement(
            "",
            "Failed test candidate. Expected sat, got unsat.",
        )
        == "disagree"
    )
    assert detect_plateau_agreement("solver output unavailable") == "unknown"


def test_run_forge_writes_results_and_sidecars(monkeypatch):
    tmp_path = _make_local_tmp_dir()
    try:
        stdout = (
            "Failed test candidate_product_state_n4. Expected unsat, got sat. "
            "Found instance #(struct:Sat witness)\n"
        )
        monkeypatch.setattr(
            "iqp_bp.experiments.run_forge.run_racket",
            lambda *_args, **_kwargs: ForgeResult("unknown", stdout, "", 0.25, 1),
        )

        cfg = _small_forge_cfg(tmp_path)
        run(cfg)

        results_dir = tmp_path / "results"
        rows = _load_rows(results_dir)

        assert rows
        row = rows[0]
        assert row["family"] == "product_state"
        assert row["status"] == "sat"
        assert row["racket_available"] is True
        assert Path(row["instance_path"]).exists()
        assert Path(row["search_path"]).exists()
        assert Path(row["raw_stdout_path"]).exists()
        assert Path(row["witness_path"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_forge_parses_sat_from_stderr(monkeypatch):
    tmp_path = _make_local_tmp_dir()
    try:
        stdout = (
            "Forge version: 5.2\n"
            "******************** TEST FAILED *******************\n"
        )
        stderr = (
            "Failed test candidate_product_state_n4. Expected unsat, got sat. "
            "Found instance #(struct:Sat witness)\n"
        )
        monkeypatch.setattr(
            "iqp_bp.experiments.run_forge.run_racket",
            lambda *_args, **_kwargs: ForgeResult("unknown", stdout, stderr, 0.25, 1),
        )

        cfg = _small_forge_cfg(tmp_path)
        run(cfg)

        results_dir = tmp_path / "results"
        rows = _load_rows(results_dir)

        assert rows
        row = rows[0]
        assert row["status"] == "sat"
        assert Path(row["witness_path"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_forge_skips_when_racket_missing(monkeypatch):
    tmp_path = _make_local_tmp_dir()
    try:
        monkeypatch.setattr(
            "iqp_bp.experiments.run_forge.run_racket",
            lambda *_args, **_kwargs: ForgeResult(
                "skipped_no_racket",
                "",
                "racket not on PATH",
                0.0,
                None,
            ),
        )

        cfg = _small_forge_cfg(tmp_path)
        run(cfg)

        results_dir = tmp_path / "results"
        rows = _load_rows(results_dir)

        assert rows
        row = rows[0]
        assert row["status"] == "skipped_no_racket"
        assert row["racket_available"] is False
        assert row["witness_path"] is None
        assert Path(row["instance_path"]).exists()
        assert Path(row["search_path"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_forge_plateau_agreement_mode_writes_results(monkeypatch):
    tmp_path = _make_local_tmp_dir()
    try:
        labels_path = _write_plateau_fixture(tmp_path)
        outputs = iter(
            [
                ForgeResult(
                    "unknown",
                    "Test passed: row_product_state_n2_kgaussian_ismall_angle_pi0",
                    "",
                    0.2,
                    0,
                ),
                ForgeResult(
                    "unknown",
                    "",
                    "Failed test row_product_state_n2_kgaussian_ismall_angle_pi1. Expected sat, got unsat.",
                    0.3,
                    1,
                ),
            ]
        )
        monkeypatch.setattr(
            "iqp_bp.experiments.run_forge.run_racket",
            lambda *_args, **_kwargs: next(outputs),
        )

        cfg = _small_plateau_cfg(tmp_path, labels_path)
        run(cfg)

        rows = _load_rows(tmp_path / "results")
        assert len(rows) == 2
        assert rows[0]["mode"] == "plateau_agreement"
        assert rows[0]["agreement"] == "agree"
        assert rows[0]["structurally_predicted"] is True
        assert Path(rows[0]["instance_path"]).exists()
        assert Path(rows[0]["search_path"]).exists()
        assert Path(rows[0]["raw_stdout_path"]).exists()

        assert rows[1]["agreement"] == "disagree"
        assert rows[1]["structurally_predicted"] is True
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.skipif(
    shutil.which("racket") is None,
    reason="racket not on PATH — live Forge smoke requires a real Racket install",
)
def test_run_forge_live_racket_smoke():
    """End-to-end live run against real Forge.

    Regression guard for three bugs the mocked suite missed in the first pass:
    - `#lang forge` not on line 1 of the library (default-load-handler error)
    - query template using `example ... for SCOPE` (parse error, NUM-CONST-TOK)
    - `run` query opening Sterling and hanging until timeout
    """
    tmp_path = _make_local_tmp_dir()
    try:
        cfg = _small_forge_cfg(tmp_path)
        cfg["forge"]["timeout_sec"] = 60

        run(cfg)

        results_dir = tmp_path / "results"
        rows = _load_rows(results_dir)

        assert len(rows) == 1, f"expected exactly one row; got {len(rows)}"
        row = rows[0]

        assert row["racket_available"] is True
        assert row["status"] in {"sat", "unsat"}, (
            f"parser failed to classify real Forge output; got {row['status']!r}. "
            f"stdout path: {row['raw_stdout_path']}"
        )

        search_text = Path(row["search_path"]).read_text(encoding="utf-8")
        assert search_text.splitlines()[0] == "#lang forge", (
            "search file must start with #lang forge — regression of bug #1"
        )

        if row["status"] == "sat":
            assert row["witness_path"] is not None
            witness = Path(row["witness_path"]).read_text(encoding="utf-8")
            assert "#(struct:Sat" in witness, (
                f"witness did not contain Sat marker; got first 200 chars: "
                f"{witness[:200]!r}"
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_query_plateau_manifests_joint_renders_predicate_call():
    from iqp_bp.forge.query_templates import build_query

    rendered = build_query(
        "plateau_manifests_joint",
        instance_name="row_x",
        thresholds={"max_weight": 0, "overlap_threshold": 0},
        bounds={"n": 4, "m": 4},
        context={"bounds_inst": "b_x"},
    )

    assert "plateau_manifests[`Exp0]" in rendered
    assert "iff `Exp0.plateau_observed = PlateauObserved" in rendered
    assert "for b_x is sat" in rendered


def test_build_query_plateau_manifests_no_escape_calls_ablation_predicate():
    from iqp_bp.forge.query_templates import build_query

    rendered = build_query(
        "plateau_manifests_joint_no_escape",
        instance_name="row_x",
        thresholds={"max_weight": 0, "overlap_threshold": 0},
        bounds={"n": 4, "m": 4},
        context={"bounds_inst": "b_x"},
    )

    assert "plateau_manifests_no_escape[`Exp0]" in rendered
    # Must not accidentally fall back to the full predicate.
    assert "plateau_manifests[`Exp0]" not in rendered


def test_run_forge_plateau_manifests_joint_template_end_to_end(monkeypatch):
    tmp_path = _make_local_tmp_dir()
    try:
        labels_path = _write_plateau_fixture(tmp_path)
        outputs = iter(
            [
                ForgeResult(
                    "unknown",
                    "Test passed: row_product_state_n2_kgaussian_ismall_angle_pi0",
                    "",
                    0.2,
                    0,
                ),
                ForgeResult(
                    "unknown",
                    "Test passed: row_product_state_n2_kgaussian_ismall_angle_pi1",
                    "",
                    0.2,
                    0,
                ),
            ]
        )
        monkeypatch.setattr(
            "iqp_bp.experiments.run_forge.run_racket",
            lambda *_args, **_kwargs: next(outputs),
        )

        cfg = _small_plateau_cfg(tmp_path, labels_path)
        cfg["forge"]["query_template"] = "plateau_manifests_joint"
        cfg["forge"]["thresholds"] = {"max_weight": 0, "overlap_threshold": 0}
        run(cfg)

        rows = _load_rows(tmp_path / "results")
        assert len(rows) == 2
        assert all(r["query_template"] == "plateau_manifests_joint" for r in rows)
        assert all(r["agreement"] == "agree" for r in rows)

        search_text = Path(rows[0]["search_path"]).read_text(encoding="utf-8")
        # The F4 query block must invoke `plateau_manifests`, not the F3 predicate.
        query_block = search_text.rsplit("test expect", 1)[-1]
        assert "plateau_manifests[`Exp0]" in query_block
        assert "plateau_structurally_predicted" not in query_block
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.skipif(
    shutil.which("racket") is None,
    reason="racket not on PATH â€” live Forge plateau smoke requires a real Racket install",
)
def test_run_forge_live_plateau_agreement_smoke():
    tmp_path = _make_local_tmp_dir()
    try:
        labels_path = _write_plateau_fixture(tmp_path)
        cfg = _small_plateau_cfg(tmp_path, labels_path)
        cfg["forge"]["timeout_sec"] = 60

        run(cfg)

        rows = _load_rows(tmp_path / "results")
        assert len(rows) == 2
        by_id = {row["row_id"]: row for row in rows}
        assert by_id["product_state_n2_kgaussian_ismall_angle_pi0"]["agreement"] == "agree"
        assert (
            by_id["product_state_n2_kgaussian_ismall_angle_pi0"]["structurally_predicted"]
            is True
        )
        assert by_id["product_state_n2_kgaussian_ismall_angle_pi1"]["agreement"] == "disagree"
        assert (
            by_id["product_state_n2_kgaussian_ismall_angle_pi1"]["structurally_predicted"]
            is True
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
