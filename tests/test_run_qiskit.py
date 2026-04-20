from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import uuid

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from iqp_bp.experiments.run_qiskit import run


def _small_qiskit_cfg(tmp_path: Path) -> dict:
    return {
        "experiment": {"output_dir": str(tmp_path / "results"), "seed": 42},
        "circuit": {"family": ["product_state"], "n_qubits": [4], "n_generators": "n"},
        "qiskit": {
            "n_shots": [100],
            "max_n": 20,
            "transpile": {"optimization_level": 1, "save_qasm": True},
            "noise": {"enabled": True, "model": "combined", "error_rate": [0.0]},
        },
        "kernel": {"type": "gaussian", "bandwidth": [1.0]},
        "init": {"scheme": "uniform", "uniform": {"low": -3.14, "high": 3.14}},
        "dataset": {"type": "product_bernoulli", "n_samples": 100},
        "estimation": {"num_a_samples": 8, "num_z_samples": 8, "num_seeds": 1},
    }


def _make_local_tmp_dir() -> Path:
    base_dir = Path(tempfile.gettempdir()) / "iqp_bp_tmp_qiskit"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / str(uuid.uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _run_and_load(tmp_path: Path) -> tuple[Path, list[dict]]:
    cfg = _small_qiskit_cfg(tmp_path)
    run(cfg)

    results_dir = tmp_path / "results"
    rows = [
        json.loads(line)
        for line in (results_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return results_dir, rows


def test_run_qiskit_writes_summary_and_sidecars():
    tmp_path = _make_local_tmp_dir()
    try:
        results_dir, rows = _run_and_load(tmp_path)

        assert (results_dir / "results.jsonl").exists()
        assert (results_dir / "qasm" / "product_state_n4.qasm").exists()
        assert rows
        assert list((results_dir / "raw").glob("*_raw.json"))
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_summary_rows_contain_required_fields():
    tmp_path = _make_local_tmp_dir()
    try:
        _, rows = _run_and_load(tmp_path)

        assert rows
        for row in rows:
            for key in [
                "family",
                "n",
                "n_shots",
                "error_rate",
                "setting_id",
                "abs_err_sv",
                "abs_err_shots",
                "raw_path",
                "mmd2_sv",
                "mmd2_shots",
                "mmd2_noisy",
            ]:
                assert key in row
            assert Path(row["raw_path"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_raw_artifact_has_required_keys():
    tmp_path = _make_local_tmp_dir()
    try:
        _, rows = _run_and_load(tmp_path)

        raw = json.loads(Path(rows[0]["raw_path"]).read_text(encoding="utf-8"))
        for key in [
            "observables",
            "exp_classical",
            "exp_sv",
            "exp_shots",
            "exp_noisy",
            "abs_errors",
            "dataset_metadata",
            "mmd2",
            "shot_counts",
        ]:
            assert key in raw
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
