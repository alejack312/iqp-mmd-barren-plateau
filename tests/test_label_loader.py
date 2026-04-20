from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from iqp_bp.forge.label_loader import load_labeled_rows


def _make_local_tmp_dir() -> Path:
    base_dir = Path(tempfile.gettempdir()) / "iqp_bp_tmp_forge_labels"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / str(uuid.uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _write_checkpoint(path: Path, G: np.ndarray) -> None:
    np.savez(path, G=G.astype(np.uint8), theta=np.zeros(G.shape[0], dtype=np.float64))


def test_load_labeled_rows_reads_labels_and_skips_missing_ac_field():
    tmp_path = _make_local_tmp_dir()
    try:
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_abs = checkpoints_dir / "abs_row.npz"
        checkpoint_rel = checkpoints_dir / "rel_row.npz"
        _write_checkpoint(checkpoint_abs, np.array([[1, 0], [1, 0]], dtype=np.uint8))
        _write_checkpoint(checkpoint_rel, np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))

        rows_path = tmp_path / "results.jsonl"
        records = [
            {
                "family": "product_state",
                "n": 2,
                "m": 2,
                "kernel": "gaussian",
                "init": "uniform",
                "param_idx": 0,
                "ac_passes_primary_threshold": True,
                "ac_checkpoint_path": str(checkpoint_abs),
            },
            {
                "family": "complete_graph",
                "n": 3,
                "m": 2,
                "kernel": "laplacian",
                "init": "small_angle",
                "param_idx": 1,
                "ac_passes_primary_threshold": False,
                "ac_checkpoint_path": str(Path("checkpoints") / checkpoint_rel.name),
            },
            {
                "family": "product_state",
                "n": 2,
                "m": 2,
                "kernel": "gaussian",
                "init": "uniform",
                "param_idx": 2,
                "ac_checkpoint_path": str(checkpoint_abs),
            },
        ]
        rows_path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        rows = load_labeled_rows(rows_path)

        assert len(rows) == 2
        assert rows[0].row_id == "product_state_n2_kgaussian_iuniform_pi0"
        assert rows[0].plateau_observed == "PlateauAbsent"
        assert rows[0].G.shape == (2, 2)

        assert rows[1].row_id == "complete_graph_n3_klaplacian_ismall_angle_pi1"
        assert rows[1].plateau_observed == "PlateauObserved"
        assert rows[1].G.shape == (2, 3)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
