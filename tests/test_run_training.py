from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from iqp_bp.config import load_config
from iqp_bp.experiments.run_training import run


def _workspace_tmp_dir() -> Path:
    path = Path("tests") / "_tmp_run_training" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_run_training_smoke_emits_ac_and_marginal_fields():
    tmp_path = _workspace_tmp_dir()
    cfg = load_config("configs/experiments/training_smoke.yaml")
    cfg["experiment"]["output_dir"] = str(tmp_path)

    summaries = run(cfg)

    assert len(summaries) == 1
    trajectory_path = Path(summaries[0]["trajectory_path"])
    assert trajectory_path.exists()
    with open(trajectory_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    assert rows
    assert all("ac_scaled_second_moment" in row for row in rows)
    assert all("marginal_orders" in row for row in rows)
    assert all("marginal_summary_path" in row for row in rows)
