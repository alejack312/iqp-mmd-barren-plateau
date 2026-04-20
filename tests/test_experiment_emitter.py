from __future__ import annotations

from pathlib import Path
import re
import shutil
import tempfile
import uuid

import numpy as np
import pytest

from iqp_bp.forge.experiment_emitter import emit_experiment_instance
from iqp_bp.forge.label_loader import LabeledRow


def _make_local_tmp_dir() -> Path:
    base_dir = Path(tempfile.gettempdir()) / "iqp_bp_tmp_forge_emitter"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / str(uuid.uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _row(*, kernel: str = "gaussian", init: str = "uniform") -> LabeledRow:
    return LabeledRow(
        row_id=f"row_{kernel}_{init}",
        family="product_state",
        n=3,
        m=3,
        kernel=kernel,
        init=init,
        G=np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        ),
        plateau_observed="PlateauObserved",
        source_row={},
    )


def test_emit_experiment_instance_binds_loss_kernel_init_and_outcome():
    tmp_path = _make_local_tmp_dir()
    try:
        output_path = tmp_path / "instance.frg"
        emit_experiment_instance(_row(), output_path, instance_name="experiment_probe")
        content = output_path.read_text(encoding="utf-8")

        assert "inst hypergraph_3_3 {" in content
        assert "Qubit = `Q0 + `Q1 + `Q2" in content
        assert "Generator = `G0 + `G1 + `G2" in content
        assert "inst experiment_probe {" in content
        assert re.search(r"loss\s*=\s*`Exp0\s*->\s*`MMD0", content)
        assert re.search(r"kernel\s*=\s*`Exp0\s*->\s*`Gaussian0", content)
        assert re.search(r"init\s*=\s*`Exp0\s*->\s*`UniformInit0", content)
        assert re.search(
            r"plateau_observed\s*=\s*`Exp0\s*->\s*`PlateauObserved0",
            content,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.parametrize(
    ("init", "expected_atom"),
    [
        ("uniform", "UniformInit0"),
        ("small_angle", "SmallAngleInit0"),
        ("data_dependent", "DataDependentInit0"),
    ],
)
def test_emit_experiment_instance_maps_all_init_schemes(init: str, expected_atom: str):
    tmp_path = _make_local_tmp_dir()
    try:
        output_path = tmp_path / f"{init}.frg"
        emit_experiment_instance(_row(init=init), output_path)
        content = output_path.read_text(encoding="utf-8")
        assert f"`{expected_atom}" in content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.parametrize(
    ("kernel", "expected_atom"),
    [
        ("gaussian", "Gaussian0"),
        ("laplacian", "Laplacian0"),
        ("multi_scale_gaussian", "MultiScaleGaussian0"),
        ("polynomial", "Polynomial0"),
        ("linear", "Linear0"),
    ],
)
def test_emit_experiment_instance_maps_all_kernel_families(
    kernel: str,
    expected_atom: str,
):
    tmp_path = _make_local_tmp_dir()
    try:
        output_path = tmp_path / f"{kernel}.frg"
        emit_experiment_instance(_row(kernel=kernel), output_path)
        content = output_path.read_text(encoding="utf-8")
        assert f"`{expected_atom}" in content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
