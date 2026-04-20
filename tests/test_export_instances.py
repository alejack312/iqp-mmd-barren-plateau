from __future__ import annotations

import re
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from iqp_bp.forge.export_instances import _overlap_edges, export_to_forge


def _make_local_tmp_dir() -> Path:
    base_dir = Path(tempfile.gettempdir()) / "iqp_bp_tmp_forge_export"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / str(uuid.uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def test_emits_overlap_relation():
    tmp_path = _make_local_tmp_dir()
    G = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )

    try:
        output_path = tmp_path / "overlap.frg"
        export_to_forge(G, 4, output_path)
        content = output_path.read_text(encoding="utf-8")

        assert re.search(r"overlaps\s*=\s*`G0->`G1 \+ `G1->`G0", content)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_emits_omitted_overlaps_binding_for_non_overlapping_case():
    """Forge 5.2 rejects `overlaps = none->none` (NONE-TOK parse error); the
    correct idiom for an empty arity-2 field is to omit the binding entirely
    and let Forge default it to empty via the `overlaps_consistent` pred."""
    tmp_path = _make_local_tmp_dir()
    G = np.eye(3, dtype=np.uint8)

    try:
        output_path = tmp_path / "disjoint.frg"
        export_to_forge(G, 3, output_path)
        content = output_path.read_text(encoding="utf-8")

        assert not re.search(r"overlaps\s*=", content), (
            "overlaps binding should be omitted for non-overlapping circuits"
        )
        assert "overlaps binding omitted" in content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_emits_params_bindings():
    tmp_path = _make_local_tmp_dir()
    G = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    try:
        output_path = tmp_path / "params.frg"
        export_to_forge(G, 3, output_path)
        content = output_path.read_text(encoding="utf-8")

        assert "Params = `Params0" in content
        assert re.search(r"max_weight = `Params0 -> \d+", content)
        assert re.search(r"overlap_threshold = `Params0 -> \d+", content)
        assert re.search(r"max_qubit_degree = `Params0 -> \d+", content)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_overlap_edges_symmetric():
    G = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    expected = {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }
    assert set(_overlap_edges(G)) == expected


def test_empty_generators_omit_arity2_bindings():
    """With zero generators both contains and overlaps are empty arity-2
    relations; Forge 5.2 requires omitting their bindings. The unary
    `Generator = none` binding is still valid."""
    tmp_path = _make_local_tmp_dir()
    G = np.zeros((0, 4), dtype=np.uint8)

    try:
        output_path = tmp_path / "empty.frg"
        export_to_forge(G, 4, output_path)
        content = output_path.read_text(encoding="utf-8")

        assert "Generator = none" in content
        assert not re.search(r"contains\s*=", content)
        assert not re.search(r"overlaps\s*=", content)
        assert "contains binding omitted" in content
        assert "overlaps binding omitted" in content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
