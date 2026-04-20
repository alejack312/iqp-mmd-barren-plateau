"""Artifact helpers for marginal-order diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_marginal_summary(
    summary: dict[str, Any],
    *,
    output_dir: str | Path,
    stem: str,
) -> Path:
    """Persist one marginal summary JSON artifact and return its path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return path
