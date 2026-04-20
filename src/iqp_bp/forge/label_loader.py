"""Load plateau-observation labels for Forge experiment instances.

This module is the bridge from "Python run_scaling results" to "Forge-ready
labeled rows". The scaling runner writes one JSONL row per (setting,
param_idx) coordinate with fields like ``family``, ``n``, ``m``, ``kernel``,
``init``, and ``ac_passes_primary_threshold`` (the anti-concentration
check). It also persists the exact hypergraph matrix used for each setting
as an ``.npz`` checkpoint; the path is recorded in ``ac_checkpoint_path``.

For the Forge plateau-agreement mode we need:
    1. The plateau label (observed or absent), derived from the AC pass/fail
       boolean. The semantics are inverted on purpose: AC failing means the
       distribution is concentrated — i.e. a barren-plateau signature — so
       we tag that row as ``"PlateauObserved"``. AC passing means the
       distribution is anti-concentrated, tagging ``"PlateauAbsent"``.
    2. The hypergraph matrix G so we can regenerate the Forge inst block
       that represents the same circuit structure.

``load_labeled_rows`` skips any JSONL row that is missing either of those
ingredients rather than guessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Upstream checkpoint loader — round-trips to the numpy .npz artifacts the
# scaling runner wrote, returning a fully-populated IQPModel including the
# hypergraph generator matrix G.
from iqp_bp.experiments.run_validation import load_iqp_checkpoint


log = logging.getLogger(__name__)

# The exact string literals used in the Forge Outcome sig. Must match the
# sub-sig names in hypergraph.frg ("PlateauObserved", "PlateauAbsent").
PlateauLabel = Literal["PlateauObserved", "PlateauAbsent"]


@dataclass(frozen=True)
class LabeledRow:
    """Observed experiment row plus the checkpoint-backed hypergraph."""

    # Stable human-readable handle; derives file names for per-row Forge files.
    row_id: str
    # Circuit family (``product_state``, ``complete_graph``, etc.).
    family: str
    # Number of qubits (matches #Qubit in the Forge query scope).
    n: int
    # Number of generators (matches #Generator in the Forge query scope).
    m: int
    # Kernel choice, passed through to the Forge Experiment inst's `kernel` field.
    kernel: str
    # Init scheme, passed through to the Forge Experiment inst's `init` field.
    init: str
    # Generator matrix (shape (m, n), uint8). Serialized into the hypergraph inst.
    G: np.ndarray
    # The ground-truth plateau label, derived from AC pass/fail.
    plateau_observed: PlateauLabel
    # Full original JSONL record — kept for downstream provenance / debugging.
    source_row: dict[str, Any]


def _resolve_checkpoint_path(results_jsonl_path: Path, raw_path: str) -> Path:
    """Resolve a (possibly relative) checkpoint path written by another host.

    Scaling results may be produced on one machine and consumed on another
    with different working directories. We try three layers of fallback
    before giving up: absolute path, as-written (cwd-relative), and relative
    to the results JSONL itself.
    """
    checkpoint_path = Path(raw_path)
    # Case 1: absolute path — use as-is.
    if checkpoint_path.is_absolute():
        return checkpoint_path
    # Case 2: relative path that already resolves against the cwd.
    if checkpoint_path.exists():
        return checkpoint_path
    # Case 3: relative path that resolves against the JSONL's parent dir.
    # This is the common case when results are relocated without rewriting
    # their embedded paths.
    candidate = results_jsonl_path.parent / checkpoint_path
    if candidate.exists():
        return candidate
    # Final fallback: return the original (probably-nonexistent) path so
    # ``load_iqp_checkpoint`` can raise a clear FileNotFoundError.
    return checkpoint_path


def _row_id(row: dict[str, Any], *, fallback_index: int) -> str:
    """Build a stable, human-readable id for a row.

    Used everywhere a filename is derived from a row: hypergraph inst file,
    experiment inst file, search `.frg`, stdout capture. Includes the
    ``param_idx`` when present so different seeds of the same setting do
    not collide.
    """
    family = str(row["family"])
    n = int(row["n"])
    kernel = str(row["kernel"])
    init = str(row["init"])
    param_idx = row.get("param_idx")
    # No param_idx: fall back to the line index so identities stay unique.
    if param_idx is None:
        return f"{family}_n{n}_k{kernel}_i{init}_row{fallback_index}"
    # Typical case: param_idx is present, so the id encodes the seed index.
    return f"{family}_n{n}_k{kernel}_i{init}_pi{param_idx}"


def load_labeled_rows(
    results_jsonl_path: str | Path,
    *,
    ac_field: str = "ac_passes_primary_threshold",
) -> list[LabeledRow]:
    """Load run-scaling rows that have a usable plateau label and checkpoint.

    Args:
        results_jsonl_path: Path to a scaling ``results.jsonl`` (one JSON
            record per line).
        ac_field: Name of the boolean field carrying the anti-concentration
            result. The default matches the scaling runner's primary check;
            experiments that want a secondary AC signal can override.

    Returns:
        List of LabeledRow, one per successfully-loaded scaling row. Rows
        missing either the AC field or a checkpoint path are skipped with
        an INFO log line — we never silently drop data.
    """
    results_path = Path(results_jsonl_path)
    rows: list[LabeledRow] = []

    # Iterate JSONL line-by-line so a single malformed row doesn't poison
    # the whole load. `enumerate` tracks the source index for logging.
    for index, raw_line in enumerate(results_path.read_text(encoding="utf-8").splitlines()):
        # Blank lines are allowed in JSONL; skip without warning.
        if not raw_line.strip():
            continue
        source_row = json.loads(raw_line)

        # Skip rows that don't carry the AC result — those are scaling rows
        # where anti-concentration wasn't evaluated (e.g. large-n skips).
        if ac_field not in source_row:
            log.info(
                "load_labeled_rows: skipping row %d from %s because %s is missing",
                index,
                results_path,
                ac_field,
            )
            continue
        # Skip rows without a checkpoint — we need G to build the Forge inst.
        checkpoint_raw = source_row.get("ac_checkpoint_path")
        if not checkpoint_raw:
            log.info(
                "load_labeled_rows: skipping row %d from %s because ac_checkpoint_path is missing",
                index,
                results_path,
            )
            continue

        # Resolve + load the checkpoint to recover the exact hypergraph.
        checkpoint_path = _resolve_checkpoint_path(results_path, str(checkpoint_raw))
        model, _metadata = load_iqp_checkpoint(checkpoint_path)
        # Flip the AC boolean into the plateau-label convention. AC fails
        # (boolean = False) <=> the distribution is concentrated <=>
        # "PlateauObserved". AC passes (True) <=> "PlateauAbsent".
        plateau_observed: PlateauLabel = (
            "PlateauObserved" if not bool(source_row[ac_field]) else "PlateauAbsent"
        )
        # Normalize G to a compact unsigned byte array before shipping to
        # Forge-land. The export_to_forge writer assumes uint8.
        G = np.asarray(model.G, dtype=np.uint8)
        n = int(source_row["n"])
        rows.append(
            LabeledRow(
                row_id=_row_id(source_row, fallback_index=index),
                family=str(source_row["family"]),
                n=n,
                # Prefer the row-declared m; fall back to the matrix shape
                # if it's missing (old-format scaling results).
                m=int(source_row.get("m", G.shape[0])),
                kernel=str(source_row["kernel"]),
                init=str(source_row["init"]),
                G=G,
                plateau_observed=plateau_observed,
                source_row=source_row,
            )
        )

    return rows
