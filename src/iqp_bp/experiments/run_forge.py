"""Forge structural modeling experiment runner."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.forge import (
    build_query,
    detect_plateau_agreement,
    detect_status,
    emit_experiment_instance,
    export_to_forge,
    load_labeled_rows,
    run_racket,
)
from iqp_bp.forge.label_loader import LabeledRow
from iqp_bp.forge.parser import extract_witness_block
from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.rng import STREAM_FORGE, experiment_stream_bundle

log = logging.getLogger(__name__)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _extract_family_kwargs(circuit_cfg: dict[str, Any], family: str) -> dict[str, Any]:
    if family == "lattice":
        return {
            "dimension": circuit_cfg.get("lattice", {}).get("dimension", 1),
            "range_": circuit_cfg.get("lattice", {}).get("range", 1),
        }
    if family == "erdos_renyi":
        p_edge = circuit_cfg.get("erdos_renyi", {}).get("p_edge", 0.1)
        if isinstance(p_edge, list):
            if len(p_edge) > 1:
                log.warning(
                    "run_forge: erdos_renyi.p_edge has %d values; using only the first (%s)",
                    len(p_edge),
                    p_edge[0],
                )
            p_edge = p_edge[0]
        return {"p_edge": p_edge}
    if family == "bounded_degree":
        return circuit_cfg.get("bounded_degree", {})
    if family == "dense":
        return {"expected_weight": circuit_cfg.get("dense", {}).get("expected_weight", 0.5)}
    if family == "community":
        return circuit_cfg.get("community", {})
    if family == "symmetric":
        return {"parity": circuit_cfg.get("symmetric", {}).get("parity", "even")}
    return {}


def _prepare_output_dirs(cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = output_dir / "runs"
    raw_dir = output_dir / "raw"
    results_path = output_dir / "results.jsonl"
    runs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, runs_dir, raw_dir, results_path


def _repo_forge_library() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    model_path = repo_root / "forge" / "models" / "hypergraph.frg"
    return model_path.read_text(encoding="utf-8")


def run(cfg: dict[str, Any]) -> None:
    """Entry point called by CLI."""
    forge_cfg = cfg.get("forge", {})
    mode = str(forge_cfg.get("mode", "template_search"))
    if mode == "template_search":
        _run_template_search(cfg)
        return
    if mode == "plateau_agreement":
        _run_plateau_agreement(cfg)
        return
    raise ValueError(f"unknown forge.mode: {mode!r}")


def _run_template_search(cfg: dict[str, Any]) -> None:
    """Existing F2 template-search flow."""
    _output_dir, runs_dir, raw_dir, results_path = _prepare_output_dirs(cfg)

    forge_cfg = cfg.get("forge", {})
    max_n = int(forge_cfg.get("max_n", 12))
    timeout_sec = int(forge_cfg.get("timeout_sec", 60))
    export_instances = bool(forge_cfg.get("export_instances", True))
    query_template = str(forge_cfg.get("query_template", "plateau_inducing_bounded"))
    thresholds_cfg = forge_cfg.get("thresholds", {})
    thresholds = {
        "max_weight": int(thresholds_cfg.get("max_weight", 3)),
        "overlap_threshold": int(thresholds_cfg.get("overlap_threshold", 2)),
    }

    circuit_cfg = cfg["circuit"]
    families = _as_list(circuit_cfg["family"])
    n_qubits_list = [int(n) for n in _as_list(circuit_cfg["n_qubits"]) if int(n) <= max_n]
    base_seed = int(cfg["experiment"].get("seed", 0))

    library = _repo_forge_library()

    for family in families:
        family_kwargs = _extract_family_kwargs(circuit_cfg, str(family))
        for n in n_qubits_list:
            m = n
            streams = experiment_stream_bundle(base_seed, "run_forge", family, n)
            rng = np.random.default_rng(streams[STREAM_FORGE])
            G = make_hypergraph(family=str(family), n=n, m=m, rng=rng, **family_kwargs)
            actual_m = int(G.shape[0])
            stem = f"{family}_n{n}"

            instance_path = runs_dir / f"{stem}.frg"
            if export_instances:
                export_to_forge(G, n, instance_path)

            query = build_query(
                query_template,
                instance_name=f"candidate_{family}_n{n}",
                thresholds=thresholds,
                bounds={"n": n, "m": actual_m},
            )
            search_path = runs_dir / f"{stem}_search.frg"
            search_path.write_text(f"{library}\n\n{query}", encoding="utf-8")

            raw = run_racket(search_path, timeout_sec=timeout_sec)
            combined_output = raw.stdout
            if raw.stderr:
                combined_output = (
                    f"{combined_output}\n---STDERR---\n{raw.stderr}"
                    if combined_output
                    else raw.stderr
                )

            status = raw.status
            if raw.status == "unknown":
                parsed_status = detect_status(combined_output)
                if parsed_status != "unknown":
                    status = parsed_status
                elif raw.returncode not in {None, 0}:
                    status = "error"
            if raw.status == "unknown" and status == "unknown":
                log.warning("run_forge: unable to classify output for %s", search_path)

            witness = extract_witness_block(combined_output) if status == "sat" else None

            raw_path = raw_dir / f"{stem}_stdout.txt"
            raw_path.write_text(combined_output, encoding="utf-8")

            witness_path = None
            if witness is not None:
                witness_path = raw_dir / f"{stem}_witness.frg"
                witness_path.write_text(witness, encoding="utf-8")

            record = {
                "family": str(family),
                "n": n,
                "m": actual_m,
                "mode": "template_search",
                "instance_path": str(instance_path) if export_instances else None,
                "search_path": str(search_path),
                "query_template": query_template,
                "thresholds": thresholds,
                "bounds": {"n": n, "m": actual_m},
                "status": status,
                "racket_available": raw.status != "skipped_no_racket",
                "witness_path": str(witness_path) if witness_path is not None else None,
                "raw_stdout_path": str(raw_path),
                "elapsed_sec": raw.elapsed_sec,
                "returncode": raw.returncode,
            }
            with open(results_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            log.info(
                "run_forge: family=%s n=%d m=%d status=%s elapsed=%.2fs%s",
                family,
                n,
                actual_m,
                status,
                raw.elapsed_sec,
                " (witness saved)" if witness_path is not None else "",
            )


def _run_plateau_agreement(cfg: dict[str, Any]) -> None:
    """Run one Forge agreement query per labeled scaling row."""
    _output_dir, runs_dir, raw_dir, results_path = _prepare_output_dirs(cfg)

    forge_cfg = cfg.get("forge", {})
    timeout_sec = int(forge_cfg.get("timeout_sec", 60))
    query_template = str(forge_cfg.get("query_template", "plateau_agreement"))
    thresholds_cfg = forge_cfg.get("thresholds", {})
    thresholds = {
        "max_weight": int(thresholds_cfg.get("max_weight", 3)),
        "overlap_threshold": int(thresholds_cfg.get("overlap_threshold", 2)),
    }
    plateau_cfg = forge_cfg.get("plateau_agreement", {})
    if "label_source" not in plateau_cfg:
        raise ValueError("forge.plateau_agreement.label_source is required in plateau mode")

    rows = load_labeled_rows(
        plateau_cfg["label_source"],
        ac_field=str(plateau_cfg.get("ac_field", "ac_passes_primary_threshold")),
    )
    library = _repo_forge_library()

    if not rows:
        log.warning("run_forge: plateau_agreement found no labeled rows to process")
        return

    summary_counts = {
        "agree": 0,
        "disagree": 0,
        "unknown": 0,
        "error": 0,
        "timeout": 0,
        "skipped_no_racket": 0,
    }

    # Dedupe rows by (family, n, kernel, init). Scaling emits up to 5 param_idx
    # rows per setting that share the same hypergraph and plateau label, and
    # therefore produce byte-identical Forge queries — running Racket once per
    # group cuts F3 wall-time ~5x. `m` and `plateau_observed` are included in
    # the key defensively: `m` appears in query bounds and `plateau_observed`
    # is embedded in the emitted experiment instance, so rows that differ on
    # either must not share a query.
    groups: "OrderedDict[tuple[str, int, int, str, str, str], list[LabeledRow]]" = OrderedDict()
    for row in rows:
        key = (row.family, row.n, row.m, row.kernel, row.init, row.plateau_observed)
        groups.setdefault(key, []).append(row)

    if len(groups) < len(rows):
        log.info(
            "run_forge: plateau_agreement deduped %d input rows into %d unique queries",
            len(rows),
            len(groups),
        )

    for group_rows in groups.values():
        rep = group_rows[0]
        experiment_inst_name = f"experiment_{rep.row_id}"
        instance_path = runs_dir / f"{rep.row_id}_instance.frg"
        emit_experiment_instance(rep, instance_path, instance_name=experiment_inst_name)

        bounds_inst_name = f"bounds_{rep.row_id}"
        query = build_query(
            query_template,
            instance_name=f"row_{rep.row_id}",
            thresholds=thresholds,
            bounds={"n": rep.n, "m": rep.m},
            context={"bounds_inst": bounds_inst_name},
        )
        instance_text = instance_path.read_text(encoding="utf-8")
        search_path = runs_dir / f"{rep.row_id}_agreement.frg"
        search_path.write_text(
            (
                f"{library}\n\n"
                f"{instance_text}\n"
                f"inst {bounds_inst_name} {{\n"
                f"  hypergraph_{rep.n}_{rep.m}\n"
                f"  {experiment_inst_name}\n"
                f"}}\n\n"
                f"{query}"
            ),
            encoding="utf-8",
        )

        raw = run_racket(search_path, timeout_sec=timeout_sec)
        if raw.status in {"timeout", "error", "skipped_no_racket"}:
            agreement = raw.status
        else:
            agreement = detect_plateau_agreement(raw.stdout, raw.stderr)
            if agreement == "unknown" and raw.returncode not in {None, 0}:
                agreement = "error"

        combined_output = raw.stdout
        if raw.stderr:
            combined_output = (
                f"{combined_output}\n---STDERR---\n{raw.stderr}"
                if combined_output
                else raw.stderr
            )

        raw_path = raw_dir / f"{rep.row_id}_stdout.txt"
        raw_path.write_text(combined_output, encoding="utf-8")

        for row in group_rows:
            structurally_predicted: bool | None
            if agreement == "agree":
                structurally_predicted = row.plateau_observed == "PlateauObserved"
            elif agreement == "disagree":
                structurally_predicted = row.plateau_observed != "PlateauObserved"
            else:
                structurally_predicted = None

            if agreement not in summary_counts:
                summary_counts["error"] += 1
            else:
                summary_counts[agreement] += 1

            record = {
                "row_id": row.row_id,
                "family": row.family,
                "n": row.n,
                "m": row.m,
                "kernel": row.kernel,
                "init": row.init,
                "mode": "plateau_agreement",
                "query_template": query_template,
                "thresholds": thresholds,
                "plateau_observed": row.plateau_observed,
                "structurally_predicted": structurally_predicted,
                "agreement": agreement,
                "instance_path": str(instance_path),
                "search_path": str(search_path),
                "raw_stdout_path": str(raw_path),
                "elapsed_sec": raw.elapsed_sec,
                "returncode": raw.returncode,
                "racket_available": raw.status != "skipped_no_racket",
                "query_representative_row_id": rep.row_id,
            }
            with open(results_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            log.info(
                "run_forge: row=%s agreement=%s elapsed=%.2fs%s",
                row.row_id,
                agreement,
                raw.elapsed_sec,
                "" if row.row_id == rep.row_id else f" (shared with {rep.row_id})",
            )

    log.info(
        (
            "run_forge: plateau_agreement processed %d rows via %d queries "
            "(agree=%d disagree=%d unknown=%d error=%d timeout=%d skipped_no_racket=%d)"
        ),
        len(rows),
        len(groups),
        summary_counts["agree"],
        summary_counts["disagree"],
        summary_counts["unknown"],
        summary_counts["error"],
        summary_counts["timeout"],
        summary_counts["skipped_no_racket"],
    )
