"""Forge structural modeling experiment runner.

Exports hypergraph instances to Forge-compatible format and
invokes Forge to find structural invariants or counterexamples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.hypergraph.families import make_hypergraph

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> None:
    """Entry point called by CLI."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    forge_cfg = cfg.get("forge", {})
    max_n = forge_cfg.get("max_n", 12)
    export = forge_cfg.get("export_instances", True)

    families = cfg["circuit"]["family"]
    if not isinstance(families, list):
        families = [families]
    n_qubits_list = [n for n in cfg["circuit"]["n_qubits"] if n <= max_n]

    from iqp_bp.forge.export_instances import export_to_forge

    # TODO: Week 7 (D9.2/D9.3) replace export-only mode with automated structural
    # searches that save counterexamples / invariants back into Python-readable results.
    # Read first: Forge docs https://forge-fm.github.io/forge-documentation/ ;
    # Forge constraints https://forge-fm.github.io/forge-documentation/building-models/constraints/constraints.html ;
    # Forge options https://forge-fm.github.io/forge-documentation/running-models/options.html ;
    # json https://docs.python.org/3/library/json.html
    for family in families:
        for n in n_qubits_list:
            m = n
            rng = np.random.default_rng(cfg["experiment"]["seed"])
            G = make_hypergraph(family=family, n=n, m=m, rng=rng)
            actual_m = G.shape[0]

            if export:
                frg_path = Path("forge/runs") / f"{family}_n{n}.frg"
                frg_path.parent.mkdir(parents=True, exist_ok=True)
                export_to_forge(G, n, frg_path)
                log.info(f"Exported {frg_path}")

            record = {
                "family": family,
                "n": n,
                "m": actual_m,
                "exported": str(frg_path) if export else None,
                "status": "exported — run Forge manually",
            }
            with open(output_dir / "instances.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
