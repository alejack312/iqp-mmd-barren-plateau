"""Forge structural modeling utilities.

This package is the Python-side bridge to the Forge (Alloy-based) model at
``forge/models/hypergraph.frg``. The pipeline it orchestrates looks like:

    scaling results.jsonl (with AC labels + checkpoints)
         │
         ▼
    label_loader.load_labeled_rows   — materialize rows + G from checkpoints
         │
         ▼
    export_instances.export_to_forge — serialize G to a Forge hypergraph inst
    experiment_emitter.emit_experiment_instance — append the Experiment inst
         │
         ▼
    query_templates.build_query      — render a test-expect query body
         │
         ▼
    runner.run_racket                — shell out to racket <.frg>
         │
         ▼
    parser.detect_plateau_agreement  — classify stdout as agree/disagree

Only the names re-exported below are part of the package's public API.
"""

from .experiment_emitter import emit_experiment_instance
from .export_instances import export_to_forge
from .label_loader import LabeledRow, load_labeled_rows
from .parser import detect_plateau_agreement, detect_status
from .query_templates import build_query
from .runner import ForgeResult, run_racket

__all__ = [
    "ForgeResult",
    "LabeledRow",
    "build_query",
    "detect_plateau_agreement",
    "detect_status",
    "emit_experiment_instance",
    "export_to_forge",
    "load_labeled_rows",
    "run_racket",
]
