"""CLI entry point for iqp_bp experiments.

Usage:
    python -m iqp_bp.cli grid-preview   configs/experiments/scaling_v1.yaml
    python -m iqp_bp.cli run-scaling    configs/experiments/scaling_v1.yaml
    python -m iqp_bp.cli run-training   configs/experiments/training_smoke.yaml
    python -m iqp_bp.cli run-qiskit     configs/experiments/qiskit_validation.yaml
    python -m iqp_bp.cli run-validation configs/experiments/validation.yaml
    python -m iqp_bp.cli run-forge      configs/experiments/forge_sprint.yaml
"""

from __future__ import annotations

import argparse
import sys
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="iqp-bp",
        description="IQP–MMD Barren Plateau experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in (
        "grid-preview",
        "run-scaling",
        "run-training",
        "run-qiskit",
        "run-validation",
        "run-forge",
    ):
        p = sub.add_parser(cmd)
        p.add_argument("config", help="Path to experiment YAML config")

    args = parser.parse_args()

    from iqp_bp.config import load_config
    cfg = load_config(args.config)

    if args.command == "grid-preview":
        from iqp_bp.config import resolve_experiment_grid
        grid = resolve_experiment_grid(cfg)
        print(json.dumps(grid, indent=2, default=str))
        sys.exit(0)

    if args.command == "run-scaling":
        from iqp_bp.experiments.run_scaling import run
    elif args.command == "run-training":
        from iqp_bp.experiments.run_training import run
    elif args.command == "run-qiskit":
        from iqp_bp.experiments.run_qiskit import run
    elif args.command == "run-validation":
        from iqp_bp.experiments.run_validation import run
    else:
        from iqp_bp.experiments.run_forge import run

    run(cfg)


if __name__ == "__main__":
    main()
