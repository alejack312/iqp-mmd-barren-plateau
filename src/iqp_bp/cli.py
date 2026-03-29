"""CLI entry point for iqp_bp experiments.

Usage:
    python -m iqp_bp.cli run-scaling configs/experiments/scaling_v1.yaml
    python -m iqp_bp.cli run-qiskit  configs/experiments/qiskit_validation.yaml
    python -m iqp_bp.cli run-forge   configs/experiments/forge_sprint.yaml
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="iqp-bp",
        description="IQP–MMD Barren Plateau experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    # TODO: Week 1 (D1.2) add a config-validation / grid-preview subcommand so the
    # locked experiment axes can be inspected before launching long sweeps.
    for cmd in ("run-scaling", "run-qiskit", "run-forge"):
        p = sub.add_parser(cmd)
        p.add_argument("config", help="Path to experiment YAML config")
        p.add_argument("--dry-run", action="store_true", help="Print config and exit")

    args = parser.parse_args()

    from iqp_bp.config import load_config
    cfg = load_config(args.config)

    if args.dry_run:
        import json
        print(json.dumps(cfg, indent=2, default=str))
        sys.exit(0)

    if args.command == "run-scaling":
        from iqp_bp.experiments.run_scaling import run
    elif args.command == "run-qiskit":
        from iqp_bp.experiments.run_qiskit import run
    # Add a dedicated grid-preview command.
    elif args.command == "grid-preview":
        from iqp_bp.config import preview_config
        preview_config(cfg)
    # Add a dedicated validation command.
    elif args.command == "run-validation":
        from iqp_bp.experiments.run_validation import run
    else:
        from iqp_bp.experiments.run_forge import run

    run(cfg)


if __name__ == "__main__":
    main()
