---
title: CLI
tags:
  - code
  - iqp_bp
  - entrypoint
---

# `iqp_bp.cli`

The single entry point for the `iqp_bp` package. Dispatches to one of four runners based on the subcommand.

**File:** [`src/iqp_bp/cli.py`](../src/iqp_bp/cli.py)

## Usage

```bash
python -m iqp_bp.cli run-scaling    configs/experiments/scaling_v1.yaml
python -m iqp_bp.cli run-qiskit     configs/experiments/qiskit_validation.yaml
python -m iqp_bp.cli run-validation configs/experiments/validation.yaml
python -m iqp_bp.cli run-forge      configs/experiments/forge_sprint.yaml
```

Also available as the `iqp-bp` console script (from `pyproject.toml`).

## Flags

- `config` — positional path to the experiment YAML
- `--dry-run` — print the merged config as JSON and exit

## Subcommand → Runner Map

| Subcommand | Runner module |
|---|---|
| `run-scaling` | [[Scaling Runner]] — `iqp_bp.experiments.run_scaling` |
| `run-qiskit` | [[Qiskit Runner]] — `iqp_bp.experiments.run_qiskit` |
| `run-validation` | [[Validation Runner]] — `iqp_bp.experiments.run_validation` |
| `run-forge` | [[Forge Runner]] — `iqp_bp.experiments.run_forge` |
| `grid-preview` | `iqp_bp.config.preview_config` *(planned)* |

## Dispatch Flow

```python
cfg = load_config(args.config)
if args.dry_run:
    print(json.dumps(cfg, indent=2, default=str))
    sys.exit(0)

if args.command == "run-scaling":
    from iqp_bp.experiments.run_scaling import run
elif args.command == "run-qiskit":
    from iqp_bp.experiments.run_qiskit import run
elif args.command == "run-validation":
    from iqp_bp.experiments.run_validation import run
else:
    from iqp_bp.experiments.run_forge import run
run(cfg)
```

Runners are imported lazily so optional dependencies (e.g. Qiskit, Forge) only fail if their subcommand is actually invoked.

## Open Work

- **D1.2** — add a `grid-preview` subcommand that validates the config against `configs/schema.yaml` and previews the resolved experiment grid before launching long sweeps.

## Related

- [[Config]]
- [[Scaling Runner]]
- [[Validation Runner]]
- [[Qiskit Runner]]
- [[Forge Runner]]
