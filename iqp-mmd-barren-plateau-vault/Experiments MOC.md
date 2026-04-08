---
title: Experiments MOC
tags:
  - moc
  - experiments
---

# Experiments — Map of Content

The four experiment runners under `iqp_bp.experiments` and their configs.

## Runners

- [[Scaling Runner]] — `run_scaling.py`, the main gradient-variance sweep
- [[Validation Runner]] — `run_validation.py`, the anti-concentration deterministic check
- [[Qiskit Runner]] — `run_qiskit.py`, the Qiskit cross-check (stub)
- [[Forge Runner]] — `run_forge.py`, Forge structural modeling export

## Supporting Module

- [[Data Factory]] — `data_factory.py`, on-the-fly dataset generation

## Configs

All experiments are config-driven via YAML files under [`configs/experiments/`](../configs/experiments/):

- `scaling_v1.yaml` — main scaling sweep
- `scaling_ac_smoke.yaml` — anti-concentration smoke test
- `validation.yaml` — anti-concentration validation
- `validation_from_checkpoint.yaml` — validation from saved checkpoint
- `qiskit_validation.yaml` — Qiskit cross-check
- `forge_sprint.yaml` — Forge structural modeling

See [[Configs]] for details.

## Walkthroughs

- [[How a Scaling Run Works]] — end-to-end walkthrough of the main runner

## CLI

All runners are invoked via the unified [[CLI]]:

```bash
python -m iqp_bp.cli run-scaling    configs/experiments/scaling_v1.yaml
python -m iqp_bp.cli run-validation configs/experiments/validation.yaml
python -m iqp_bp.cli run-qiskit     configs/experiments/qiskit_validation.yaml
python -m iqp_bp.cli run-forge      configs/experiments/forge_sprint.yaml
```

## Output Formats

| Runner | Primary output | Secondary outputs |
|---|---|---|
| Scaling | `results.jsonl` | `anti_concentration/*`, `checkpoints/*.npz` |
| Validation | `*.summary.json` | `*.thresholds.csv`, `*.threshold.png`, `*.diagnostics.png` |
| Qiskit | `results.jsonl` (stub) | — |
| Forge | `instances.jsonl` | `forge/runs/*.frg` |

## Related

- [[Code MOC]]
- [[CLI]]
- [[Configs]]
- [[How a Scaling Run Works]]
