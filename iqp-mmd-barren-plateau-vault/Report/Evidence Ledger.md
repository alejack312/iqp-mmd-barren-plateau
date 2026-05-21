---
title: Evidence Ledger
tags:
  - report
  - evidence
  - provenance
status: generated-summary
---

# Evidence Ledger

This note records the artifact surface used by [[Final Findings - IQP MMD Barren Plateaus]]. The machine-readable ledger is generated at:

- `results/final_report/evidence_ledger.csv`
- `results/final_report/evidence_ledger.json`

Regenerate with:

```bash
python scripts/build_final_report_artifacts.py
```

## Core Evidence Table

| Label | Kind | Path | Rows | Exists |
|---|---|---|---:|---|
| training | jsonl | `results/scaling_ac_diverse/results.jsonl` | 111 | true |
| holdout_n | jsonl | `results/scaling_ac_holdout_n/results.jsonl` | 80 | true |
| holdout_families | jsonl | `results/scaling_ac_holdout_families/results.jsonl` | 92 | true |
| f4_training_main | jsonl | `results/forge_f4_manifests/results.jsonl` | 111 | true |
| f4_training_ablation | jsonl | `results/forge_f4_manifests_ablation/results.jsonl` | 111 | true |
| f4_holdout_n_main | jsonl | `results/forge_f4_holdout_n/results.jsonl` | 80 | true |
| f4_holdout_n_ablation | jsonl | `results/forge_f4_holdout_n_ablation/results.jsonl` | 80 | true |
| f4_holdout_families_main | jsonl | `results/forge_f4_holdout_families/results.jsonl` | 92 | true |
| f4_holdout_families_ablation | jsonl | `results/forge_f4_holdout_families_ablation/results.jsonl` | 92 | true |
| ac11_small_n_exact | jsonl | `results/ac_ghosh_kim/small_n_exact/results.jsonl` | 3 | true |
| ac11_large_n_sampled | jsonl | `results/ac_ghosh_kim/large_n_sampled/results.jsonl` | 3 | true |
| ac12_bandwidth_sweep | jsonl | `results/bandwidth_marginal_sweep/results.jsonl` | 4 | true |
| pauli_estimator_validation_n16 | json | `results/pauli_estimator_validation_n16.json` | | true |
| grid5000_native_iqp_mmd_ac | directory | `results/grid5000_iqp_mmd_ac` | | true |
| qiskit_validation | jsonl | `results/qiskit_validation/results.jsonl` | 0 | false |
| qiskit_validation_report_smoke | jsonl | `results/qiskit_validation_report_smoke/results.jsonl` | 4 | true |
| frozen_f4_predicate | markdown | `results/forge_f4_manifests/PREDICATE_FROZEN.md` | | true |

## Notes

- The full `results/qiskit_validation/results.jsonl` artifact is still absent; the report uses the smaller durable smoke artifact.
- Forge claims should cite the frozen predicate snapshot and the holdout result files together.
- AC claims should cite both the run-level JSONL and the relevant vault writeup, because the writeups contain interpretation and caveats not present in the raw rows.
- Large-n genomic claims are estimator-based. Do not describe them as exact enumeration.

## Linked Writeups

- [[Forge Pipeline Overview]]
- [[F4 Holdout n=10,12]]
- [[F4 Holdout Unseen Families]]
- [[Anti-Concentration]]
- [[AC11 Ghosh-Kim Learned AC Results 2026-05-08]]
- [[AC12 Bandwidth Sweep Results 2026-04-21]]
- [[Grid5000 Native iqp_mmd AC and Marginals 2026-05-03]]
- [[Pauli Estimator Scale Up 2026-04-24]]
- [[Qiskit Runner]]
