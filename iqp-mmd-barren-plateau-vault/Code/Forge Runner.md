---
title: Forge Runner
tags:
  - code
  - runner
  - forge
---

# `iqp_bp.experiments.run_forge`

Exports IQP hypergraph instances to Forge-compatible format for structural modeling.

**File:** [`src/iqp_bp/experiments/run_forge.py`](../src/iqp_bp/experiments/run_forge.py)

## What Forge Is

[Forge](https://forge-fm.github.io/forge-documentation/) is a **bounded-model-finding tool** descended from Alloy, used here for formal structural reasoning about small instances.

> [!info] Why Forge
> The [[Project Overview|Part IV]] methodology uses Forge to:
> - Model hypergraph overlap patterns
> - Identify minimal plateau-inducing configurations
> - Explore symmetry-induced constraints
> - Analyze structural invariants of commuting generators

Forge is well suited for small-$n$ structural reasoning and counterexample discovery, where exhaustive Monte Carlo and Qiskit simulation are too expensive but the combinatorial structure still has enough variety to be worth searching.

## Current Runner Behavior

For each `(family, n)` with $n \le$ `max_n` (default 12):

1. Build `G` via [[Hypergraph Families|`make_hypergraph`]]
2. Call `export_to_forge(G, n, frg_path)` — writes `forge/runs/{family}_n{n}.frg`
3. Write a JSONL record with the export path + status

The runner is **export-only** right now; the actual Forge search is done manually.

## Config

```yaml
forge:
  max_n: 12

circuit:
  family: [product_state, lattice, erdos_renyi, complete_graph]
  n_qubits: [4, 6, 8, 9, 12]
```

## Supporting Module

- [[Forge Export|`iqp_bp.forge.export_instances.export_to_forge`]] — the `G → .frg` serializer

## Open Work

- **D9.2 / D9.3** — replace export-only mode with automated structural searches that save counterexamples/invariants back into Python-readable results.

## Related

- [[Forge Export]]
- [[Research Questions]] — Q2 (structural dependence)
- [[Project Overview]] — Part IV
- [[TODO Roadmap]]
