---
title: Forge Export
tags:
  - code
  - iqp_bp
  - forge
---

# `iqp_bp.forge.export_instances`

Writes an IQP hypergraph `G` to a `.frg` file that Forge can load for structural modeling.

**File:** [`src/iqp_bp/forge/export_instances.py`](../src/iqp_bp/forge/export_instances.py)

## Public API

```python
def export_to_forge(G: np.ndarray, n: int, path: Path) -> None
```

## What It Does

Emits a Forge model containing:

- One `Qubit` atom per column of `G`
- One `Generator` atom per row of `G`
- The `acts_on` relation mapping each generator to the qubits it touches

## Use in the Runner

Called from [[Forge Runner]]:

```python
frg_path = Path("forge/runs") / f"{family}_n{n}.frg"
export_to_forge(G, n, frg_path)
```

Models are written to `forge/runs/` and then searched manually using the Forge CLI.

## Intended Use Cases

From the project methodology (Part IV):

- **Model hypergraph overlap patterns** — find minimal configurations where overlap exceeds a bound
- **Identify minimal plateau-inducing configurations** — search for the smallest generator sets that show specific structural signatures
- **Explore symmetry-induced constraints** — check whether enforcing parity symmetries rules out certain topologies
- **Structural invariants of commuting generators** — verify invariants Forge can check at small $n$ that would be intractable by Monte Carlo

## Open Work

- **D9.2 / D9.3** — automated structural searches integrated with the Python pipeline, rather than export-only mode

## Related

- [[Forge Runner]]
- [[Hypergraph Families]]
- [[Research Questions]] — Q2
