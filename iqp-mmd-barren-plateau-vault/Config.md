---
title: Config
tags:
  - code
  - iqp_bp
  - config
---

# `iqp_bp.config`

Loads YAML experiment configs and merges with `configs/base.yaml` defaults.

**File:** [`src/iqp_bp/config.py`](../src/iqp_bp/config.py)

## Functions

### `load_config(path, base_path=_BASE_CONFIG_PATH) -> dict`

Loads an experiment YAML and deep-merges it over the base config.

```python
def load_config(path, base_path=_BASE_CONFIG_PATH):
    base = _load_yaml(base_path)
    override = _load_yaml(path)
    return _deep_merge(base, override)
```

### `_deep_merge(base, override)`

Recursive merge — nested dicts are merged key-by-key; leaves are replaced by `override`.

## Base Config Path

Resolved relative to the package: `configs/base.yaml` at the repo root.

## The Merge Pattern

Every experiment YAML under `configs/experiments/` is a **sparse override**. Fields not present in the experiment YAML fall back to `base.yaml`. This lets experiment files be small and focused on what's actually different.

Example:

```yaml
# configs/experiments/scaling_v1.yaml
experiment:
  name: scaling_v1
  seed: 42
  output_dir: results/scaling_v1

circuit:
  family: [product_state, lattice, erdos_renyi, complete_graph]
  n_qubits: [4, 9, 16, 25, 36]

kernel:
  type: gaussian
  bandwidth: [0.5, 1.0, 2.0]

init:
  scheme: [uniform, small_angle]
```

Everything else — `dataset`, `estimation`, `qiskit`, `forge` — comes from [[Configs#base.yaml]].

## Open Work

- **D1.2 / P1** — schema validation against `configs/schema.yaml`, plus persisting the fully resolved experiment grid alongside the JSONL output. Neither is implemented yet.

## Related

- [[Configs]]
- [[CLI]]
- [[Scaling Runner]]
