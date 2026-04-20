---
title: RNG
tags:
  - code
  - iqp_bp
  - reproducibility
---

# `iqp_bp.rng`

Deterministic seeding utilities. Every source of randomness in the pipeline threads through this module so experiments stay reproducible across refactors.

**File:** [`src/iqp_bp/rng.py`](../src/iqp_bp/rng.py)

## Functions

### `make_rng(seed)`

Trivial wrapper around `np.random.default_rng(seed)`.

### `make_jax_key(seed)`

Returns `jax.random.PRNGKey(seed)`. Lazy JAX import.

### `split_seeds(base_seed, n)`

Derive $n$ independent seeds from a base seed via a fresh `Generator.integers` call. Used for legacy splits.

### `derive_seed(base_seed, *parts) -> int`

The **core primitive**. Derives a stable integer seed from a base seed plus labeled parts:

```python
payload = json.dumps(
    {"base_seed": int(base_seed), "parts": parts},
    sort_keys=True, separators=(",", ":"), default=str,
).encode("utf-8")
digest = hashlib.blake2b(payload, digest_size=8).digest()
return int.from_bytes(digest, "big") % (2**31 - 1)
```

**Why this matters:** seeds derived this way are **stable under refactors** of loop ordering. If you reorder the nested loops in `run_scaling.run`, each `(setting, param_idx)` still gets the same seeds because the derivation only depends on the labeled parts, not the iteration order.

### `named_seed_streams(base_seed, stream_names, *parts)`

Convenience wrapper that returns a dict `{name: derive_seed(base_seed, *parts, "stream", name)}`.

Used in `run_scaling.run`:

```python
streams = named_seed_streams(
    base_seed,
    ("circuit", "data"),
    "run_scaling",
    setting_key,
)
# streams["circuit"] = seed for make_hypergraph
# streams["data"]    = seed for make_dataset
```

## Open Work

- **P2** — reserve named seed streams for circuit, data, theta, kernel, and Qiskit noise sampling so cross-checks are exactly reproducible. Partial today: `circuit` and `data` are reserved; `theta` uses `derive_seed(..., "theta", idx)` inline.

## Determinism Contract

Every JSONL row written by the scaling runner is reproducible from `(experiment.seed, setting, param_idx)` alone. This is the reproducibility guarantee the pipeline makes.

## Related

- [[Scaling Runner]]
- [[How a Scaling Run Works]]
- [[Config]]
