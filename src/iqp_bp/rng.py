"""Deterministic seeding utilities.

All experiment components must seed through this module to ensure full
reproducibility. We support both NumPy and JAX RNG.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy Generator."""
    return np.random.default_rng(seed)


def make_jax_key(seed: int):
    """Return a JAX PRNGKey from an integer seed."""
    import jax
    return jax.random.PRNGKey(seed)


def split_seeds(base_seed: int, n: int) -> list[int]:
    """Derive n independent seeds from a base seed."""
    # TODO: Week 1 (D1.2) reserve named seed streams for circuit, data, theta,
    # kernel, and Qiskit noise sampling so cross-checks are exactly reproducible.
    # Read first: NumPy Generator https://numpy.org/doc/stable/reference/random/generator.html ;
    # JAX random https://docs.jax.dev/en/latest/jax.random.html
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 2**31 - 1, size=n).tolist()


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable integer seed from a base seed and labeled parts.

    This is used when experiments need deterministic, named seed streams whose
    values stay stable across refactors of loop ordering.
    """
    payload = json.dumps(
        {"base_seed": int(base_seed), "parts": parts},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def named_seed_streams(
    base_seed: int,
    stream_names: list[str] | tuple[str, ...],
    *parts: object,
) -> dict[str, int]:
    """Return deterministic named seeds for the given experiment coordinate."""
    return {
        str(name): derive_seed(base_seed, *parts, "stream", str(name))
        for name in stream_names
    }
