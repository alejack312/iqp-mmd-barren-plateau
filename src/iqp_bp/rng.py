"""Deterministic seeding utilities.

All experiment components must seed through this module to ensure full
reproducibility. We support both NumPy and JAX RNG.

Canonical stream names
----------------------
Each stream has a distinct responsibility so that refactoring one component
(e.g. changing ``num_a_samples``) does not shift any other component's draws:

* ``STREAM_CIRCUIT``    – hypergraph / circuit structure sampling
* ``STREAM_DATA``       – dataset / training-sample generation
* ``STREAM_THETA``      – parameter-vector initialisation (one sub-seed per index)
* ``STREAM_KERNEL``     – kernel Z-word (a-sample) draws in MMD estimators
* ``STREAM_ESTIMATION`` – IQP expectation and gradient z-sample draws
* ``STREAM_QISKIT``     – Qiskit shot sampling and Aer noise-model randomness
* ``STREAM_FORGE``      – Forge structural-search instance generation

Use :func:`experiment_stream_bundle` to obtain the full bundle for a given
experiment coordinate instead of constructing stream labels ad hoc.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

# ---------------------------------------------------------------------------
# Canonical stream names
# ---------------------------------------------------------------------------

STREAM_CIRCUIT: str = "circuit"
STREAM_DATA: str = "data"
STREAM_THETA: str = "theta"
STREAM_KERNEL: str = "kernel"
STREAM_ESTIMATION: str = "estimation"
STREAM_QISKIT: str = "qiskit"
STREAM_FORGE: str = "forge"

CANONICAL_STREAMS: tuple[str, ...] = (
    STREAM_CIRCUIT,
    STREAM_DATA,
    STREAM_THETA,
    STREAM_KERNEL,
    STREAM_ESTIMATION,
    STREAM_QISKIT,
    STREAM_FORGE,
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def make_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy Generator."""
    return np.random.default_rng(seed)


def make_jax_key(seed: int):
    """Return a JAX PRNGKey from an integer seed."""
    import jax
    return jax.random.PRNGKey(seed)


def split_seeds(base_seed: int, n: int) -> list[int]:
    """Derive n independent seeds from a base seed."""
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


def experiment_stream_bundle(base_seed: int, *coord_parts: object) -> dict[str, int]:
    """Return the canonical named seed bundle for an experiment coordinate.

    Produces one deterministic seed integer per :data:`CANONICAL_STREAMS` entry.
    Callers should use this instead of constructing stream labels ad hoc so that
    the full set of streams stays consistent across modules.

    Args:
        base_seed: Top-level experiment seed (from config).
        *coord_parts: Arbitrary coordinate parts (family, n, …) that uniquely
            identify the experiment setting within the sweep.

    Returns:
        Dict mapping each canonical stream name to a stable seed integer in
        ``[0, 2**31 - 1)``.
    """
    return named_seed_streams(base_seed, CANONICAL_STREAMS, *coord_parts)


def split_rng(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Derive *n* independent child generators from a parent generator.

    Draws one seed integer from *rng* and uses :class:`numpy.random.SeedSequence`
    to spawn ``n`` independent children.  This keeps the parent state advancing
    by exactly one draw regardless of ``n``, and ensures the children are
    statistically independent of each other.

    Args:
        rng: Parent generator (consumed by one integer draw).
        n: Number of child generators to create.

    Returns:
        List of ``n`` freshly seeded :class:`numpy.random.Generator` instances.
    """
    parent_seed = int(rng.integers(0, 2**31 - 1))
    ss = np.random.SeedSequence(parent_seed)
    return [np.random.default_rng(s) for s in ss.spawn(n)]
