"""Deterministic seeding utilities.

All experiment components must seed through this module to ensure full
reproducibility. We support both NumPy and JAX RNG.
"""

from __future__ import annotations

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
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 2**31 - 1, size=n).tolist()
