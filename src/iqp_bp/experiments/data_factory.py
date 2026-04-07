"""Runtime dataset factory for scaling and validation experiments.

This module keeps dataset generation aligned with the `iqp_bp` experiment
configuration contract while avoiding ad hoc generators inside runner modules.
"""

from __future__ import annotations

from math import isqrt
from typing import Any

import numpy as np


def make_dataset(
    dataset_cfg: dict[str, Any],
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Materialize a binary dataset and JSON-serializable provenance metadata."""
    dataset_type = str(dataset_cfg.get("type", "product_bernoulli"))
    n_samples = int(dataset_cfg.get("n_samples", 1000))
    rng = np.random.default_rng(seed)

    if dataset_type == "product_bernoulli":
        data = rng.integers(0, 2, size=(n_samples, n), dtype=np.uint8)
        return data, {
            "type": dataset_type,
            "n_samples": n_samples,
            "seed": int(seed),
        }

    if dataset_type == "ising":
        data, metadata = _make_ising_dataset(
            n=n,
            n_samples=n_samples,
            cfg=dataset_cfg.get("ising", {}),
            rng=rng,
        )
        metadata.update(
            {
                "type": dataset_type,
                "n_samples": n_samples,
                "seed": int(seed),
            }
        )
        return data, metadata

    if dataset_type == "binary_mixture":
        data, metadata = _make_binary_mixture_dataset(
            n=n,
            n_samples=n_samples,
            cfg=dataset_cfg.get("binary_mixture", {}),
            rng=rng,
        )
        metadata.update(
            {
                "type": dataset_type,
                "n_samples": n_samples,
                "seed": int(seed),
            }
        )
        return data, metadata

    raise NotImplementedError(f"Dataset type {dataset_type!r} not yet implemented")


def _make_binary_mixture_dataset(
    *,
    n: int,
    n_samples: int,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a binarized Gaussian mixture in {0,1}^n."""
    n_modes = int(cfg.get("n_modes", 4))
    noise = float(cfg.get("noise", 0.1))
    threshold = float(cfg.get("threshold", 0.0))

    centers = rng.normal(loc=0.0, scale=1.0, size=(n_modes, n))
    assignments = rng.integers(0, n_modes, size=n_samples)
    latent = centers[assignments] + rng.normal(loc=0.0, scale=noise, size=(n_samples, n))
    data = (latent > threshold).astype(np.uint8)
    return data, {
        "n_modes": n_modes,
        "noise": noise,
        "threshold": threshold,
    }


def _make_ising_dataset(
    *,
    n: int,
    n_samples: int,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a sparse Ising-like binary dataset from runtime config."""
    beta = float(cfg.get("beta", 1.0))
    coupling_std = float(cfg.get("coupling_std", 1.0))
    topology = str(cfg.get("topology", "grid_2d"))
    burn_in_sweeps = int(cfg.get("burn_in_sweeps", max(20 * n, 100)))
    thinning = int(cfg.get("thinning", 2))
    num_chains = int(cfg.get("num_chains", min(4, max(1, n_samples))))

    adjacency, topology_metadata = _make_ising_topology(
        n=n,
        topology=topology,
        coupling_std=coupling_std,
        rng=rng,
    )
    data = _sample_ising_binary(
        adjacency=adjacency,
        beta=beta,
        n_samples=n_samples,
        burn_in_sweeps=burn_in_sweeps,
        thinning=thinning,
        num_chains=num_chains,
        rng=rng,
    )
    metadata = {
        "beta": beta,
        "coupling_std": coupling_std,
        "topology": topology,
        "burn_in_sweeps": burn_in_sweeps,
        "thinning": thinning,
        "num_chains": num_chains,
    }
    metadata.update(topology_metadata)
    return data, metadata


def _make_ising_topology(
    *,
    n: int,
    topology: str,
    coupling_std: float,
    rng: np.random.Generator,
) -> tuple[list[list[tuple[int, float]]], dict[str, Any]]:
    """Build sparse adjacency lists with Gaussian couplings."""
    edges: list[tuple[int, int]] = []
    metadata: dict[str, Any] = {}

    if topology == "grid_2d":
        side = isqrt(n)
        if side * side != n:
            raise ValueError(f"grid_2d Ising dataset requires a perfect-square n, got {n}")
        for row in range(side):
            for col in range(side):
                node = row * side + col
                if col + 1 < side:
                    edges.append((node, node + 1))
                if row + 1 < side:
                    edges.append((node, node + side))
        metadata["grid_side"] = side
    elif topology == "erdos_renyi":
        average_degree = float(2.0)
        p_edge = min(1.0, average_degree / max(n, 1))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p_edge:
                    edges.append((i, j))
        metadata["average_degree_target"] = average_degree
        metadata["p_edge"] = p_edge
    else:
        raise ValueError(
            f"Unsupported Ising topology {topology!r}; choose 'grid_2d' or 'erdos_renyi'"
        )

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    coupling_scale = coupling_std / np.sqrt(max(n, 1))
    for i, j in edges:
        weight = float(rng.normal(loc=0.0, scale=coupling_scale))
        adjacency[i].append((j, weight))
        adjacency[j].append((i, weight))

    metadata["num_edges"] = len(edges)
    return adjacency, metadata


def _sample_ising_binary(
    *,
    adjacency: list[list[tuple[int, float]]],
    beta: float,
    n_samples: int,
    burn_in_sweeps: int,
    thinning: int,
    num_chains: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an Ising model with sequential Gibbs updates on sparse couplings."""
    n = len(adjacency)
    allocations = [n_samples // num_chains for _ in range(num_chains)]
    for index in range(n_samples % num_chains):
        allocations[index] += 1

    samples = np.zeros((n_samples, n), dtype=np.uint8)
    cursor = 0
    for chain_samples in allocations:
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=n).astype(np.int8)
        total_sweeps = burn_in_sweeps + max(chain_samples - 1, 0) * thinning + 1
        for sweep in range(total_sweeps):
            for site in range(n):
                local_field = 0.0
                for neighbor, weight in adjacency[site]:
                    local_field += weight * float(spins[neighbor])
                prob_up = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))
                spins[site] = 1 if rng.random() < prob_up else -1
            if sweep >= burn_in_sweeps and (sweep - burn_in_sweeps) % thinning == 0:
                samples[cursor] = ((spins + 1) // 2).astype(np.uint8)
                cursor += 1
                if cursor >= n_samples:
                    break
    return samples
