"""Ising model dataset generation via MCMC sampling.

Supports 2D square lattice and scale-free (Barabasi-Albert) network topologies.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import networkx as nx
import yaml


@dataclass
class IsingDataset:
    """Container for generated Ising dataset."""

    train_samples: np.ndarray
    test_samples: np.ndarray
    j_matrix: np.ndarray
    graph: nx.Graph
    magnetizations_train: np.ndarray
    magnetizations_test: np.ndarray
    energies_train: np.ndarray
    energies_test: np.ndarray
    settings: dict


def generate_ising_lattice(
    width: int = 3,
    temperature: float = 2.5,
    num_chains: int = 8,
    num_samples_per_chain: int = 2000,
    burn_in: int = 1000,
    num_train: int = 1000,
    num_test: int = 1000,
    periodic: bool = True,
    random_weights: bool = False,
    seed: int = 666,
    output_dir: str | Path = None,
) -> IsingDataset:
    """Generate Ising model dataset on a 2D square lattice.

    Uses MCMC sampling via numpyro (from qml_benchmarks).

    Args:
        width: Grid width (total spins = width^2).
        temperature: Ising temperature.
        num_chains: Number of independent MCMC chains.
        num_samples_per_chain: Samples per chain.
        burn_in: Warmup samples to discard.
        num_train: Number of training samples.
        num_test: Number of test samples.
        periodic: Use periodic boundary conditions.
        random_weights: Randomize coupling weights.
        seed: Random seed.
        output_dir: If provided, save CSVs and metadata here.

    Returns:
        IsingDataset with train/test samples and metadata.
    """
    import numpyro
    from qml_benchmarks.data.ising import IsingSpins, energy

    np.random.seed(seed)
    numpyro.set_host_device_count(min(num_chains, 8))

    n_spins = width * width
    G = nx.grid_2d_graph(width, width, periodic=periodic)
    J = nx.adjacency_matrix(G).toarray().astype(float)
    if random_weights:
        J = J * np.random.rand(*J.shape) * 2
    b = np.zeros(n_spins)

    total_samples = num_samples_per_chain * num_chains
    thinning = total_samples // (num_train + num_test)

    model = IsingSpins(N=n_spins, J=J, b=b, T=temperature)
    all_samples = model.sample(num_samples_per_chain, num_chains=num_chains, thinning=thinning, num_warmup=burn_in)
    all_samples = all_samples * 2 - 1

    magnetizations = np.array([np.mean(s) for s in all_samples])
    energies = np.array([energy(s, J, b) for s in all_samples])

    idxs = np.random.permutation(len(all_samples))
    train_idx = idxs[:num_train]
    test_idx = idxs[num_train : num_train + num_test]

    dataset = IsingDataset(
        train_samples=(all_samples[train_idx] + 1) // 2,
        test_samples=(all_samples[test_idx] + 1) // 2,
        j_matrix=J,
        graph=G,
        magnetizations_train=magnetizations[train_idx],
        magnetizations_test=magnetizations[test_idx],
        energies_train=energies[train_idx],
        energies_test=energies[test_idx],
        settings={
            "width": width,
            "temperature": temperature,
            "num_chains": num_chains,
            "num_samples_per_chain": num_samples_per_chain,
            "burn_in": burn_in,
            "num_train": num_train,
            "num_test": num_test,
            "random_weights": random_weights,
        },
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "train.csv", dataset.train_samples, delimiter=",", fmt="%d")
        np.savetxt(output_dir / "test.csv", dataset.test_samples, delimiter=",", fmt="%d")
        with open(output_dir / "settings.yaml", "w") as f:
            yaml.dump(dataset.settings, f, default_flow_style=False, sort_keys=False)

    return dataset


def generate_ising_network(
    n_nodes: int = 10,
    connectivity: int = 2,
    temperature: float = 2.0,
    num_chains: int = 8,
    num_samples_per_chain: int = 10000,
    burn_in: int = 1000,
    num_train: int = 5000,
    num_test: int = 10000,
    seed: int = 666,
    output_dir: str | Path = None,
) -> IsingDataset:
    """Generate Ising model dataset on a Barabasi-Albert scale-free network.

    Args:
        n_nodes: Number of nodes in the graph.
        connectivity: Number of edges per new node (BA model parameter m).
        temperature: Ising temperature.
        num_chains: Number of independent MCMC chains.
        num_samples_per_chain: Samples per chain.
        burn_in: Warmup samples to discard.
        num_train: Number of training samples.
        num_test: Number of test samples.
        seed: Random seed.
        output_dir: If provided, save CSVs and metadata here.

    Returns:
        IsingDataset with train/test samples and metadata.
    """
    import numpyro
    from qml_benchmarks.data.ising import IsingSpins, energy

    np.random.seed(seed)
    numpyro.set_host_device_count(min(num_chains, 8))

    G = nx.barabasi_albert_graph(n_nodes, connectivity)
    J = nx.adjacency_matrix(G).toarray().astype(float)
    b = np.zeros(n_nodes)

    total_samples = num_samples_per_chain * num_chains
    thinning = total_samples // (num_train + num_test)

    model = IsingSpins(N=n_nodes, J=J, b=b, T=temperature)
    all_samples = model.sample(num_samples_per_chain, num_chains=num_chains, thinning=thinning, num_warmup=burn_in)
    all_samples = all_samples * 2 - 1

    magnetizations = np.array([np.mean(s) for s in all_samples])
    energies = np.array([energy(s, J, b) for s in all_samples])

    idxs = np.random.permutation(len(all_samples))
    train_idx = idxs[:num_train]
    test_idx = idxs[num_train : num_train + num_test]

    dataset = IsingDataset(
        train_samples=(all_samples[train_idx] + 1) // 2,
        test_samples=(all_samples[test_idx] + 1) // 2,
        j_matrix=J,
        graph=G,
        magnetizations_train=magnetizations[train_idx],
        magnetizations_test=magnetizations[test_idx],
        energies_train=energies[train_idx],
        energies_test=energies[test_idx],
        settings={
            "n_nodes": n_nodes,
            "connectivity": connectivity,
            "temperature": temperature,
            "num_chains": num_chains,
            "num_samples_per_chain": num_samples_per_chain,
            "burn_in": burn_in,
            "num_train": num_train,
            "num_test": num_test,
        },
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "train.csv", dataset.train_samples, delimiter=",", fmt="%d")
        np.savetxt(output_dir / "test.csv", dataset.test_samples, delimiter=",", fmt="%d")
        nx.write_adjlist(G, str(output_dir / "graph.adjlist"))
        with open(output_dir / "settings.yaml", "w") as f:
            yaml.dump(dataset.settings, f, default_flow_style=False, sort_keys=False)

    return dataset
