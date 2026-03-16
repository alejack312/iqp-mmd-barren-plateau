"""Graph-structured Energy-Based Models using JAX/Flax.

Provides MaskedMLP-based and GCN-based EBM variants for learning
distributions over graph-structured binary data.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import networkx as nx
from qml_benchmarks.models.energy_based_model import EnergyBasedModel
from qml_benchmarks.model_utils import mmd_loss, median_heuristic


class MaskedMLP(nn.Module):
    """MLP with symmetric masked weight matrices reflecting graph structure."""

    n_layers: int
    mask: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        for i in range(self.n_layers - 1):
            weights = self.param(f"weights_{i}", jax.nn.initializers.lecun_normal(), (dim, dim))
            weights = (weights + weights.T) / 2
            weights = weights * self.mask
            bias = self.param(f"bias{i}", jax.nn.initializers.zeros, (dim,))
            x = jnp.dot(x, weights) + bias
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


class DeepGraphEBM(EnergyBasedModel):
    """Energy-based model with graph-structured MaskedMLP energy function.

    Args:
        n_layers: Number of hidden layers in the MaskedMLP.
        G: NetworkX graph object defining the structure.
    """

    def __init__(
        self,
        learning_rate=0.001,
        batch_size=32,
        max_steps=10000,
        cdiv_steps=100,
        convergence_interval=None,
        random_state=42,
        jit=True,
        G=None,
        n_layers=1,
        mmd_kwargs=None,
    ):
        super().__init__(
            dim=None,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            cdiv_steps=cdiv_steps,
            convergence_interval=convergence_interval,
            random_state=random_state,
            jit=jit,
        )
        if G is None:
            raise ValueError("Must specify graph G.")
        if mmd_kwargs is None:
            mmd_kwargs = {"n_samples": 1000, "n_steps": 1000, "sigma": 1.0}

        self.n_layers = n_layers
        self.G = G
        self.adj_matrix = jnp.array(nx.adjacency_matrix(G).toarray())
        self.mmd_kwargs = mmd_kwargs
        self.model = None

    def initialize(self, x):
        dim = x.shape[1]
        if not isinstance(dim, int):
            raise NotImplementedError("dim must be an integer.")
        self.dim = dim
        self.model = MaskedMLP(n_layers=self.n_layers, mask=self.adj_matrix)
        self.params_ = self.model.init(self.generate_key(), x)

    def energy(self, params, x):
        return self.model.apply(params, x)

    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        sigma = self.mmd_kwargs["sigma"]
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        score = np.mean(
            [mmd_loss(X, self.sample(self.mmd_kwargs["n_samples"], self.mmd_kwargs["n_steps"]), s) for s in sigmas]
        )
        return float(-score)


_batched_matmul = jax.jit(jax.vmap(jax.numpy.matmul, in_axes=(0, 0)))


class GCNLayer(nn.Module):
    """Single Graph Convolutional Network layer."""

    c_out: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        num_neighbours = adj_matrix.sum(axis=-1, keepdims=True)
        node_feats = nn.Dense(features=self.c_out, name="projection")(node_feats)
        node_feats = _batched_matmul(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


class GCNModel(nn.Module):
    """Multi-layer GCN with global average pooling and dense output."""

    hidden_layers: list

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        for c_out in self.hidden_layers:
            node_feats = GCNLayer(c_out=c_out)(node_feats, adj_matrix)
            node_feats = nn.relu(node_feats)
        graph_feats = jnp.mean(node_feats, axis=1)
        return nn.Dense(features=1)(graph_feats)


class GraphEBM(EnergyBasedModel):
    """Energy-based model with GCN energy function.

    Args:
        hidden_layers: Output dimensions per GCN hidden layer.
        adj_matrix: Adjacency matrix of the graph.
    """

    def __init__(
        self,
        learning_rate=0.001,
        batch_size=32,
        max_steps=10000,
        cdiv_steps=100,
        convergence_interval=None,
        random_state=42,
        jit=True,
        hidden_layers=None,
        adj_matrix=None,
        mmd_kwargs=None,
    ):
        super().__init__(
            dim=None,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            cdiv_steps=cdiv_steps,
            convergence_interval=convergence_interval,
            random_state=random_state,
            jit=jit,
        )
        if adj_matrix is None:
            raise ValueError("Adjacency matrix must be specified.")
        if hidden_layers is None:
            hidden_layers = [8, 4]
        if mmd_kwargs is None:
            mmd_kwargs = {"n_samples": 1000, "n_steps": 1000, "sigma": 1.0}

        self.hidden_layers = hidden_layers
        self.adj_matrix = adj_matrix
        self.mmd_kwargs = mmd_kwargs

    def initialize(self, x):
        dim = x.shape[1]
        if not isinstance(dim, int):
            raise NotImplementedError("dim must be an integer.")
        x = jnp.expand_dims(x, -1)
        self.dim = dim
        self.model = GCNModel(hidden_layers=self.hidden_layers)
        self.params_ = self.model.init(self.generate_key(), x[:1], jnp.array([self.adj_matrix]))

    def energy(self, params, x):
        x = jnp.expand_dims(x, -1)
        return self.model.apply(params, x, jnp.array([self.adj_matrix] * x.shape[0]))

    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        sigma = self.mmd_kwargs["sigma"]
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        score = np.mean(
            [mmd_loss(X, self.sample(self.mmd_kwargs["n_samples"], self.mmd_kwargs["n_steps"]), s) for s in sigmas]
        )
        return float(-score)
