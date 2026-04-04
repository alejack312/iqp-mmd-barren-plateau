"""IQP model class — wraps generator matrix and parameter vector."""

from __future__ import annotations

import numpy as np

from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.iqp.expectation import iqp_expectation, iqp_expectation_exact


class IQPModel:
    """Parameterized IQP circuit model.

    Attributes:
        G: Generator matrix, shape (m, n), uint8
        theta: Current parameters, shape (m,)
        n: Number of qubits
        m: Number of generators
    """

    def __init__(self, G: np.ndarray, theta: np.ndarray | None = None):
        self.G = G.astype(np.uint8)
        self.m, self.n = G.shape
        if theta is None:
            self.theta = np.zeros(self.m)
        else:
            self.theta = np.asarray(theta, dtype=np.float64)

    @classmethod
    def from_family(
        cls,
        family: str,
        n: int,
        m: int | None = None,
        rng: np.random.Generator | None = None,
        **family_kwargs,
    ) -> "IQPModel":
        """Construct from a named circuit family."""
        if m is None:
            m = n
        # TODO: Week 1 (D1.3) preserve family / generation metadata on the model so
        # experiment records and Qiskit exports can recover the exact circuit provenance.
        # Read first: json https://docs.python.org/3/library/json.html ; pathlib.Path
        # https://docs.python.org/3/library/pathlib.html#pathlib.Path
        G = make_hypergraph(family=family, n=n, m=m, rng=rng, **family_kwargs)
        return cls(G)

    def expectation(
        self,
        a: np.ndarray,
        num_z_samples: int = 1024,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float]:
        """Estimate ⟨Z_a⟩_{q_θ}."""
        return iqp_expectation(self.theta, self.G, a, num_z_samples=num_z_samples, rng=rng)

    def expectation_exact(self, a: np.ndarray) -> float:
        """Exact ⟨Z_a⟩_{q_θ} (only for n ≤ 20)."""
        return iqp_expectation_exact(self.theta, self.G, a)
