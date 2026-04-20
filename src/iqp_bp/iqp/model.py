"""IQP model class — wraps generator matrix and parameter vector."""

from __future__ import annotations

import numpy as np

from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.iqp.expectation import iqp_expectation, iqp_expectation_exact


def _coerce_for_json(v: object) -> object:
    """Recursively convert numpy scalars and arrays to JSON-serializable Python types."""
    if hasattr(v, "item"):
        return v.item()  # numpy scalar → Python int/float/bool
    if isinstance(v, dict):
        return {str(k): _coerce_for_json(w) for k, w in v.items()}
    if isinstance(v, (list, tuple)):
        coerced = [_coerce_for_json(w) for w in v]
        return coerced if isinstance(v, list) else tuple(coerced)
    return v


class IQPModel:
    """Parameterized IQP circuit model.

    Attributes:
        G: Generator matrix, shape (m, n), uint8
        theta: Current parameters, shape (m,)
        n: Number of qubits
        m: Number of generators
        provenance: JSON-serializable dict recording the circuit family and
            generation details (family name, requested/generated n and m,
            family-specific kwargs, optional rng seed label).
    """

    def __init__(
        self,
        G: np.ndarray,
        theta: np.ndarray | None = None,
        provenance: dict | None = None,
    ):
        self.G = G.astype(np.uint8)
        self.m, self.n = G.shape
        if theta is None:
            self.theta = np.zeros(self.m)
        else:
            self.theta = np.asarray(theta, dtype=np.float64)
        self.provenance: dict = dict(provenance) if provenance is not None else {}

    @classmethod
    def from_family(
        cls,
        family: str,
        n: int,
        m: int | None = None,
        rng: np.random.Generator | None = None,
        rng_seed: int | None = None,
        **family_kwargs,
    ) -> "IQPModel":
        """Construct from a named circuit family.

        Args:
            family: Named hypergraph family (e.g. ``"complete_graph"``).
            n: Number of qubits.
            m: Number of generators; defaults to ``n`` when ``None``.
            rng: NumPy random generator for circuit sampling.
            rng_seed: Integer seed label stored in provenance so downstream
                consumers can recover the exact circuit generation seed without
                inspecting the generator's internal state.
            **family_kwargs: Extra keyword arguments forwarded to the family
                constructor (e.g. ``p_edge`` for ``"erdos_renyi"``).

        Returns:
            A new :class:`IQPModel` whose ``provenance`` attribute records
            ``family``, requested and generated ``m``, ``n``, ``family_kwargs``,
            and (if provided) ``rng_seed``.
        """
        m_requested = m
        if m is None:
            m = n
        G = make_hypergraph(family=family, n=n, m=m, rng=rng, **family_kwargs)
        m_generated = int(G.shape[0])
        provenance: dict = {
            "family": str(family),
            "n": int(n),
            "m_requested": int(m_requested) if m_requested is not None else int(n),
            "m_generated": m_generated,
            "family_kwargs": _coerce_for_json(dict(family_kwargs)),
        }
        if rng_seed is not None:
            provenance["rng_seed"] = _coerce_for_json(rng_seed)
        return cls(G, provenance=provenance)

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

    @staticmethod
    def _basis_bits_exact(n: int) -> np.ndarray:
        """Enumerate all computational-basis bitstrings in integer index order.

        The returned matrix has shape ``(2**n, n)`` and row ``i`` matches the
        bitstring produced by ``format(i, f"0{n}b")``.
        """
        # Create the integer labels 0, 1, ..., 2^n - 1 for the computational basis.
        basis_indices = np.arange(2**n, dtype=np.uint64)
        # Build bit positions from most-significant to least-significant so the
        # column order matches the repo's existing exact-enumeration convention.
        bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
        # Shift each basis index by each bit position and keep only the low bit.
        basis_bits = (basis_indices[:, None] >> bit_positions[None, :]) & 1
        # Store the result as uint8 because the entries are binary.
        return basis_bits.astype(np.uint8)

    @staticmethod
    def _fwht_inplace(values: np.ndarray) -> np.ndarray:
        """Apply the unnormalized Walsh-Hadamard transform in place.

        For a vector ``v`` of length ``2**n``, the transformed vector is

            ``w[x] = sum_z (-1)^(x·z) v[z]``.

        This is the transform needed to convert the diagonal-phase vector of an
        IQP circuit into computational-basis amplitudes after the final layer of
        Hadamards.
        """
        # Start with a butterfly width of 1, which combines adjacent pairs.
        h = 1
        # Double the butterfly width until it spans the whole vector.
        while h < len(values):
            # Walk over each block of size 2h that shares the same butterfly pattern.
            for start in range(0, len(values), 2 * h):
                # Split the block into its top and bottom halves.
                top = values[start : start + h].copy()
                bottom = values[start + h : start + 2 * h].copy()
                # Write the Hadamard sum branch into the top half.
                values[start : start + h] = top + bottom
                # Write the Hadamard difference branch into the bottom half.
                values[start + h : start + 2 * h] = top - bottom
            # Move to the next butterfly scale.
            h *= 2
        # Return the transformed vector for convenience.
        return values

    def probability_vector_exact(self, max_qubits: int = 20) -> np.ndarray:
        """Return the exact output probability vector for the current IQP model.

        What this method is doing:

        1. Enumerate every computational-basis bitstring ``z``.
        2. Evaluate the diagonal IQP phase

               ``phi(z) = sum_j theta_j (-1)^(z · g_j)``.

        3. Form the diagonal-phase vector ``d(z) = exp(-i phi(z))``.
        4. Apply the Walsh-Hadamard transform corresponding to the final layer
           of Hadamards in the IQP circuit.
        5. Square the amplitudes to obtain exact Born probabilities.

        Why we are implementing this:

        - the anti-concentration checker on the deterministic validation side
          needs the full probability vector for small ``n``
        - exact probabilities let us study output-distribution shape without
          introducing sampling, shot, or Monte Carlo ambiguity
        - this is the clean bridge from a trained IQP parameter vector to the
          anti-concentration metrics in ``iqp_bp.experiments.run_validation``

        Args:
            max_qubits:
                Safety cap for exact enumeration. Runtime and memory scale
                exponentially with ``n``, so this path is intentionally limited
                to the small-system deterministic regime.

        Returns:
            A one-dimensional array ``p`` of length ``2**n`` whose entries are
            the exact output probabilities in computational-basis index order.
        """
        # Reject large systems up front because exact probability extraction is exponential in n.
        if self.n > max_qubits:
            # Surface a clear error that keeps callers on the deterministic small-n path only.
            raise ValueError(
                f"Exact probability vector infeasible for n={self.n} > max_qubits={max_qubits}"
            )
        # Enumerate all computational-basis strings in the repo's exact bit ordering.
        basis_bits = self._basis_bits_exact(self.n)
        # Allocate one phase angle per basis string.
        phase_angles = np.zeros(len(basis_bits), dtype=np.float64)
        # Loop over generators because each one contributes additively to the diagonal phase.
        for theta_j, generator in zip(self.theta, self.G, strict=False):
            # Find which qubit positions participate in this generator.
            support = np.flatnonzero(generator)
            # If the generator is empty, it contributes the same global phase to every basis state.
            if support.size == 0:
                # Add that global phase contribution directly.
                phase_angles += theta_j
                # Move to the next generator.
                continue
            # Restrict all basis strings to the qubits touched by this generator.
            restricted_bits = basis_bits[:, support]
            # Compute the generator parity z · g_j mod 2 for every basis string.
            parities = np.sum(restricted_bits, axis=1) % 2
            # Convert parity to the diagonal IQP sign (-1)^(z · g_j).
            signs = 1.0 - 2.0 * parities.astype(np.float64)
            # Accumulate theta_j * (-1)^(z · g_j) into the diagonal phase.
            phase_angles += theta_j * signs
        # Convert the diagonal phases into the complex diagonal vector d(z).
        diagonal_phases = np.exp(-1j * phase_angles)
        # Copy the diagonal vector because the fast Walsh-Hadamard transform is in place.
        amplitudes = diagonal_phases.astype(np.complex128, copy=True)
        # Apply the unnormalized Walsh-Hadamard transform to obtain sum_z (-1)^(x·z) d(z).
        self._fwht_inplace(amplitudes)
        # Divide by 2^n because the IQP circuit has Hadamards before and after the diagonal block.
        amplitudes /= 2**self.n
        # Convert amplitudes into exact Born probabilities.
        probabilities = np.abs(amplitudes) ** 2
        # Remove tiny roundoff artifacts before normalization.
        probabilities = np.where(np.abs(probabilities) < 1e-15, 0.0, probabilities)
        # Renormalize to protect downstream diagnostics from accumulated floating-point drift.
        probabilities /= probabilities.sum()
        # Return the exact probability vector in computational-basis index order.
        return probabilities

    def output_probabilities_exact(self, max_qubits: int = 20) -> np.ndarray:
        """Alias for ``probability_vector_exact``.

        The planning docs alternate between "output probabilities" and
        "probability vector". This alias keeps both names available without
        duplicating the implementation.
        """
        # Forward to the canonical exact probability-vector implementation.
        return self.probability_vector_exact(max_qubits=max_qubits)
