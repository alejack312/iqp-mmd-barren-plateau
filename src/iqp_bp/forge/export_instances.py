"""Export hypergraph instances to Forge (.frg) format.

Forge (Alloy-based relational logic tool) is used for finite model finding
and structural invariant checking. This module serializes Python hypergraph
instances into Forge fact blocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_to_forge(
    G: np.ndarray,
    n: int,
    output_path: str | Path,
    model_template: str | Path | None = None,
) -> None:
    """Export generator matrix G as Forge instance facts.

    Appends instance facts to the Forge model template (or creates standalone file).

    Args:
        G: Generator matrix, shape (m, n), uint8
        n: Number of qubits
        output_path: Output .frg file path
        model_template: Optional path to base model template to prepend
    """
    m = G.shape[0]
    output_path = Path(output_path)

    # TODO: Week 7 (D9.1) emit overlap-graph, degree-constraint, and threshold facts
    # so Forge can reason directly about the plateau-inducing structures from the spec.
    # Read first: Forge docs https://forge-fm.github.io/forge-documentation/ ;
    # Forge constraints https://forge-fm.github.io/forge-documentation/building-models/constraints/constraints.html ;
    # model intuition https://forge-fm.github.io/book/chapters/solvers/bounds_booleans_how_forge_works.html
    lines = []
    if model_template is not None:
        with open(model_template) as f:
            lines.extend(f.read().splitlines())
        lines.append("")

    lines.append(f"// Auto-generated instance: n={n}, m={m}")
    lines.append(f"inst hypergraph_{n}_{m} {{")
    lines.append(f"  // {m} generators over {n} qubits")
    lines.append(f"  Qubit = " + " + ".join(f"Q{i}" for i in range(n)))
    lines.append(f"  Generator = " + " + ".join(f"G{j}" for j in range(m)))
    lines.append("")
    lines.append("  // Generator-qubit containment relation: contains[G_j][Q_i] iff G[j,i]=1")
    lines.append("  contains = {")
    for j in range(m):
        support = np.where(G[j] == 1)[0]
        if len(support) > 0:
            pairs = " + ".join(f"G{j}->Q{i}" for i in support)
            lines.append(f"    {pairs}")
    lines.append("  }")
    lines.append("}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def compute_overlap_matrix(G: np.ndarray) -> np.ndarray:
    """Compute pairwise generator overlap matrix.

    overlap[j, k] = |support(g_j) ∩ support(g_k)| = g_j · g_k

    Returns:
        overlap: shape (m, m), int
    """
    return G.astype(int) @ G.astype(int).T


def overlap_stats(G: np.ndarray) -> dict:
    """Compute overlap statistics for a generator matrix."""
    m = G.shape[0]
    O = compute_overlap_matrix(G)
    off_diag = O[np.triu_indices(m, k=1)]
    weights = G.sum(axis=1)
    return {
        "max_overlap": int(off_diag.max()) if len(off_diag) > 0 else 0,
        "mean_overlap": float(off_diag.mean()) if len(off_diag) > 0 else 0.0,
        "mean_weight": float(weights.mean()),
        "max_weight": int(weights.max()),
        "fraction_zero_overlap": float((off_diag == 0).mean()) if len(off_diag) > 0 else 1.0,
    }
