"""Export hypergraph instances to Forge (.frg) format.

Forge (Alloy-based relational logic tool) is used for finite model finding
and structural invariant checking. This module serializes Python hypergraph
instances into Forge `inst` blocks — the fully-pinned counterparts to the
`sig` declarations in hypergraph.frg. Whereas a sig declares a set of atoms
the solver is free to populate, an `inst` block pins every atom and every
field, effectively giving the solver zero structural freedom. That is what
we want for F3/F4: the hypergraph is a fact, not a variable.

Typical output block (for an n=4, m=4 example):

    inst hypergraph_4_4 {
      Qubit = `Q0 + `Q1 + `Q2 + `Q3
      Generator = `G0 + `G1 + `G2 + `G3
      contains = `G0->`Q0 + `G0->`Q1 + ...
      overlaps = `G0->`G1 + `G1->`G0 + ...
      Params = `Params0
      max_weight = `Params0 -> 3
      overlap_threshold = `Params0 -> 2
      max_qubit_degree = `Params0 -> 2
    }

Backticks mark literal atom names (Forge requires them inside inst blocks);
``+`` is relation/set union; ``X -> Y`` is a tuple in a binary relation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _relation_literal(pairs: list[str]) -> str | None:
    """Format a Forge relation literal, or return None for an empty relation.

    Forge 5.2's `inst` blocks reject `none->none` (NONE-TOK parse error) and
    do not accept `= none` for arity-2 fields. The idiom for an empty field
    is to OMIT the binding entirely; the caller is responsible for skipping
    the `field = ...` line when this returns None.
    """
    # Empty list = empty relation = field binding must be skipped upstream.
    if not pairs:
        return None
    # Forge treats ``A + B + C`` as the union of the singletons A, B, C.
    return " + ".join(pairs)


def _atom(name: str) -> str:
    """Forge inst bounds require explicit backtick atoms."""
    # e.g. _atom("Q3") -> "`Q3". Without the backtick, Forge treats the
    # identifier as a free variable rather than a named literal atom.
    return f"`{name}"


def _overlap_edges(G: np.ndarray) -> list[tuple[int, int]]:
    """Directed overlap edges where two distinct generators share a qubit.

    Returns one tuple ``(j, k)`` per ordered pair of distinct generators
    whose supports intersect. We emit both directions (j,k) and (k,j) so
    that `overlaps_consistent` in hypergraph.frg can compare against the
    symmetric reconstruction from `contains`.
    """
    # O[j, k] = |supp(g_j) ∩ supp(g_k)| by integer matrix multiplication.
    O = compute_overlap_matrix(G)
    m = O.shape[0]
    # Drop the diagonal (no self-loops) and keep only strictly-positive
    # intersections. Both directions included: (j,k) and (k,j).
    return [(j, k) for j in range(m) for k in range(m) if j != k and O[j, k] > 0]


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
    # Pull dimensions from the matrix; m is the number of generators (rows),
    # n is the number of qubits (columns; also passed explicitly so callers
    # using empty matrices still get the right Qubit scope).
    m = G.shape[0]
    output_path = Path(output_path)
    # Pre-compute max weight / overlap / qubit degree so we can pin them
    # inside the Params atom at the end of the inst block. Useful when a
    # downstream predicate wants to reference "the structural limits of
    # this particular hypergraph" without having to recompute them.
    stats = overlap_stats(G)

    # Accumulate output as a list of strings; join and write at the end.
    lines = []
    # Optional: splice a pre-existing model file at the top. Used by some
    # standalone harnesses that want a single self-contained `.frg`.
    if model_template is not None:
        with open(model_template) as f:
            lines.extend(f.read().splitlines())
        lines.append("")

    # Header comment + inst block open. Name convention `hypergraph_<n>_<m>`
    # so downstream callers (run_forge.py's plateau_agreement mode) can
    # reference the instance by a predictable handle.
    lines.append(f"// Auto-generated instance: n={n}, m={m}")
    lines.append(f"inst hypergraph_{n}_{m} {{")
    lines.append(f"  // {m} generators over {n} qubits")
    # Bind the Qubit sig to exactly n backticked atoms Q0..Q(n-1).
    lines.append("  Qubit = " + " + ".join(_atom(f"Q{i}") for i in range(n)))
    # Same for Generator atoms. `none` is Forge's empty-relation literal,
    # used when m == 0 so the inst block still parses.
    lines.append(
        "  Generator = "
        + (" + ".join(_atom(f"G{j}") for j in range(m)) if m else "none")
    )
    lines.append("")

    # Build the `contains` relation pair-by-pair: for every 1 in G[j, i],
    # emit the tuple `Gj -> Qi`. Forge requires explicit enumeration for
    # inst blocks — there is no sparse-matrix shorthand.
    contains_pairs: list[str] = []
    for j in range(m):
        support = np.where(G[j] == 1)[0]
        contains_pairs.extend(f"{_atom(f'G{j}')}->{_atom(f'Q{i}')}" for i in support)

    lines.append("  // Generator-qubit containment relation: contains[G_j][Q_i] iff G[j,i]=1")
    contains_literal = _relation_literal(contains_pairs)
    if contains_literal is not None:
        lines.append(f"  contains = {contains_literal}")
    else:
        # Empty-relation fallback: omit the binding entirely (see docstring
        # of _relation_literal for the Forge 5.2 parser limitation).
        lines.append("  // contains binding omitted -- no generator has any support")
    lines.append("")

    # Build the `overlaps` edge set the same way. Both directions included
    # to match `overlaps_consistent`'s symmetric check.
    overlap_pairs = [f"{_atom(f'G{j}')}->{_atom(f'G{k}')}" for j, k in _overlap_edges(G)]
    lines.append("  // Overlap graph: directed edges for every non-zero off-diagonal overlap")
    overlaps_literal = _relation_literal(overlap_pairs)
    if overlaps_literal is not None:
        lines.append(f"  overlaps = {overlaps_literal}")
    else:
        lines.append("  // overlaps binding omitted -- no two generators share any qubit")
    lines.append("")

    # Pin the `Params` one-sig atom and its three threshold fields. Using
    # `stats` means the pinned values reflect this hypergraph's actual
    # structural limits, not arbitrary picks.
    lines.append(f"  Params = {_atom('Params0')}")
    lines.append(f"  max_weight = {_atom('Params0')} -> {stats['max_weight']}")
    lines.append(f"  overlap_threshold = {_atom('Params0')} -> {stats['max_overlap']}")
    lines.append(f"  max_qubit_degree = {_atom('Params0')} -> {stats['max_qubit_degree']}")
    lines.append("}")

    # Write everything out. Truncates any prior content at `output_path`;
    # the Experiment inst (if any) is appended afterwards by experiment_emitter.
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def compute_overlap_matrix(G: np.ndarray) -> np.ndarray:
    """Compute pairwise generator overlap matrix.

    overlap[j, k] = |support(g_j) ∩ support(g_k)| = g_j · g_k

    Returns:
        overlap: shape (m, m), int
    """
    # Integer matrix multiply G @ G.T counts the number of qubits each pair
    # of rows jointly contains — exactly the overlap cardinality.
    return G.astype(int) @ G.astype(int).T


def overlap_stats(G: np.ndarray) -> dict:
    """Compute overlap statistics for a generator matrix.

    The returned dict feeds the `Params` inst block so Forge can reason
    about this specific hypergraph's structural limits without having to
    recompute them from the `contains` relation.
    """
    m = G.shape[0]
    O = compute_overlap_matrix(G)
    # Upper-triangle (k=1 excludes the diagonal of row-self-overlaps) gives
    # us just the unordered pair statistics.
    off_diag = O[np.triu_indices(m, k=1)]
    # Per-generator Hamming weights (row sums of G).
    weights = G.sum(axis=1)
    # Each return field is named for its role inside Forge's Params atom:
    return {
        "max_overlap": int(off_diag.max()) if len(off_diag) > 0 else 0,
        "mean_overlap": float(off_diag.mean()) if len(off_diag) > 0 else 0.0,
        "mean_weight": float(weights.mean()) if len(weights) > 0 else 0.0,
        "max_weight": int(weights.max()) if len(weights) > 0 else 0,
        # Column sums = per-qubit appearances across generators.
        "max_qubit_degree": int(G.sum(axis=0).max()) if G.size else 0,
        "fraction_zero_overlap": float((off_diag == 0).mean()) if len(off_diag) > 0 else 1.0,
    }
