"""Parameterized Forge query templates.

Every Forge-side experiment this repo runs boils down to appending a
`test expect { ... }` block to the main model library (hypergraph.frg) plus
a row-specific `inst` block. That `test expect` block is exactly the
assertion we want the Racket solver to check. The strings in
``QUERY_TEMPLATES`` below are those assertion bodies, with Python-style
``{placeholder}`` holes that ``build_query`` fills in.

Because the template strings are literal Forge code, Forge's own braces
``{ }`` must be doubled (``{{`` / ``}}``) so Python's ``str.format`` does not
interpret them as format specifiers.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Named query bodies
# ---------------------------------------------------------------------------
# Each entry below is one strategy for asking Forge a yes/no question. The
# key is a short name referenced from configs (``forge.query_template``);
# the value is the Forge source rendered into each `.frg` search file at
# run time. Every template keeps an ``overlaps_consistent`` conjunct so that
# whatever Forge synthesizes (or loads from an inst block) respects the
# invariant that the `overlaps` field mirrors real support intersections.
QUERY_TEMPLATES: dict[str, str] = {
    # -------------------------------------------------------------------
    # F2 template search. No labels required. Asks: "over all hypergraphs
    # with exactly m generators and n qubits, is it impossible to satisfy
    # bounded_degree[k] AND high_overlap[t] simultaneously?" `is unsat`
    # means the structural conjunction has NO model at that scope, which
    # is the signal F2 interprets as "this structural signature is
    # incompatible with barren-plateau-inducing overlaps". The `0 Experiment`
    # scope suppresses Experiment-atom allocation because this query is
    # purely structural (no init/kernel/plateau binding needed).
    # -------------------------------------------------------------------
    "plateau_inducing_bounded": (
        "test expect {{\n"
        "  {name}: {{\n"
        "    overlaps_consistent\n"
        "    bounded_degree[{max_weight}]\n"
        "    high_overlap[{overlap_threshold}]\n"
        "  }} for exactly {m} Generator, {n} Qubit, 8 Int, 0 Experiment is unsat\n"
        "}}\n"
    ),
    # -------------------------------------------------------------------
    # F3 plateau agreement (structure-only predictor). For a single labeled
    # row bound as `Exp0`, check that the F2 structural predicate agrees
    # (iff) with the empirical `Exp0.plateau_observed` value. `is sat` here
    # means "there is a way to bind the atoms such that the iff holds" —
    # and because every atom is pinned by the inst block, the only degree
    # of freedom is truth of the iff itself. SAT = predicate agrees; UNSAT
    # = predicate disagrees. This is the F3 predicate that got 0/88 recall.
    # -------------------------------------------------------------------
    "plateau_agreement": (
        "test expect {{\n"
        "  {name}: {{\n"
        "    overlaps_consistent\n"
        "    plateau_structurally_predicted[{max_weight}, {overlap_threshold}]\n"
        "      iff `Exp0.plateau_observed = PlateauObserved\n"
        "  }} for {bounds_inst} is sat\n"
        "}}\n"
    ),
    # -------------------------------------------------------------------
    # F4 joint (Init, n, structure) predicate. Same iff-shape as F3 but
    # calls the richer `plateau_manifests` predicate that branches on the
    # Experiment's init field and #Qubit before falling back to structure.
    # -------------------------------------------------------------------
    "plateau_manifests_joint": (
        "test expect {{\n"
        "  {name}: {{\n"
        "    overlaps_consistent\n"
        "    plateau_manifests[`Exp0]\n"
        "      iff `Exp0.plateau_observed = PlateauObserved\n"
        "  }} for {bounds_inst} is sat\n"
        "}}\n"
    ),
    # -------------------------------------------------------------------
    # F4 ablation: plateau_manifests WITHOUT the complete-graph escape
    # hatch. Running this + the full F4 on the same 111 rows measures
    # exactly how many predictions the structure clause changed.
    # -------------------------------------------------------------------
    "plateau_manifests_joint_no_escape": (
        "test expect {{\n"
        "  {name}: {{\n"
        "    overlaps_consistent\n"
        "    plateau_manifests_no_escape[`Exp0]\n"
        "      iff `Exp0.plateau_observed = PlateauObserved\n"
        "  }} for {bounds_inst} is sat\n"
        "}}\n"
    ),
}


def build_query(
    template: str,
    instance_name: str,
    *,
    thresholds: dict[str, int],
    bounds: dict[str, int],
    context: dict[str, int | str] | None = None,
) -> str:
    """Render a named query template by substituting Python values into Forge code.

    The returned string is pure Forge source; callers append it to a model
    library (hypergraph.frg) plus an `inst` block to produce a runnable
    `.frg` file.

    Args:
        template: Key into ``QUERY_TEMPLATES``.
        instance_name: Human-readable name for the test block (e.g.
            ``"row_product_state_n4"``); surfaces in Racket output.
        thresholds: Integer thresholds for structural predicates
            (``max_weight``, ``overlap_threshold``). Kept even when the
            template does not reference them — ``str.format`` tolerates
            unused keys.
        bounds: Scope numbers (``n``, ``m``) for the universe of Qubits
            and Generators.
        context: Extra render keys specific to certain templates — in
            particular plateau-mode templates need ``bounds_inst`` to
            name the inst block they scope over.

    Returns:
        Rendered Forge source ready to be concatenated into a `.frg` file.
    """
    # Guard against typos / deprecated template names coming from configs.
    if template not in QUERY_TEMPLATES:
        raise ValueError(f"unknown query template: {template!r}")

    # Base substitution map — the keys every template can reference.
    render_context: dict[str, int | str] = {
        "name": instance_name,
        "max_weight": int(thresholds["max_weight"]),
        "overlap_threshold": int(thresholds["overlap_threshold"]),
        "n": int(bounds["n"]),
        "m": int(bounds["m"]),
    }
    # Merge template-specific extras (e.g. ``bounds_inst``) on top.
    if context is not None:
        render_context.update(context)

    # `.format(**render_context)` replaces the `{placeholder}` holes;
    # unused keys are silently ignored by Python's format machinery.
    return QUERY_TEMPLATES[template].format(
        **render_context,
    )
