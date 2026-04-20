"""Emit Forge instances for observed experiment rows.

A Forge "inst" block pins every atom and field in the universe — it is the
fully-concrete counterpart to a sig declaration in hypergraph.frg. For the
plateau-agreement mode we need to freeze not only the hypergraph structure
(handled by export_instances.py) but also the Experiment-side metadata:
which kernel, which init, what plateau label was observed. This module
appends a second inst block of that shape to the per-row `.frg` file.

The mapping dicts below translate the Python string names used everywhere
else in the project into the exact backtick-prefixed atom names defined as
sub-sigs in hypergraph.frg. If those atom names ever drift apart, the
generated inst block will fail to parse and Forge will complain loudly.
"""

from __future__ import annotations

from pathlib import Path

from iqp_bp.forge.export_instances import export_to_forge
from iqp_bp.forge.label_loader import LabeledRow


# Python kernel name (matches configs/experiments/*.yaml) -> Forge sub-sig.
# Forge concrete atoms are named by the sub-sig + a per-atom suffix (see
# hypergraph.frg: `one sig Gaussian, Laplacian, ... extends Kernel`). We
# append "0" below to name the single atom of each `one sig`.
KERNEL_ATOM_MAP: dict[str, str] = {
    "gaussian": "Gaussian",
    "laplacian": "Laplacian",
    "multi_scale_gaussian": "MultiScaleGaussian",
    "polynomial": "Polynomial",
    "linear": "Linear",
}

# Python init-scheme name (matches run_training.py and configs) -> Forge sub-sig.
# Same atom-suffix convention as KERNEL_ATOM_MAP.
INIT_ATOM_MAP: dict[str, str] = {
    "uniform": "UniformInit",
    "small_angle": "SmallAngleInit",
    "data_dependent": "DataDependentInit",
}


def _atom(name: str) -> str:
    """Wrap a Forge atom name in a backtick, as `inst` blocks require."""
    return f"`{name}"


def emit_experiment_instance(
    row: LabeledRow,
    output_path: str | Path,
    *,
    instance_name: str | None = None,
) -> None:
    """Emit the hypergraph inst plus an Experiment inst for one labeled row.

    Writes two blocks to ``output_path``:
        1. The hypergraph inst (via ``export_to_forge``). This pins every
           Qubit, Generator, containment relation, and overlap edge.
        2. An Experiment inst that names `Exp0` and binds its loss, kernel,
           init, and plateau_observed fields to the Forge atoms that match
           the Python row's string-valued metadata.

    The caller (run_forge.py) then concatenates this file with a bounds_inst
    and a query template to produce the final runnable `.frg`.

    Args:
        row: Labeled row carrying the hypergraph matrix G plus kernel/init/
            plateau metadata.
        output_path: Destination `.frg` file (will be created/overwritten by
            export_to_forge, then appended to below).
        instance_name: Optional override for the experiment inst's name;
            defaults to ``"experiment_<row.row_id>"``.
    """
    # Name the Experiment inst; this is the handle the bounds_inst + query
    # template refer to when pulling the Experiment atom into scope.
    experiment_inst_name = instance_name or f"experiment_{row.row_id}"
    # Resolve Python kernel/init strings to Forge sub-sig names. A KeyError
    # here means a new kernel or init scheme was added on the Python side
    # without a matching entry in the atom maps above.
    try:
        kernel_atom = KERNEL_ATOM_MAP[row.kernel]
    except KeyError as exc:
        raise ValueError(f"unknown kernel atom mapping for {row.kernel!r}") from exc
    try:
        init_atom = INIT_ATOM_MAP[row.init]
    except KeyError as exc:
        raise ValueError(f"unknown init atom mapping for {row.init!r}") from exc

    # Step 1: write the hypergraph inst block (overwrites output_path).
    output_file = Path(output_path)
    export_to_forge(row.G, row.n, output_file)

    # Step 2: build the Experiment inst body line-by-line. Each `X = A -> B`
    # assignment binds field X of atom A to atom B — this is Forge inst
    # syntax for "the value of A.X is B". We use the `one sig`-generated
    # atoms `Exp0`, `MMD0`, etc., suffix "0" because `one sig` produces
    # exactly one atom per sub-sig and Forge names it by appending "0".
    lines = [
        "",
        (
            # Human-readable reminder of what this block is for; surfaces in
            # diffs and solver traces.
            f"// Auto-generated experiment: kernel={row.kernel}, "
            f"init={row.init}, plateau={row.plateau_observed}"
        ),
        # Declare the inst with its name; everything indented below is the
        # body. Forge inst blocks use `{ }` as scope delimiters.
        f"inst {experiment_inst_name} {{",
        # Bind the Experiment sig to the single atom `Exp0`. Must appear
        # before any `Exp0 -> ...` assignment so Forge knows Exp0 exists.
        f"  Experiment = {_atom('Exp0')}",
        # Pin the four Experiment fields. `Exp0 -> MMD0` means the `loss`
        # relation maps the Exp0 atom to the MMD0 atom; that is exactly how
        # Forge represents `Exp0.loss = MMD` post-compilation.
        f"  loss = {_atom('Exp0')} -> {_atom('MMD0')}",
        f"  kernel = {_atom('Exp0')} -> {_atom(f'{kernel_atom}0')}",
        f"  init = {_atom('Exp0')} -> {_atom(f'{init_atom}0')}",
        # Ground-truth plateau label, mapped to its Outcome sub-sig atom.
        f"  plateau_observed = {_atom('Exp0')} -> {_atom(f'{row.plateau_observed}0')}",
        "}",
    ]
    # Append the Experiment inst after the hypergraph inst written above.
    # Opening in "a" mode preserves the hypergraph-inst content from step 1.
    with output_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
