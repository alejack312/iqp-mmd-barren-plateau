"""Best-effort parsing of Forge stdout.

Racket/Forge's ``test expect { ... }`` harness prints human-readable lines
to stdout/stderr; it does not expose a structured JSON or exit-code-based
API. To turn those lines into typed Python results we look for a small set
of marker substrings (case-insensitive) documented below. If we ever see a
new Forge version whose output phrasing changes, the markers here are the
only thing that needs updating.

Three kinds of question the parser answers:
    * ``detect_status`` — did an F2-style "is unsat" test find a SAT witness
      (i.e., the structural signature IS achievable at this scope)?
    * ``detect_plateau_agreement`` — did an F3/F4-style iff test pass,
      meaning the predicate agrees with the observed plateau label?
    * ``extract_witness_block`` — if we found a SAT witness, pull the
      struct-printed body back out so we can diff against expectations.
"""

from __future__ import annotations

import re
from typing import Literal


def detect_status(stdout: str) -> Literal["sat", "unsat", "unknown"]:
    """Infer Forge SAT/UNSAT status from the inverted test-expect query output.

    F2 templates phrase their assertion as ``... is unsat``; Forge's failure
    messages for that assertion contain phrases like "Expected unsat, got sat"
    when a counter-witness *does* exist. That counter-witness is exactly
    what F2 wants (structural signature is realizable), so we interpret it
    as ``"sat"``.
    """
    # Normalize to lowercase so marker comparisons ignore Forge's casing.
    text = stdout.lower()
    # Any of these substrings indicates the solver found a structural
    # witness — i.e. the assertion "is unsat" was refuted.
    if (
        "#(struct:sat" in text
        or "expected unsat, got sat" in text
        or "found instance #(struct:sat" in text
    ):
        return "sat"
    # These markers mean the solver confirmed no witness exists (for F2)
    # or that the test assertion held exactly ("test passed:").
    if (
        "#(struct:unsat" in text
        or "expected sat, got unsat" in text
        or "failed test" in text
        or "test passed:" in text
    ):
        return "unsat"
    # Fall-through: Racket crashed, timed out, or output a format we have
    # not taught this parser about. Caller decides how to handle `unknown`.
    return "unknown"


def detect_plateau_agreement(
    stdout: str,
    stderr: str = "",
) -> Literal["agree", "disagree", "unknown"]:
    """Infer plateau-agreement semantics from live Forge test output.

    Plateau-mode templates use ``... is sat``: the assertion claims the
    iff holds. If the solver prints "Test passed:" the iff is satisfied
    and predicate and label agree; a "Failed test" or
    "Expected sat, got unsat" means the iff was violated — predicate and
    label disagreed. Both stdout and stderr get concatenated because
    Forge occasionally emits its failure lines on stderr depending on
    flags and version.
    """
    combined = (stdout + "\n" + stderr).lower()
    # Solver confirmed the iff holds on the labeled row.
    if "test passed:" in combined:
        return "agree"
    # Solver found the iff was violated — predicate and label disagreed.
    if "failed test" in combined or "expected sat, got unsat" in combined:
        return "disagree"
    # Unrecognized output (timeout, error, unfamiliar Forge version).
    return "unknown"


def extract_witness_block(stdout: str) -> str | None:
    """Return the first Forge witness-like block if one is present.

    Two shapes show up in practice: (1) Forge's struct-printed ``#(struct:Sat
    ...)`` witness as the very last thing on stdout, and (2) an inline
    ``inst <name> { ... }`` block when the query was phrased to render the
    witness as an inst declaration. Tries the struct form first, falls back
    to the inst form.
    """
    # Shape 1: Racket struct print at end of stdout. Non-greedy inner match
    # and anchoring to end-of-string so we don't accidentally grab later
    # noise.
    sat_match = re.search(r"(#\(struct:Sat[\s\S]*?\)\s*)$", stdout.strip())
    if sat_match is not None:
        return sat_match.group(1).strip()
    # Shape 2: inline inst declaration. Broad, tolerant regex that tolerates
    # nested braces and multi-line content.
    inst_match = re.search(r"(inst\s+\w+\s*\{[\s\S]*\})", stdout)
    if inst_match is not None:
        return inst_match.group(1).strip()
    # No witness found. Caller should fall back to raw stdout.
    return None
