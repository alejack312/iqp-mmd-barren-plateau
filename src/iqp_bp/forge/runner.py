"""Subprocess wrapper for running Forge through Racket.

Forge ships as a Racket language extension — there is no standalone
``forge`` CLI. To run a ``.frg`` file we shell out to ``racket <file>``,
which reads the ``#lang forge`` pragma at the top of the file and evaluates
the model plus any ``test expect`` blocks. This module is the thinnest
possible wrapper around that subprocess call.

Design choices:
    * Every call returns a ``ForgeResult`` with a coarse ``status`` tag.
      Later parsing (parser.py) refines the status by grepping stdout;
      here we only report subprocess-level states (timeout, no racket,
      unknown).
    * No racket on PATH is treated as a first-class outcome (not an
      error). This lets CI and the live smoke tests skip Forge-dependent
      assertions gracefully on machines without a Forge install.
    * ``timeout_sec`` is mandatory. Long-running Forge queries are the
      dominant cost in F3/F4; a per-query timeout lets the outer driver
      decide whether to retry, skip, or abort.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# All the statuses a caller can observe. ``unknown`` is the default for a
# subprocess that completed without our wrapper classifying it — parser.py
# will refine it to sat/unsat/agree/disagree based on stdout content.
Status = Literal["sat", "unsat", "timeout", "error", "skipped_no_racket", "unknown"]


@dataclass(frozen=True)
class ForgeResult:
    """Everything the caller needs to reason about a single Racket invocation."""

    # Coarse outcome; see Status type above.
    status: Status
    # Full captured stdout — handed to parser.py for fine-grained classification.
    stdout: str
    # Captured stderr — Forge sometimes reports failures here.
    stderr: str
    # Wall-clock seconds the subprocess took, including Racket startup cost.
    elapsed_sec: float
    # Process exit code, or None if we never got one (timeout / missing binary).
    returncode: int | None


def run_racket(frg_path: Path, *, timeout_sec: int) -> ForgeResult:
    """Execute a Forge file with the local Racket binary when available.

    Args:
        frg_path: Path to a `.frg` file whose first line is ``#lang forge``.
        timeout_sec: Hard wall-clock budget; a longer-running query aborts
            with ``status == "timeout"`` and partial stdout/stderr captured.

    Returns:
        A ``ForgeResult`` regardless of success or failure — callers are
        expected to branch on ``status`` rather than on exceptions.
    """
    # Look up racket in PATH. If it is not installed we return a clearly
    # marked "skipped" result so the outer driver can emit a warning row
    # without bailing out the entire sweep.
    racket = shutil.which("racket")
    if racket is None:
        return ForgeResult("skipped_no_racket", "", "racket not on PATH", 0.0, None)

    # Record start time before the subprocess call so we can measure its
    # wall-clock duration even in the timeout-expired branch.
    start = time.perf_counter()
    try:
        # Capture both streams as text; never raise on non-zero exit code
        # because Forge exits non-zero when any `test expect` assertion
        # fails, and we want that stdout for parser.py to interpret.
        cp = subprocess.run(
            [racket, str(frg_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # Preserve any partial output the subprocess managed to emit before
        # it was killed — often useful for debugging hangs in large scopes.
        return ForgeResult(
            "timeout",
            exc.stdout or "",
            exc.stderr or "",
            time.perf_counter() - start,
            None,
        )

    # Subprocess finished in time. Elapsed seconds measured end-to-end so
    # callers can build latency histograms for the sweep.
    elapsed = time.perf_counter() - start
    # Return "unknown" — parser.py looks at stdout content to decide
    # whether this was a sat, unsat, agree, or disagree run.
    return ForgeResult("unknown", cp.stdout, cp.stderr, elapsed, cp.returncode)
