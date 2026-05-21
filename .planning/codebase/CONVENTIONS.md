# Coding Conventions

**Analysis Date:** 2026-04-22

## Naming Patterns

**Files:**
- `snake_case.py` throughout `src/iqp_bp/` and `tests/`
- Test files prefixed with `test_` and mirror the module they cover (e.g., `tests/test_mmd_kernel.py` tests `src/iqp_bp/mmd/kernel.py`)
- Experiment runners prefixed with `run_` (e.g., `src/iqp_bp/experiments/run_forge.py`, `run_training.py`, `run_scaling.py`)

**Functions:**
- `snake_case` for all public and private functions
- Private/internal helpers prefixed with a single underscore (e.g., `_gaussian_tau`, `_as_list`, `_extract_family_kwargs` in `src/iqp_bp/experiments/run_forge.py`)
- Test helpers also underscore-prefixed (e.g., `_minimal_valid_cfg`, `_workspace_tmp_dir`, `_small_forge_cfg`)

**Variables:**
- `snake_case` for locals and module-level data
- Uppercase for module-level constants (e.g., `STREAM_CIRCUIT`, `CANONICAL_STREAMS`, `_REPO_ROOT`, `_VALID_FAMILIES` in `src/iqp_bp/rng.py` and `src/iqp_bp/config.py`)
- Single-letter conventional math names preserved where domain-standard: `G` (generator matrix / hypergraph), `n` (qubits), `m` (generators), `w` (Hamming weight), `a` (Z-word bitmask), `rng` (NumPy Generator)

**Types:**
- `PascalCase` for classes (e.g., `Trainer` in `src/iqp_bp/training/trainer.py`, `ForgeResult` in `src/iqp_bp/forge/runner.py`, `LabeledRow`, `IQPModel`)
- `PascalCase` type aliases (e.g., `CheckpointCallback` in `src/iqp_bp/training/trainer.py`)

## Code Style

**Formatting:**
- Ruff is the canonical formatter/linter (declared in `pyproject.toml` `[project.optional-dependencies].dev`)
- Line length: **120** (`[tool.ruff] line-length = 120` in `pyproject.toml`)
- Target Python: **3.10** (`target-version = "py310"`)

**Linting:**
- `ruff>=0.1` (dev extra); no custom rule list — defaults apply
- No `.ruff.toml`, no `.flake8`, no `black` config — Ruff is the single source of truth

## Import Organization

**Order (observed, ruff/isort default):**
1. `from __future__ import annotations` (first line after docstring in most modules)
2. Standard library (`json`, `hashlib`, `logging`, `pathlib`, `typing`, `uuid`, etc.)
3. Third-party (`numpy as np`, `pytest`, `yaml`, `hypothesis`, `jax`)
4. First-party absolute imports from `iqp_bp.*`

**Path Aliases:**
- None — always absolute imports from the `iqp_bp` package (e.g., `from iqp_bp.mmd.loss import mmd2`, `from iqp_bp.rng import STREAM_FORGE, experiment_stream_bundle`)
- No relative imports observed in `src/iqp_bp/`

## Error Handling

**Patterns (133 raises across 24 modules):**
- Raise specific exceptions — predominantly `ValueError` for config/input validation, `KeyError`, `RuntimeError`, `TypeError`, `NotImplementedError`
- Heavy use in `src/iqp_bp/config.py` (23 raises) and `src/iqp_bp/distributions/marginals.py` (15 raises) for up-front input validation
- Prefer "fail loud early" — validate config at entry points (see `validate_config` in `src/iqp_bp/config.py`) rather than defensive checks deep in the stack
- No bare `except:` and no generic `except Exception` swallowing observed
- Errors include descriptive messages with the offending value (f-strings)

## Logging

**Framework:** Python stdlib `logging`

**Patterns:**
- Module-level logger pattern: `log = logging.getLogger(__name__)` (lowercase `log`, not `logger`)
- Present in every experiment runner: `src/iqp_bp/experiments/run_forge.py`, `run_qiskit.py`, `run_scaling.py`, `run_training.py`, `run_validation.py`, and `src/iqp_bp/forge/label_loader.py`
- Use `log.warning(...)` / `log.info(...)` with `%s`-style formatting (lazy interpolation), e.g.:
  ```python
  log.warning(
      "run_forge: erdos_renyi.p_edge has %d values; using only the first (%s)",
      len(p_edge), p_edge[0],
  )
  ```
- Log level is configurable via `experiment.log_level` in YAML config (validated against `_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING"}`)

## Comments

**When to Comment:**
- Module docstrings describe *purpose* and list canonical names/streams the module exposes (see `src/iqp_bp/rng.py` docstring cataloguing `STREAM_*` constants)
- Section banners via `# ---` rulers separate logical blocks in long modules:
  ```python
  # ---------------------------------------------------------------------------
  # Canonical stream names
  # ---------------------------------------------------------------------------
  ```
- Inline comments explain *why*, especially around physics/math conventions (e.g., `gaussian_kernel` notes the +/-1 encoding)

**Docstrings:**
- Every public function has a one-line summary; complex ones include parameter and return notes in a loose NumPy-style
- Module docstrings cross-reference glossary entries, e.g. `spectral weight: docs/technical/glossary.md#spectral-weight` in `src/iqp_bp/mmd/kernel.py`

## Function Design

**Type Hints:**
- Pervasive modern type hints using PEP 604 union syntax (`int | None`, `str | Path`, `dict[str, Any]`)
- `from __future__ import annotations` is used selectively (not universal — only 5 modules in `src/iqp_bp/` use it, but most test files do)
- `Literal[...]` for finite string domains (e.g., `detect_status` returns `Literal["sat", "unsat", "unknown"]` in `src/iqp_bp/forge/parser.py`)
- `Callable[...]` type aliases at module scope for long signatures (e.g., `CheckpointCallback` in `src/iqp_bp/training/trainer.py`)

**Parameters:**
- Keyword-only arguments (`*,`) for optional/config-like parameters on public APIs — see `Trainer.__init__` in `src/iqp_bp/training/trainer.py` which puts `output_dir`, `kernel`, `kernel_params`, `optimizer`, etc. after `*`
- RNG always passed explicitly as `rng: np.random.Generator | None = None` with an internal `rng = np.random.default_rng()` default — never use a module-global RNG

**Return Values:**
- Scientific returns coerced to native Python `float` (not `np.float64`) at the boundary, e.g. `return float(np.tanh(...))` in `gaussian_tau` and `gaussian_kernel`
- Complex results returned as `dict[str, Any]` with documented keys (e.g., `mmd2` returns `{"mmd2", "a_samples", "exp_p", "exp_q", "contributions", "mc_diagnostics"}` when `return_details=True`)

## Module Design

**Exports:**
- Submodules expose their public API via `__init__.py` re-exports (see `src/iqp_bp/forge/__init__.py` re-exporting `build_query`, `detect_plateau_agreement`, `export_to_forge`, `load_labeled_rows`, `run_racket`, etc.)
- No explicit `__all__` lists observed — re-exports are the contract

**Barrel Files:**
- Used for subpackages with many entry points (`forge/__init__.py`). Individual `*.py` modules do not use barrels internally.

## Reproducibility / RNG Conventions

- All randomness flows through `src/iqp_bp/rng.py` using seven canonical stream names: `circuit`, `data`, `theta`, `kernel`, `estimation`, `qiskit`, `forge`
- Use `experiment_stream_bundle(seed, name, coord)` to derive per-stream seeds rather than ad-hoc `np.random.default_rng(seed + i)` arithmetic
- Each stream constant (`STREAM_CIRCUIT`, `STREAM_DATA`, etc.) must be imported by name; hard-coded string labels are discouraged

## Configuration

- All experiment configuration is YAML, loaded via `iqp_bp.config.load_config` (merges a user override onto `configs/base.yaml`)
- `validate_config` enforces enum domains declared as `_VALID_*` frozensets at module scope in `src/iqp_bp/config.py`
- Never read config keys with bare dict access in runner code — go through `load_config` / `resolve_experiment_grid` so validation and grid expansion happen in one place

---

*Convention analysis: 2026-04-22*
