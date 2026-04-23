# Testing Patterns

**Analysis Date:** 2026-04-22

## Test Framework

**Runner:**
- `pytest>=7.0` (declared in `pyproject.toml` under `[project.optional-dependencies].dev`)
- No `pytest.ini`, `setup.cfg`, or `[tool.pytest.ini_options]` section — pytest runs with defaults. Tests are discovered from the `tests/` directory and collected via the default `test_*.py` / `test_*` pattern.
- No `tests/conftest.py` — fixtures are defined inline per-module

**Property-based testing:**
- `hypothesis>=6.90` (dev extra)
- Custom strategies live in `src/iqp_bp/hypergraph/hypothesis_strategies.py` (exports `family_config`, `init_config`, `iqp_parameters`, `kernel_config`, `mmd_instance`, `smart_family_instance`)
- Driven from `tests/test_hypothesis.py`

**Assertion Library:**
- Plain `assert` plus `pytest.raises(...)` / `pytest.approx(...)`
- Numerical equality via `np.isclose` / `np.allclose` with explicit tolerances (e.g., `atol=0.02` for Monte-Carlo distribution checks)

**Run Commands:**
```bash
pytest                                  # Run all tests from repo root
pytest tests/test_mmd_kernel.py         # Single file
pytest tests/test_rng.py -k canonical   # Filter by keyword
pytest -x                                # Stop on first failure
```

## Test File Organization

**Location:**
- Separate top-level `tests/` directory (not co-located with source)
- One test module per source module, named `test_<module>.py` — e.g., `tests/test_mmd_kernel.py` ↔ `src/iqp_bp/mmd/kernel.py`, `tests/test_run_forge.py` ↔ `src/iqp_bp/experiments/run_forge.py`

**Naming:**
- Files: `test_<subject>.py`
- Test functions: `test_<subject>_<behavior>` — very descriptive, sentence-like (e.g., `test_gaussian_kernel_uses_locked_half_sigma_squared_normalization`, `test_run_training_smoke_emits_ac_and_marginal_fields`)

**Structure (~34 test files):**
```
tests/
  test_anti_concentration.py
  test_config.py
  test_expectation_small_n.py
  test_expectation_streaming.py
  test_experiment_emitter.py
  test_gradients_small_n.py
  test_hypergraph_families.py
  test_hypothesis.py            # property-based
  test_mmd_kernel.py
  test_mmd_exact_small_n.py
  test_mmd_loss_details.py
  test_rng.py
  test_run_forge.py
  test_run_training.py
  test_run_scaling.py
  test_run_qiskit.py
  test_run_validation.py
  test_trainer.py
  ...
  _tmp_config/                  # per-test scratch dirs (created & cleaned by tests)
  _tmp_run_training/
  _tmp_scaling/
  _tmp_validation/
  _tmp_qiskit/
  _tmp_iqp_mmd_checkpoint/
  _tmp_provenance/
  _pytest_tmp/
```

## Test Structure

**Suite Organization:**
- Primarily flat module-level `test_*` functions
- Classes used only for logical grouping of related assertions (not for shared setup/teardown) — see `TestCanonicalStreamNames`, `TestExperimentStreamBundle` in `tests/test_rng.py`:
  ```python
  class TestCanonicalStreamNames:
      def test_all_seven_streams_present(self):
          assert set(CANONICAL_STREAMS) == {
              "circuit", "data", "theta", "kernel", "estimation", "qiskit", "forge"
          }
  ```
- Section dividers via `# ---` rulers mirror the source convention

**Patterns:**
- **Setup:** inline builder helpers (e.g., `_minimal_valid_cfg(**overrides)` in `tests/test_config.py`, `_small_forge_cfg(tmp_path)` in `tests/test_run_forge.py`) that return dict configs; tests override individual fields per-case
- **Teardown:** largely unmanaged — tests write into `tests/_tmp_<suite>/<uuid4_hex>/` sub-directories (see `_workspace_tmp_dir()` pattern). This keeps artifacts inspectable after failure but requires periodic manual cleanup.
- **Assertion:** direct `assert` with human-readable messages for numeric/scientific checks, e.g. `assert np.isfinite(value), "MMD is not finite"`

## Fixtures and Factories

**`@pytest.fixture` (25 occurrences across 11 files):**
- Used sparingly and locally — no shared `conftest.py`
- Example (`tests/test_mmd_loss_details.py`):
  ```python
  @pytest.fixture
  def small_instance():
      rng = np.random.default_rng(42)
      n, m = 4, 6
      G = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
      theta = rng.uniform(-np.pi, np.pi, size=m)
      data = rng.integers(0, 2, size=(50, n), dtype=np.uint8)
      return {"G": G, "theta": theta, "data": data, "n": n, "m": m}
  ```

**Builder helpers (preferred over fixtures for configs):**
- Underscore-prefixed module-level functions: `_minimal_valid_cfg`, `_small_forge_cfg`, `_small_plateau_cfg`, `_write_checkpoint`, `_write_plateau_fixture`
- Accept `**overrides` or explicit args so each test can tweak one field

**Scratch directories:**
```python
def _workspace_tmp_dir() -> Path:
    path = Path("tests") / "_tmp_<suite>" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
```
Each test gets a fresh UUID-named directory under `tests/_tmp_<suite>/`. `tempfile.gettempdir()` is used for the Forge runner tests (`_make_local_tmp_dir` in `tests/test_run_forge.py`) because Racket's path handling prefers OS-temp.

**RNG seeding:**
- Always construct `np.random.default_rng(<fixed_int>)` inside the test — never rely on global NumPy state
- Seeds are small integers chosen for reproducibility, not domain-meaningful (`0`, `7`, `42` are the most common)

## Mocking

**Framework:** None beyond pytest built-ins

**Usage:**
- `monkeypatch` / `mocker` / `unittest.mock.patch` appear only in `tests/test_run_forge.py` (the Forge/Racket subprocess is the one external boundary mocked)
- Philosophy: **prefer real computation on small instances over mocking.** Tests exercise the actual `mmd2`, `iqp_phase`, `Trainer`, etc., at tiny `n` (e.g., 4–6 qubits) rather than stubbing.

**What to Mock:**
- External subprocesses (Racket/Forge binary) — see `tests/test_run_forge.py`
- Nothing else in this repo is mocked

**What NOT to Mock:**
- NumPy, JAX, the IQP model, MMD kernels — always call the real implementation on small-scale inputs
- Filesystem I/O — use real temp directories under `tests/_tmp_*/` instead of in-memory fakes

## `@pytest.mark.parametrize`

- Used in 11 files, covering enum domains, sweepable config axes, and algebraic identities
- Example domains parametrized: kernel types, init schemes, family names, small-`n` qubit counts, optimizer choices
- Prefer parametrize over duplicated test bodies when the only difference is an input value

## Coverage

**Requirements:** None enforced — no `pytest-cov` configuration, no coverage gate in `pyproject.toml`

**View Coverage (if needed, not wired up):**
```bash
pip install pytest-cov
pytest --cov=iqp_bp --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Scope: single function or class (`test_mmd_kernel.py`, `test_rng.py`, `test_config.py`, `test_marginals.py`)
- Majority of the suite

**Small-`n` Integration Tests:**
- Naming convention: `test_*_small_n.py` — exercise the exact (brute-force) code paths that are only feasible for small qubit counts
- Files: `test_expectation_small_n.py`, `test_gradients_small_n.py`, `test_iqp_probability_vector_small_n.py`, `test_mmd_exact_small_n.py`
- Used as ground-truth oracles to cross-check the Monte-Carlo sampling paths

**Runner / Smoke Tests:**
- `tests/test_run_forge.py`, `test_run_training.py`, `test_run_scaling.py`, `test_run_qiskit.py`, `test_run_validation.py`
- Load a real YAML config (or a builder-dict), invoke the experiment `run(cfg)` end-to-end, assert on emitted JSONL/NPZ artifacts

**Property-based Tests:**
- `tests/test_hypothesis.py` — uses strategies from `src/iqp_bp/hypergraph/hypothesis_strategies.py`
- Conventional settings: `@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])`
  (deadlines disabled because MMD/IQP computations are slow even at small `n`)

**E2E Tests:**
- Not used as a separate category; smoke runners fill this role

## Common Patterns

**Monte-Carlo estimator validation:**
```python
rng = np.random.default_rng(7)
samples = gaussian_sample_a(n=n, num_a_samples=20000, sigma=sigma, rng=rng)
empirical = np.bincount(samples.sum(axis=1), minlength=n + 1) / num_a_samples
expected = ...  # closed-form distribution
assert np.allclose(empirical, expected, atol=0.02)
```

**Config-builder + override pattern:**
```python
def _small_forge_cfg(tmp_path: Path) -> dict:
    return { ... baseline config ... }

def _small_plateau_cfg(tmp_path: Path, label_source: Path) -> dict:
    cfg = _small_forge_cfg(tmp_path)
    cfg["forge"]["mode"] = "plateau_agreement"
    return cfg
```

**Error Testing:**
```python
with pytest.raises(ValueError, match="...substring..."):
    validate_config(bad_cfg)
```

**Artifact assertions on experiment runners:**
```python
summaries = run(cfg)
assert len(summaries) == 1
trajectory_path = Path(summaries[0]["trajectory_path"])
assert trajectory_path.exists()
rows = [json.loads(line) for line in open(trajectory_path) if line.strip()]
assert all("ac_scaled_second_moment" in row for row in rows)
```

## Adding a New Test

1. Create `tests/test_<module>.py` mirroring the source path
2. Add `from __future__ import annotations` + import the symbol under test directly from `iqp_bp.*`
3. If the test writes artifacts, add a `_workspace_tmp_dir()` helper pointing at `tests/_tmp_<suite>/<uuid4_hex>/`
4. If the test needs a config, write a `_minimal_valid_cfg(**overrides)` or `_small_<suite>_cfg(tmp_path)` builder — do not hand-construct deeply nested dicts in each test
5. For Monte-Carlo checks, always pass an explicit `rng=np.random.default_rng(<int>)`; never rely on global state
6. Keep Hypothesis tests capped at `max_examples=20` with `deadline=None` to match existing conventions

---

*Testing analysis: 2026-04-22*
