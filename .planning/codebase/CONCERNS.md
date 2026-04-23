# Codebase Concerns

**Analysis Date:** 2026-04-22

## Tech Debt

**Two parallel top-level packages:**
- Issue: The repo ships two distinct Python packages, `src/iqp_bp/` (the active barren-plateau research pipeline) and `src/iqp_mmd/` (a dormant sklearn-style scaffold based on XanaduAI/scaling-gqml). `pyproject.toml` declares both as wheel targets and four CLI entry points (`iqp-train`, `iqp-evaluate`, `iqp-sample`, `iqp-dataset`) resolve into `iqp_mmd`, while only `iqp-bp` points at the code that is actually maintained. The README documents the `iqp_mmd` API; the TODO roadmap, tests, and ongoing work live in `iqp_bp`. New contributors will not know which package is canonical.
- Files: `pyproject.toml:34-42`, `src/iqp_mmd/`, `src/iqp_bp/`, `README.md:9-24`, `tests/test_imports.py`, `tests/test_iqp_mmd_checkpoint_export.py`
- Impact: Doubles the surface area to maintain, confuses onboarding, and risks silent drift between the two kernels/model implementations. The stub `predict`/`predict_proba` methods in `src/iqp_mmd/models/iqp_simulator.py:165-169` are still shipped as part of a sklearn estimator.
- Fix approach: Decide whether `iqp_mmd` is deprecated, vendored-for-comparison, or a real product. If deprecated, move it under `legacy/` or remove it and update `pyproject.toml` and `README.md` to describe the `iqp_bp` CLI only.

**Open TODOs still flagged in source (cross-referenced by `TODOS.md`):**
- Issue: Six explicit `TODO:` markers remain in `src/iqp_bp` with SMART deliverable IDs.
- Files:
  - `src/iqp_bp/mmd/kernel.py:98` - Laplacian kernel kept as explicit stub; decomposition not derived.
  - `src/iqp_bp/mmd/kernel.py:199` - multi-scale Gaussian (phase-2 kernel) not yet exact-validated.
  - `src/iqp_bp/mmd/gradients.py:54` - JAX autodiff estimator promised for D2.1 not implemented.
  - `src/iqp_bp/mmd/gradients.py:243` - gradient-variance summary missing aggregate norm proxies and heavy-tail diagnostics.
  - `src/iqp_bp/iqp/expectation.py:146` - exact IQP expectation path not wired into automated MC-vs-exact validation harness.
  - `src/iqp_bp/hypergraph/families.py:269` - four-family primary sweep policy not centrally enforced.
  - `src/iqp_bp/hypergraph/hypothesis_strategies.py:69,117` - Hypothesis strategies for the four SMART families and data-dependent init are not yet covered.
  - `src/iqp_bp/mmd/mixture.py:40` - structured target-data helpers and cached parity stats missing.
- Impact: These are load-bearing gaps for the May 15 success criterion in `ScopeLock.md`. Several downstream TODOs in `TODOS.md` depend on them (e.g. V3 depends on gradients.py:54, V5 on hypothesis_strategies.py, K1 on kernel.py:199).
- Fix approach: Track each open `TODO:` to the corresponding ID in `TODOS.md` and close them inline. The dependency graph is already mapped in `TODOS.md` sections `Depends On Other TODOs`.

**Orphan/legacy package mirroring code in the active package:**
- Issue: `src/iqp_mmd/metrics/mmd_eval.py`, `src/iqp_mmd/observables/hamiltonian.py`, and `src/iqp_mmd/datasets/ising.py` re-implement concepts that `src/iqp_bp/mmd/`, `src/iqp_bp/experiments/data_factory.py`, and `src/iqp_bp/distributions/` already provide. Any future change (e.g. kernel convention fix) has to be reconciled in two places.
- Files: `src/iqp_mmd/` (entire tree), `src/iqp_bp/mmd/`, `src/iqp_bp/experiments/data_factory.py`
- Impact: Risk of using the wrong kernel convention in a paper figure or regression test.
- Fix approach: Either delete `iqp_mmd` (preferred given the TODO focus) or mark it read-only and bolt its conventions to the locked derivation in `docs/technical/glossary.md`.

**Mutable CLI test output directories committed:**
- Issue: The `tests/` directory contains committed scratch directories named `_pytest_tmp/`, `_tmp_config/`, `_tmp_iqp_mmd_checkpoint/`, `_tmp_provenance/`, `_tmp_qiskit/`, `_tmp_run_training/`, `_tmp_scaling/`, `_tmp_validation/`. These look like test-generated state that leaked into version control.
- Files: `tests/_pytest_tmp/`, `tests/_tmp_*`
- Impact: Noise in diffs, confusion about what is fixture vs generated output; tests that write there will mutate the repo during local runs.
- Fix approach: Audit what is inside each; if fixture, move under a `tests/fixtures/` path and document; otherwise delete and gitignore the patterns.

## Known Bugs

**`.bashrc` initialization error on Windows:**
- Symptoms: Every bash command in the repo emits `bash: line 1: export: '/c/Users/cuqui/.bashrc': not a valid identifier`.
- Files: user shell profile (not in repo), but reproducible on every invocation via any `Bash` tool call.
- Trigger: Running any shell command from the repo root on Windows with the current Git Bash setup.
- Workaround: None in-repo; documented as non-blocking noise. Could add a `scripts/` README note warning contributors.

**Fragile Forge output parsing:**
- Symptoms: `parse_witness(...)` returns `None` when the Racket `#(struct:Sat ...)` shape or inline `inst { ... }` shape is not matched, and the caller is expected to fall back to raw stdout. Regex uses `$` anchoring and nested-brace tolerance that will silently miss any Forge output layout change.
- Files: `src/iqp_bp/forge/parser.py:120-132`, `src/iqp_bp/forge/runner.py`
- Trigger: Forge v5.2 upgrade or non-default output mode.
- Workaround: Callers should not assume parsed witnesses are complete; always keep raw stdout in artifacts.

## Security Considerations

**Forge subprocess execution:**
- Risk: `src/iqp_bp/forge/runner.py:80` invokes `subprocess.run` on Racket-compiled `.frg` files. If a config or experiment template ever forwards user-supplied strings into command arguments or file paths, this becomes a command-injection vector.
- Files: `src/iqp_bp/forge/runner.py`, `src/iqp_bp/forge/experiment_emitter.py`, `src/iqp_bp/experiments/run_forge.py`
- Current mitigation: `shell=True` is not used; command argv is constructed from known config keys.
- Recommendations: Validate file paths resolve inside the expected `results/` or `forge/` subtree before handing them to subprocess; keep `shell=False` invariants documented.

**Broad `except Exception` clauses swallow failures:**
- Risk: Multiple diagnostic paths use `except Exception as exc:  # noqa: BLE001` and log only at warning level, then return `None`. A silent swallow can cause trainer rows to be emitted without AC diagnostics and never alert the caller.
- Files: `src/iqp_bp/experiments/run_training.py:300-319`, `src/iqp_bp/experiments/run_validation.py:319-324`, `src/iqp_bp/forge/parser.py:132`, `src/iqp_bp/experiments/run_qiskit.py:199`
- Current mitigation: These sites log at WARNING and comment the rationale (`best-effort diagnostic`).
- Recommendations: At minimum, persist the exception class/message into the JSONL row alongside the `None` so downstream analysis can see failure modes without re-running.

## Performance Bottlenecks

**Large runner files mix concerns:**
- Problem: The three biggest `iqp_bp` files are runners (`run_validation.py` 638 lines, `run_qiskit.py` 618 lines, `run_scaling.py` 483 lines, `run_training.py` 482 lines). Long monolithic runners make it hard to profile a single stage (dataset draw vs. MMD eval vs. AC summary) without reading the whole file.
- Files: `src/iqp_bp/experiments/run_validation.py`, `src/iqp_bp/experiments/run_qiskit.py`, `src/iqp_bp/experiments/run_scaling.py`, `src/iqp_bp/experiments/run_training.py`
- Cause: Runners own config parsing, stream setup, model construction, artifact writing, summary computation, and CSV/plot emission in one module.
- Improvement path: Extract stage-level helpers (dataset build, per-setting loop body, summary/plot emitter) into sibling modules under `src/iqp_bp/experiments/` so each can be unit-tested and timed independently. `marginal_artifacts.py` already demonstrates this pattern.

**No batching guarantee for exact small-n enumeration:**
- Problem: `IQPModel.probability_vector_exact` uses an in-place Walsh-Hadamard transform over the full `2**n` vector. Memory is acceptable for the TODOs' `n <= anti_concentration.max_n` path, but nothing in the runner enforces a hard cap at the entry point.
- Files: `src/iqp_bp/iqp/model.py`, `src/iqp_bp/experiments/run_scaling.py` (AC summary branch)
- Cause: `n` ceiling is schema-configurable, not asserted at the function boundary.
- Improvement path: Add an explicit `assert n <= 24` (or config-driven) guard inside `probability_vector_exact`, and fail fast with a clear message rather than OOM.

## Fragile Areas

**Config schema coverage vs. resolved-grid expansion:**
- Files: `src/iqp_bp/config.py` (463 lines), `configs/schema.yaml`, `src/iqp_bp/experiments/run_scaling.py`
- Why fragile: Multiple overlapping responsibilities (merge, validate, persist manifest, resolve Cartesian grid). Nested keys like `dataset.ising.*`, `dataset.binary_mixture.*`, and per-kernel/per-init sub-axes are schema-checked, but new dataset types or new kernel subaxes must be added to both the factory and the schema to avoid silent drop. TODO `P6` (grid-preview CLI) is still partial.
- Safe modification: Always extend `configs/schema.yaml` first, add a failing config test to `tests/test_config.py`, then wire the factory branch.
- Test coverage: Good for existing enum-valid paths; weaker for invalid-shape rejection in rarely-used axes (`multi_scale_gaussian`, data-dependent init config).

**Hypothesis cache drift:**
- Files: `.hypothesis/constants/` (2,516 generated files present on disk, 786 already tracked by `git ls-files`), `.gitignore`
- Why fragile: `.gitignore` does not mention `.hypothesis/`, so Hypothesis's autogenerated property-based test database is being committed. Every test run on a new machine produces diffs; PRs accumulate thousands of untracked `??` entries (visible in current `git status`).
- Safe modification: Add `.hypothesis/` to `.gitignore` and purge the currently-tracked files with `git rm -r --cached .hypothesis`. Coordinate with collaborators because they will see the deletion as a large diff.
- Test coverage: N/A.

**Pytest cache scratch in repo root:**
- Files: `pytest-cache-files-sez4iock/`, `pytest-cache-files-u4k5ertx/`, `bash.exe.stackdump`
- Why fragile: Random-suffix directories at repo root are not gitignored. `bash.exe.stackdump` is also committed as an artifact.
- Safe modification: Add `pytest-cache-files-*/` and `*.stackdump` to `.gitignore`; purge current copies.

**RNG stream contract depends on discipline:**
- Files: `src/iqp_bp/rng.py`, callers in `run_scaling.py`, `mmd/loss.py`, `mmd/gradients.py`, `qiskit/noise.py`
- Why fragile: Reproducibility (per `P2` in `TODOS.md`) relies on every caller using `named_seed_streams` with the canonical label list (`circuit`, `data`, `theta`, `kernel`, `estimation`, `qiskit`). There is no lint or runtime check that prevents a future caller from passing a bespoke seed label and breaking rerun determinism.
- Safe modification: Add a stream-name whitelist inside `rng.py` that rejects unknown names; extend `tests/test_rng.py` with a negative test.
- Test coverage: Strong for the happy path; no negative tests for unknown labels.

## Scaling Limits

**Exact small-n AC checker:**
- Current capacity: Controlled by `anti_concentration.max_n` in configs; default is small enough that `2**n` probability vector enumeration fits in memory.
- Limit: Roughly `n <= 24` for a comfortable classical machine; larger falls back to histogram mode.
- Scaling path: Histogram (sampled) mode already exists in `run_validation.py`; ensure `AC11` (long experiment results) truly uses it for `n > 24`.

**Cartesian experiment grid:**
- Current capacity: `resolve_scaling_settings` expands the full product over family * kernel * init * n * bandwidth * ER-degree * small-angle-std.
- Limit: Grid size grows multiplicatively; with the SMART plan (6 families * 4 kernels * 3 inits * 6 n-values * >=100 seeds) a single CPU run becomes impractical.
- Scaling path: Emit the resolved grid manifest (already implemented via `P1`), then shard by coordinate across workers.

## Dependencies at Risk

**Dual quantum backends (PennyLane + Qiskit/Aer):**
- Risk: `iqp_mmd` wires PennyLane (`pennylane>=0.33`, `iqpopt>=2024.7.0`) while `iqp_bp` wires Qiskit (`qiskit>=1.0`, `qiskit-aer>=0.14` under the optional `qiskit` extra). Keeping both healthy is a burden for one student.
- Impact: Version drift or API break in either breaks half the repo.
- Migration plan: Decide which backend is authoritative (Qiskit, based on active work in `src/iqp_bp/qiskit/`). Drop the PennyLane path if it is not presentation-critical.

**Forge tooling is external and unpinned at runtime:**
- Risk: Forge v5.2 is installed as a `raco pkg` outside the repo (per `reference_forge_install.md`). Racket/Java version mismatches on a collaborator's machine will silently produce different Forge outputs, which then feed `src/iqp_bp/forge/parser.py` whose regexes are tuned to v5.2.
- Impact: Cross-machine results drift without a clear failure signal.
- Migration plan: Print and log the Racket and Forge versions at the top of every Forge run (into the JSONL summary sidecars).

**Heavy ML stack on a research project:**
- Risk: `pyproject.toml` pulls JAX, Flax, PennyLane, numpyro, iqpopt in the base dependencies, but not all modules use them; JAX autodiff is still TODO (`src/iqp_bp/mmd/gradients.py:54`).
- Impact: Longer `pip install`, CI complexity, and install-time friction on Windows without WSL.
- Migration plan: Move `jax/jaxlib/flax/numpyro/pennylane/iqpopt` to an optional extra until `iqp_bp` actually imports them.

## Missing Critical Features

**JAX autodiff gradient estimator (`V3`):**
- Problem: The analytic and finite-difference gradient paths exist, but the JAX autodiff path promised by the spec for cross-check is not implemented.
- Files: `src/iqp_bp/mmd/gradients.py:54`
- Blocks: The three-way analytic/finite-diff/autodiff agreement check planned for D2.1; gradient-variance heavy-tail diagnostics (`S7`).

**Polynomial-vs-exponential fit and interim memo artifacts (`S8`):**
- Problem: Raw scaling JSONL is produced, but no scipy `curve_fit` summary or memo figures are generated.
- Files: `src/iqp_bp/experiments/run_scaling.py:86`
- Blocks: The May 15 success criterion in `ScopeLock.md` section 9.

**Grid-preview / validate-config CLI subcommand (`P6`):**
- Problem: A dedicated subcommand that prints the resolved scalar grid before committing to a long run is still partial.
- Files: `src/iqp_bp/cli.py`
- Blocks: Fast iteration on new experiment configs (`AC11`, `AC12`).

**Laplacian MMD^2 decomposition (`T2`):**
- Problem: The Laplacian kernel is approximated but not theory-locked; `w_L(a; σ)` in `ScopeLock.md:69` is flagged as `[approx; exact via Walsh-Hadamard transform]`.
- Files: `src/iqp_bp/mmd/kernel.py:92-98`
- Blocks: Any BP conclusion for Laplacian rows in the 6x4 grid.

## Test Coverage Gaps

**Stub sklearn methods in `iqp_mmd`:**
- What's not tested: `predict`/`predict_proba` bodies are empty `pass` statements; no test asserts that callers never invoke them. This is a latent `AttributeError`/`None` bug.
- Files: `src/iqp_mmd/models/iqp_simulator.py:165-169`
- Risk: If a notebook or downstream script calls `.predict(X)` on the simulator, it gets `None` silently.
- Priority: Medium (only if `iqp_mmd` stays in the repo).

**Hypothesis strategies missing for two of four SMART families:**
- What's not tested: Data-dependent initializer (`I3`) and the full small-angle sweep across `{0.01, 0.1, 0.3}` lack property-based coverage.
- Files: `src/iqp_bp/hypergraph/hypothesis_strategies.py:69,117`
- Risk: Structural regressions in `families.py` won't surface in CI until an experiment runs.
- Priority: High (TODO `V5`).

**MC-vs-exact IQP expectation plot harness is absent:**
- What's not tested: The exact probability-vector path is unit-tested (`test_iqp_probability_vector_small_n.py`) but no automated harness compares sampled `iqp_expectation` against the exact path across a grid of `n` and produces the validation plot called for in `V2`.
- Files: `src/iqp_bp/iqp/expectation.py:146`
- Risk: Silent drift between estimator and ground truth at moderate `n` values.
- Priority: High.

**Exception messages not captured in JSONL artifacts:**
- What's not tested: Failure paths in AC evaluators return `None` and log a warning; no test verifies that the resulting JSONL row distinguishes "AC not applicable" from "AC failed to compute".
- Files: `src/iqp_bp/experiments/run_training.py:300-319`, `src/iqp_bp/experiments/run_validation.py`
- Risk: Downstream plots conflate missing-by-design with missing-by-bug.
- Priority: Medium.

**Forge output regex fragility:**
- What's not tested: `parse_witness` has two regex shapes but no test covers malformed Racket output, truncation, or empty `stdout`.
- Files: `src/iqp_bp/forge/parser.py:120-132`, `tests/test_run_forge.py`
- Risk: Forge upgrade silently returns raw stdout instead of parsed witness.
- Priority: Low (research-local), Medium if the Forge track becomes a deliverable.

---

*Concerns audit: 2026-04-22*
