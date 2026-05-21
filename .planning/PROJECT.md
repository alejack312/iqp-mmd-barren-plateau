# Option 2 — Pauli-Estimator AC + Marginals at Scale

## What This Is

A scalable extension to the project's anti-concentration (AC) and high-order marginal-agreement diagnostics. The 2026-04-23 investigation could only evaluate two paper datasets at `n = 16` because `IQPModel.probability_vector_exact` enumerates all `2^n` probabilities. This milestone replaces that exact path with a **Monte-Carlo estimator over Pauli-Z expectations** (`⟨Z_a⟩_{q_θ}`), using `iqpopt.IqpSimulator.op_expval` (the Recio-Armengol paper's Proposition 1 primitive), so we can compute AC and low-order marginals at `n ∈ {256, 484, 784, 805, 1000}`. Target user: Alejandro's supervisor (24-hour deliverable deadline).

## Core Value

**Produce scaled-second-moment AC and low-order marginal mismatch numbers for the 5 big-n Recio-Armengol datasets within 24 hours**, with honest caveats and a reproducible pipeline.

## Requirements

### Validated

<!-- Inferred from existing code completed before this milestone -->

- ✓ Exact AC on trained `iqp_mmd` checkpoints at `n = 16` — delivered 2026-04-23 in [[iqp_mmd AC Investigation 2026-04-23]]
- ✓ Per-order marginal mismatch `M_k` via full enumeration — `scripts/investigate_iqp_mmd_ac.py`
- ✓ Deterministic `(G, θ)` checkpoint bridge between `iqpopt` trainer and `iqp_bp` validator — `src/iqp_mmd/checkpoint_export.py`
- ✓ Codex-audit workaround: use `iqpopt.probs()` when `spin_sym=True` — [[Codex Audit - spin_sym Export Gap]]

### Active

<!-- v1 scope for this milestone -->

- [ ] **EST-01**: Pauli-expectation-based scaled-second-moment estimator works on any `iqpopt.IqpSimulator(params, gates, spin_sym)` — via Parseval: `2^n Σ q(x)² = (1/2^n) Σ_a ⟨Z_a⟩²`, summed by Monte Carlo over uniformly random Pauli strings `a ∈ {0,1}^n`.
- [ ] **EST-02**: Estimator provides a configurable number of MC samples (Pauli strings) with reported standard error.
- [ ] **EST-03**: Per-order low-`k` marginal mismatch via Pauli expectations: for each subset `S` of size `k`, estimate `⟨Z_a⟩` for `a ⊆ S` and compute TV via the inverse Walsh-Hadamard on the subset.
- [ ] **VAL-01**: Estimator validated against `probability_vector_exact` ground truth at `n = 16` on the existing Ising + blobs checkpoints — learned and empirical AC numbers agree within MC error.
- [ ] **DATA-01**: Attempt to acquire all 5 big-n datasets via `src/iqp_mmd/datasets/*.py` or the `iqp-dataset` CLI; skip-and-document any that fail (best-effort).
- [ ] **TRAIN-01**: For each successfully acquired dataset, train an `iqpopt` model with **reduced budget (`n_iters ≤ 500`)** at paper-spec `max_weight`, `sigma`, `spin_sym` — export deterministic checkpoint.
- [ ] **RUN-01**: Run estimator on each trained checkpoint; emit JSON summary with AC scalars + low-k marginal mismatch + MC uncertainty.
- [ ] **WRITE-01**: Obsidian vault note under `Anti-Concentration/` summarizing results, caveats, comparison to 2026-04-23 n=16 baseline.
- [ ] **PLOT-01**: Headline PNG showing scaled-second-moment vs. `n` across the 5 datasets, plus a per-order marginal-mismatch panel at whichever k-orders are tractable.
- [ ] **REPRO-01**: Pipeline runnable end-to-end via a single CLI (`python scripts/pauli_estimator_investigation.py --dataset <name>`), with logged progress.

### Out of Scope

- **Full-joint TV at n > 20** — infeasible (can't enumerate `2^n`). Only low-`k` marginals are reported.
- **Paper-faithful training (`n_iters = 10 000`)** — budget does not permit. Results will carry an "undertrained surrogate" caveat.
- **Refitting the iqp_bp `IQPModel` to handle `spin_sym`** — workaround via `iqpopt.probs()` is already validated for the n=16 case; not a blocker here.
- **New exploration of barren-plateau trainability** — this milestone is about output-distribution diagnostics, not gradient variance.
- **GUI / notebook integration** — CLI + vault note only.
- **Any work on the 2 small-n datasets (`2D_ising`, `8_blobs`)** — already answered on 2026-04-23.

## Context

- **Prior work to reuse:**
  - `scripts/investigate_iqp_mmd_ac.py` — training + exact-enumeration path; reuse data generators, training harness, output-dir conventions, dedupe bit-ordering.
  - `scripts/rerun_ac_via_iqpopt.py` — the `iqpopt.probs()` workaround for `spin_sym=True`. Pattern for loading a checkpoint + rebuilding the iqpopt model.
  - `scripts/_verify_checkpoint_faithful.py` — TV-comparison template; reuse the "round-trip" idea for VAL-01.
  - `iqp_bp/experiments/run_validation.check_anti_concentration` — feeds a probability vector; we will *not* build a full probability vector but we *will* reuse the threshold/`β̂` logic where relevant.
- **Pauli-estimator math:**
  - Parseval: `Σ_x q(x)² = (1/2^n) Σ_a ⟨Z_a⟩²`, where `a ∈ {0,1}^n` indexes Z-Pauli supports.
  - MC estimator: draw `M` random `a`; estimate `scaled_second_moment ≈ 2^n × (1/M) Σ_m ⟨Z_{a_m}⟩²`. Variance decreases as `O(1/M)`.
  - Low-`k` marginals: the marginal of `q_θ` on subset `S` is determined by `⟨Z_a⟩` for `a` supported in `S`; there are `2^|S|` such Paulis, so tractable for `|S| ≤ ~16`.
- **`iqpopt.op_expval` surface** (confirmed via `inspect`):
  `op_expval(params, ops, n_samples, key, init_coefs=None, indep_estimates=False, return_samples=False, max_batch_ops=None, max_batch_samples=None)`. `ops` is a binary matrix `(num_ops, n_qubits)` encoding Pauli-Z support. Returns estimated expectations (optionally with samples).
- **Known risks:**
  - Dataset acquisition may fail for any of dwave (Zenodo URL), genomic-805 (INRIA GitLab), MNIST (torchvision), Ising-family (`qml_benchmarks` + jax version conflicts).
  - At `n = 1000` and paper `max_weight = 2`, the gate count is `O(n²) ≈ 500k` and a single training step may be expensive; reduced `n_iters = 500` is essential.
  - MC noise at small `M` may exceed signal for nearly-AC distributions. Report standard error alongside every scalar.

## Constraints

- **Timeline**: 24 hours from 2026-04-23 — non-negotiable.
- **Compute**: Single laptop, 16 GB RAM, no GPU. Use `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform` when training to control OOM risk already seen at `n=16` `max_weight=6`.
- **Tech stack**: Python 3.13, `iqpopt` (`pip install git+https://github.com/XanaduAI/iqpopt.git`), `jax` 0.10, `qml_benchmarks` (installed; may break for some generators).
- **Deliverable format**: Runnable code (single-CLI script), JSON/CSV results, and a vault markdown note under `iqp-mmd-barren-plateau-vault/Anti-Concentration/`.
- **Data acquisition**: Best-effort. Skip any dataset whose acquisition breaks, with an explicit note.
- **Training budget**: `n_iters ≤ 500` (paper uses 2000–10 000). Document as "reduced-budget surrogate."
- **Authorship**: Commits as `Alejandro`, no Claude co-author (project convention).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use `iqpopt.op_expval` as the Pauli-expectation primitive | Already present; it *is* Proposition 1 of the paper; JAX-JIT backed | — Pending |
| Parseval Monte Carlo over random Pauli strings for AC | Avoids enumerating `2^n` strings; standard error tractable as `σ/√M` | — Pending |
| Low-`k` marginals only (k ≤ 8 target) | Full joint infeasible at n > 20; low-k is still diagnostic and matches Rudolph framing | — Pending |
| `n_iters ≤ 500` reduced training budget | 24h deadline; honest "undertrained surrogate" caveat | — Pending |
| Best-effort dataset acquisition (skip broken) | Avoids consuming the 24h budget on dependency-fixing; still delivers something for most datasets | — Pending |
| Validate at `n = 16` before running large-n | Cheapest possible ground-truth check — catches bugs before they corrupt expensive runs | — Pending |
| Vault writeup + single PNG as primary deliverable | Matches format of the 2026-04-23 deliverable; already understood by supervisor | — Pending |

---
*Last updated: 2026-04-23 after initialization*
