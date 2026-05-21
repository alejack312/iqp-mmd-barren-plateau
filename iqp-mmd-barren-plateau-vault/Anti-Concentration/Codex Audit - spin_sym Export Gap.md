---
title: Codex Audit — spin_sym Export Gap in iqp_mmd → iqp_bp Checkpoint Bridge
date: 2026-04-23
tags:
  - bug
  - codex-audit
  - iqp-mmd
  - iqp-bp
  - checkpoint
  - spin-sym
aliases:
  - spin_sym Export Bug
  - Codex Audit spin_sym
  - Checkpoint Bridge Bug
status: partially-fixed
related:
  - "[[iqp_mmd AC Investigation 2026-04-23]]"
  - "[[Anti-Concentration vs Marginal Agreement]]"
---

# Codex Audit — spin_sym Export Gap

> [!abstract] TL;DR
> `iqp_mmd.checkpoint_export.save_deterministic_iqp_checkpoint` and `iqp_bp.load_iqp_checkpoint` round-trip `(G, θ)` correctly — but `iqp_bp.IQPModel.probability_vector_exact` silently ignores the `spin_sym` flag that `iqpopt.IqpSimulator` applies during training. Any checkpoint trained with `spin_sym=True` loaded into `iqp_bp` produces the **wrong** exact distribution. At `n = 16` the TV gap between the two representations is `0.258` (vs. `≤ 10⁻¹⁵` when `spin_sym=False`). Empirically confirmed 2026-04-23 on a real trained Ising checkpoint.

> [!warning] Affected analyses
> Any `iqp_bp.check_anti_concentration`, `samples_to_probability_vector`-based cross-check, or `per_order_marginal_mismatch` result computed off a checkpoint whose `iqpopt` training used `spin_sym=True`. This includes the paper's `2D_ising` hyperparameter config (`configs/hyperparameters.yaml:30` sets `spin_sym: True`).

---

## 1. How Codex surfaced it

Adversarial `/codex` run on 2026-04-23 (full transcript `results/codex_challenge.jsonl`). Codex's P1 framing:

> "The checkpoint bridge may not be exporting the trained circuit semantics at all. The exporter rebuilds `G` from raw `gates`, but the training model distinguishes `self.gates` from `self.generators`; one parameter can iterate over multiple `gen` supports. The exporter has no test coverage for nested/grouped gates, only flat supports. If `local_gates` or `spin_sym=True` groups generators, the validated `(G, θ)` is a different model."

Codex was partly wrong (for `local_gates`, each "gate" in `iqpopt` is a single-support `[[support]]` — we verified by introspection) and partly right: the `spin_sym` flag **is** a second code path in `iqpopt.IqpSimulator` that the deterministic `iqp_bp.IQPModel` does not implement. `.npz` only carries `(G, θ)` — `spin_sym` is dropped at export and has no place to live on the `iqp_bp` side.

---

## 2. Empirical evidence

Script: `scripts/_verify_checkpoint_faithful.py` — for each checkpoint, compute `q_bp = iqp_bp.IQPModel.probability_vector_exact(θ)` and `q_opt = iqpopt.IqpSimulator(spin_sym=<original>).probs(θ)` on the **same `θ`**, compare.

| Checkpoint | `spin_sym` used in training | `TV(q_bp, q_opt)` | `L∞` | Verdict |
|---|:---:|---:|---:|---|
| `spin_blobs_n16_iters1000_seed666.npz` | False | **0.000000** | 3.47e-17 | ✅ faithful |
| `ising_n16_iters1000_seed666.npz` | **True**  | **0.257637** | 6.41e-02 | ❌ different circuit |
| `ising_…npz` loaded with `spin_sym=False` (control) | False | **0.000000** | 1.19e-15 | ✅ gap = spin_sym exactly |

The control confirms the discrepancy is *entirely* caused by `spin_sym`. When both sides agree on `spin_sym=False`, TV drops to numerical zero.

---

## 3. Collateral findings (not spin_sym)

### 3.1 Vacuous second-moment-threshold flag

`iqp_bp.experiments.run_validation.DEFAULT_SECOND_MOMENT_THRESHOLD = 1.0`, and the check passes iff `2^n Σ p(x)² ≥ 1`. By Cauchy–Schwarz this holds for every probability distribution (equality iff uniform). So `passes_second_moment_threshold` is `True` for *every* distribution — even a delta. Never cite it. The meaningful signal is the **magnitude** of `scaled_second_moment`: values ≫ 1 mean concentrated; value ≈ 1 means close to uniform.

> [!todo] Fix in `src/iqp_bp/experiments/run_validation.py`
> Either (a) remove `passes_second_moment_threshold` from the output schema, or (b) replace the `≥ 1.0` check with something meaningful — e.g., a configurable maximum ratio (mass is spread across at least `1/β` of the support), or the BMS-style asymptotic check `scaled_ss ≤ C` for small `C > 1`.

### 3.2 `generator_matrix_from_gates` brittleness

`iqp_mmd.checkpoint_export.generator_matrix_from_gates` emits one row per top-level "gate." This is correct when each gate has a single generator support (always true for `local_gates`), but **silently wrong** if:

- A builder groups multiple generator supports under one parameter (`init_gates`/`init_coefs` construction allows this).
- Future gate builders return `list[list[list[int]]]` shapes with `len(gate) > 1`.

> [!todo] Guard against this
> Raise if any "gate" has more than one generator support, until the `iqp_bp.IQPModel` supports grouped parameters natively. Test stub:
> ```python
> def test_checkpoint_export_rejects_grouped_gates():
>     grouped = [[[0, 1], [0, 2]]]  # one param over two supports
>     with pytest.raises(ValueError, match="grouped"):
>         generator_matrix_from_gates(grouped, n_qubits=3)
> ```

---

## 4. Fix applied

Used `iqpopt.IqpSimulator.probs(θ)` directly as the source of `q_θ` instead of `iqp_bp.IQPModel.probability_vector_exact`. The `(G, θ)` checkpoint is still useful as a way to persist `θ` and the circuit family; `iqp_bp`'s own exact path is only valid for checkpoints trained with `spin_sym=False`. Corrected pipeline: `scripts/rerun_ac_via_iqpopt.py` (loads `θ` from `.npz`, rebuilds the original `iqpopt` model with the *remembered* `spin_sym` flag, calls `probs`, feeds into existing `check_anti_concentration` + `per_order_marginal_mismatch`).

## 5. Proper fix (not done)

The architecturally clean fix is to add `spin_sym` handling to `iqp_bp.IQPModel`:

$$
q_\theta(z) \;=\; \frac{1}{2}\!\left[ q^{(0)}_\theta(z) \;+\; q^{(0)}_\theta(\bar z) \right]
$$

where $q^{(0)}$ is the standard-IQP distribution (the current `probability_vector_exact`) and $\bar z$ is the bitwise-flipped string. Implementation sketch:

```python
def probability_vector_exact(self, max_qubits=20, spin_sym=False):
    p = self._exact_p_std(max_qubits)
    if spin_sym:
        p = 0.5 * (p + p[::-1])  # Walsh-Hadamard index reversal = bitwise flip
    return p
```

This also requires propagating the `spin_sym` flag through the `.npz` metadata (currently dropped) so the correct forward path can be chosen automatically. TODO on the roadmap.

> [!todo] Ship checklist
> - [x] Add `spin_sym` to `save_deterministic_iqp_checkpoint` metadata and `load_iqp_checkpoint` return.
> - [x] Add `spin_sym` kwarg to `IQPModel.probability_vector_exact` and `check_anti_concentration`-via-model wrappers.
> - [x] Guard `generator_matrix_from_gates` against multi-support gates.
> - [x] Meaningfully gate `passes_second_moment_threshold` in documentation: keep the field for schema compatibility, but treat `scaled_second_moment` magnitude and `beta_hat` as the decision fields.
> - [x] Add checkpoint spin-symmetry round-trip coverage for metadata-driven exact probabilities.

---

## 6. Why this slipped through

- `checkpoint_export.py` tests (`tests/test_iqp_mmd_checkpoint_export.py`) cover flat-gate roundtrip but not `spin_sym`-on / `spin_sym`-off symmetry.
- `iqp_bp.IQPModel` predates the `iqp_mmd` bridge; when the bridge was added, nobody audited `iqpopt.IqpSimulator`'s knobs for ones that change forward semantics.
- The bridge helper docstring in `checkpoint_export.py` says "deterministic checkpoint compatible with `iqp_bp`" — *compatible* is too strong when `spin_sym=True`.

---

## 7. Related

- [[iqp_mmd AC Investigation 2026-04-23]] — the investigation where this bug was found and mitigated.
- [[Anti-Concentration vs Marginal Agreement]] — conceptual framing.
- Memory: `Codex implements, not Claude` — [codex handed off the adversarial audit, we implemented the fix].

%%
Change log:
- 2026-04-23: initial, after Codex challenge + verification + corrected rerun.
%%
