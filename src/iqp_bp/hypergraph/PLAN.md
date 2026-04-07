Hypergraph Validation Plan Status

Current status:
- `T3` complete: `lattice` is now the exact 2D square-grid nearest-neighbor ZZ family.
- `S1` complete: `erdos_renyi` is now the SMART sparse pairwise family with bounded expected degree.
- `V1` complete: the Hypothesis layer now samples the four SMART families only, and the family smoke tests cover the real lattice and ER invariants.

Completed work for V1:
1. Added `smart_family_config()` and `smart_family_instance()` in `src/iqp_bp/hypergraph/hypothesis_strategies.py`.
2. Switched `mmd_instance()` to SMART families only.
3. Tightened the lattice family to the exact deterministic 2D nearest-neighbor ZZ construction.
4. Tightened the ER family to the SMART sparse pairwise regime with intrinsic row count.
5. Added family-specific Hypothesis smoke tests in `tests/test_hypothesis.py`.
6. Added deterministic family tests in `tests/test_hypergraph_families.py`.

What remains:

V5
Goal: add Hypothesis coverage for:
- full small-angle sweep `{0.01, 0.1, 0.3}`
- data-dependent init

Current status:
- `init_config()` now samples the exact small-angle sweep.
- `mmd_instance()` now materializes `uniform`, `small_angle`, and `data_dependent`.
- Hypothesis tests now check exact small-angle choices, theta shape, and finite theta.
- `run_scaling.py` no longer uses the old small Gaussian stub for `data_dependent`.

Remaining gap:
- if we hold `V5` to the stricter original wording, the only unfinished part is the stronger post-`S4` validation layer:
  - prove `data_dependent` depends on data in comparative tests
  - prove two different datasets can produce different `theta` for the same `G` and seed
  - decide whether the current data-aware parity warm start is sufficient as the final `S4` interpretation, or whether a stronger covariance-specific initializer is still required

Recommended closeout order:
1. Add the comparative `data_dependent` tests in `tests/test_hypothesis.py`.
2. Decide whether the current `data_dependent` implementation fully satisfies `S4`.
3. Mark `V5` complete once that decision and the comparative tests are in place.
