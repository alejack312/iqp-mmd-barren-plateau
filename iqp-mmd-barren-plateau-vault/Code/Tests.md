---
title: Tests
tags:
  - code
  - tests
---

# Test Suite

The test suite lives in `tests/` and targets **invariants, not numeric snapshots**. That is deliberate — the implementation still moves as the SMART spec gets tighter, but the mathematics should not.

## Files

```
tests/
├── test_imports.py                      # import + basic API smoke tests
├── test_hypergraph_families.py          # per-family invariants
├── test_hypothesis.py                   # property-based strategies
├── test_mmd_kernel.py                   # kernel formulas + sampler shapes
├── test_expectation_small_n.py          # MC vs exact ⟨Z_a⟩
├── test_gradients_small_n.py            # analytic vs finite-difference
├── test_iqp_probability_vector_small_n.py  # exact FWHT path
├── test_anti_concentration.py           # AC checker on known inputs
├── test_iqp_mmd_checkpoint_export.py    # iqp_mmd → iqp_bp checkpoint bridge
├── test_run_scaling.py                  # scaling runner end-to-end
├── test_run_validation.py               # validation runner end-to-end
└── test_scaling_data_factory.py         # dataset factory invariants
```

## Core Invariants

From the test suite and [[Implementation Choices]]:

- Expectation values $\langle Z_a\rangle \in [-1, 1]$
- Exact 2D lattice edge structure (row count, weight, no duplicates)
- Pairwise ER structure with no duplicate edges
- `theta.shape == G.shape[0]`
- Consistency between MC cosine path and exact path on small $n$
- Anti-concentration scalar identity: `scaled_second_moment = 2^n · Σ p²`
- FWHT round-trip: probability vector sums to 1
- Parity sign math on data-side expectations (catches the [[Mixture Module|uint8 underflow bug]])

## Running Tests

```bash
pytest                      # full suite
pytest tests/test_mmd_kernel.py -v
pytest -k "small_n"         # only small-n validation tests
```

## What Is NOT Tested

Deliberately excluded:

- Frozen numeric output comparisons (they break every time the formula changes)
- End-to-end gradient variance scaling laws (too expensive to run in CI)
- Qiskit path (out-of-scope until the runner is real)

## Related

- [[Hypothesis Strategies]]
- [[Hypergraph Families]]
- [[Implementation Choices]]
- [[TODO Roadmap]]
