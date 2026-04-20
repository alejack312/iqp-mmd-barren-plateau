"""Tests: deterministic chunking and exact-vs-MC agreement for the streaming estimator.

Covers:
  1. Different batch sizes (divisible and non-divisible) produce the same
     result as unbatched for the same RNG seed.
  2. The batched estimator agrees with the exact brute-force value for small n.
  3. Extreme chunk sizes: batch_size=1 and batch_size >= num_z_samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from iqp_bp.hypergraph.families import bounded_degree
from iqp_bp.iqp.expectation import iqp_expectation, iqp_expectation_exact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instance(n: int, m: int, seed: int):
    rng = np.random.default_rng(seed)
    G = bounded_degree(n=n, m=m, max_weight=2, rng=rng)
    theta = rng.uniform(-np.pi, np.pi, size=m)
    a = rng.integers(0, 2, size=n, dtype=np.uint8)
    a[0] = 1  # ensure a is nonzero
    return G, theta, a


# ---------------------------------------------------------------------------
# 1. Deterministic chunking — different batch sizes, same RNG seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("batch_size,num_z_samples", [
    (64, 256),    # divisible: 4 full chunks
    (100, 300),   # divisible: 3 full chunks
    (100, 250),   # non-divisible: chunks of 100, 100, 50
    (30, 100),    # non-divisible: chunks of 30, 30, 30, 10
    # Note: batch_size=1 is intentionally excluded here.  NumPy's Generator
    # does NOT guarantee that drawing size=(N, n) at once produces the same
    # bit sequence as N separate size=(1, n) draws (internal buffer alignment
    # differs).  The batch_size=1 case is tested for self-consistency and
    # correctness against the exact brute-force value in separate tests below.
    (16, 16),     # single chunk equal to total
    (256, 256),   # single chunk equal to total (larger)
])
def test_chunked_matches_unbatched(n: int, batch_size: int, num_z_samples: int):
    """Any batch_size > 1 must yield the same (estimate, stderr) as the unbatched path."""
    seed = 42
    G, theta, a = _make_instance(n, m=n, seed=seed)

    est_ref, se_ref = iqp_expectation(
        theta, G, a,
        num_z_samples=num_z_samples,
        rng=np.random.default_rng(seed + 1),
    )
    est_batched, se_batched = iqp_expectation(
        theta, G, a,
        num_z_samples=num_z_samples,
        batch_size=batch_size,
        rng=np.random.default_rng(seed + 1),
    )

    assert abs(est_batched - est_ref) < 1e-10, (
        f"n={n}, batch_size={batch_size}, num_z={num_z_samples}: "
        f"estimate mismatch {est_batched:.12f} vs {est_ref:.12f}"
    )
    assert abs(se_batched - se_ref) < 1e-10, (
        f"n={n}, batch_size={batch_size}, num_z={num_z_samples}: "
        f"stderr mismatch {se_batched:.12f} vs {se_ref:.12f}"
    )


# ---------------------------------------------------------------------------
# 2. Exact-vs-MC agreement for batched estimator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed,batch_size", [
    (4, 0, 128),
    (6, 1, 256),
    (8, 2, 512),
    (10, 3, 256),
])
def test_batched_vs_exact(n: int, seed: int, batch_size: int):
    """Batched MC estimate should agree with brute-force exact value within 3σ."""
    G, theta, a = _make_instance(n, m=n, seed=seed)
    exact = iqp_expectation_exact(theta, G, a)

    est, stderr = iqp_expectation(
        theta, G, a,
        num_z_samples=4096,
        batch_size=batch_size,
        rng=np.random.default_rng(seed + 200),
    )
    tol = max(3 * stderr, 0.05)
    assert abs(est - exact) < tol, (
        f"n={n}, batch_size={batch_size}: |est-exact|={abs(est - exact):.4f} > tol={tol:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Extreme chunk sizes
# ---------------------------------------------------------------------------

def test_batch_size_one_is_deterministic():
    """batch_size=1 must give the same result when called twice with the same seed."""
    n, m = 6, 6
    seed = 99
    num_z_samples = 32
    G, theta, a = _make_instance(n, m, seed)

    est1, se1 = iqp_expectation(
        theta, G, a, num_z_samples=num_z_samples, batch_size=1,
        rng=np.random.default_rng(1234),
    )
    est2, se2 = iqp_expectation(
        theta, G, a, num_z_samples=num_z_samples, batch_size=1,
        rng=np.random.default_rng(1234),
    )
    assert est1 == est2, f"batch_size=1 not deterministic: {est1} vs {est2}"
    assert se1 == se2


def test_batch_size_one_vs_exact():
    """batch_size=1 estimate must agree with brute-force exact value within 3σ."""
    n, m = 6, 6
    G, theta, a = _make_instance(n, m, seed=11)
    exact = iqp_expectation_exact(theta, G, a)

    est, stderr = iqp_expectation(
        theta, G, a, num_z_samples=4096, batch_size=1,
        rng=np.random.default_rng(9999),
    )
    tol = max(3 * stderr, 0.05)
    assert abs(est - exact) < tol, (
        f"batch_size=1: |est-exact|={abs(est - exact):.4f} > tol={tol:.4f}"
    )


def test_batch_size_larger_than_total():
    """batch_size > num_z_samples should produce a single chunk (no chunking at all)."""
    n, m = 6, 6
    seed = 55
    num_z_samples = 64
    G, theta, a = _make_instance(n, m, seed)

    ref, se_ref = iqp_expectation(
        theta, G, a,
        num_z_samples=num_z_samples,
        rng=np.random.default_rng(777),
    )
    batched, se_batched = iqp_expectation(
        theta, G, a,
        num_z_samples=num_z_samples,
        batch_size=num_z_samples * 10,  # much larger than num_z_samples
        rng=np.random.default_rng(777),
    )
    assert abs(batched - ref) < 1e-10
    assert abs(se_batched - se_ref) < 1e-10


def test_nondivisible_exhausts_all_samples():
    """Non-divisible batch_size must consume exactly num_z_samples total samples.

    We verify this indirectly: the result must match the unbatched path drawn
    from the same seed, which requires all 100 samples to be consumed in order.
    """
    n, m = 6, 6
    seed = 77
    G, theta, a = _make_instance(n, m, seed)

    # 100 samples, batch_size=30 → chunks [30, 30, 30, 10]
    ref, _ = iqp_expectation(
        theta, G, a,
        num_z_samples=100,
        rng=np.random.default_rng(5678),
    )
    batched, _ = iqp_expectation(
        theta, G, a,
        num_z_samples=100,
        batch_size=30,
        rng=np.random.default_rng(5678),
    )
    assert abs(batched - ref) < 1e-10


def test_zero_num_z_samples_rejected():
    G, theta, a = _make_instance(4, m=4, seed=123)
    with pytest.raises(ValueError, match="num_z_samples must be positive"):
        iqp_expectation(theta, G, a, num_z_samples=0, rng=np.random.default_rng(1))


@pytest.mark.parametrize("bad_batch_size", [0, -1])
def test_nonpositive_batch_size_rejected(bad_batch_size: int):
    G, theta, a = _make_instance(4, m=4, seed=456)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        iqp_expectation(
            theta,
            G,
            a,
            num_z_samples=16,
            batch_size=bad_batch_size,
            rng=np.random.default_rng(2),
        )
