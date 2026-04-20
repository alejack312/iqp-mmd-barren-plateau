"""Tests for the canonical RNG stream policy in iqp_bp.rng."""

from __future__ import annotations

import numpy as np
import pytest

from iqp_bp.rng import (
    CANONICAL_STREAMS,
    STREAM_CIRCUIT,
    STREAM_DATA,
    STREAM_ESTIMATION,
    STREAM_FORGE,
    STREAM_KERNEL,
    STREAM_QISKIT,
    STREAM_THETA,
    derive_seed,
    experiment_stream_bundle,
    split_rng,
)


# ---------------------------------------------------------------------------
# Canonical stream name constants
# ---------------------------------------------------------------------------

class TestCanonicalStreamNames:
    def test_all_seven_streams_present(self):
        assert set(CANONICAL_STREAMS) == {
            "circuit", "data", "theta", "kernel", "estimation", "qiskit", "forge"
        }

    def test_constants_are_non_empty_strings(self):
        for name in CANONICAL_STREAMS:
            assert isinstance(name, str) and len(name) > 0

    def test_each_constant_is_in_canonical(self):
        assert STREAM_CIRCUIT in CANONICAL_STREAMS
        assert STREAM_DATA in CANONICAL_STREAMS
        assert STREAM_THETA in CANONICAL_STREAMS
        assert STREAM_KERNEL in CANONICAL_STREAMS
        assert STREAM_ESTIMATION in CANONICAL_STREAMS
        assert STREAM_QISKIT in CANONICAL_STREAMS
        assert STREAM_FORGE in CANONICAL_STREAMS

    def test_no_duplicate_stream_names(self):
        assert len(CANONICAL_STREAMS) == len(set(CANONICAL_STREAMS))


# ---------------------------------------------------------------------------
# experiment_stream_bundle
# ---------------------------------------------------------------------------

class TestExperimentStreamBundle:
    def test_returns_all_canonical_keys(self):
        bundle = experiment_stream_bundle(42, "test", "coord")
        assert set(bundle.keys()) == set(CANONICAL_STREAMS)

    def test_values_are_valid_seed_ints(self):
        bundle = experiment_stream_bundle(0)
        for v in bundle.values():
            assert isinstance(v, int)
            assert 0 <= v < 2**31 - 1

    def test_stable_across_calls(self):
        """Same arguments → identical dict on repeated invocations."""
        b1 = experiment_stream_bundle(7, "run_scaling", "n4_complete_graph")
        b2 = experiment_stream_bundle(7, "run_scaling", "n4_complete_graph")
        assert b1 == b2

    def test_different_base_seeds_produce_different_bundles(self):
        b1 = experiment_stream_bundle(1, "coord")
        b2 = experiment_stream_bundle(2, "coord")
        assert b1 != b2

    def test_different_coord_parts_produce_different_bundles(self):
        b1 = experiment_stream_bundle(42, "coord_a")
        b2 = experiment_stream_bundle(42, "coord_b")
        assert b1 != b2

    def test_streams_are_mutually_distinct(self):
        """All canonical seed values in the bundle must be different."""
        bundle = experiment_stream_bundle(42, "test")
        values = list(bundle.values())
        assert len(values) == len(set(values)), (
            "Two streams produced the same seed — they are not independent"
        )

    def test_loop_order_invariance(self):
        """Seeds for a coordinate do not depend on which loop iteration resolves it."""
        # Simulate two different loop orderings over (family, n) pairs
        pairs = [("lattice", 4), ("complete_graph", 8), ("erdos_renyi", 6)]
        forward  = [experiment_stream_bundle(99, "run", fam, n) for fam, n in pairs]
        backward = [experiment_stream_bundle(99, "run", fam, n) for fam, n in reversed(pairs)]

        for (fam, n), b_fwd in zip(pairs, forward):
            b_rev = next(b for (f, nn), b in zip(reversed(pairs), backward) if f == fam and nn == n)
            assert b_fwd == b_rev, f"Seed for ({fam}, {n}) changed under different iteration order"


# ---------------------------------------------------------------------------
# split_rng
# ---------------------------------------------------------------------------

class TestSplitRng:
    def test_returns_requested_count(self):
        rng = np.random.default_rng(0)
        children = split_rng(rng, 4)
        assert len(children) == 4

    def test_returns_generators(self):
        rng = np.random.default_rng(0)
        for c in split_rng(rng, 3):
            assert isinstance(c, np.random.Generator)

    def test_children_differ_from_each_other(self):
        rng = np.random.default_rng(0)
        c1, c2 = split_rng(rng, 2)
        v1 = c1.integers(0, 2**31)
        v2 = c2.integers(0, 2**31)
        assert v1 != v2

    def test_deterministic_given_same_parent_seed(self):
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        (a1, b1) = split_rng(r1, 2)
        (a2, b2) = split_rng(r2, 2)
        assert a1.integers(0, 2**31) == a2.integers(0, 2**31)
        assert b1.integers(0, 2**31) == b2.integers(0, 2**31)

    def test_single_child(self):
        rng = np.random.default_rng(7)
        (child,) = split_rng(rng, 1)
        assert isinstance(child, np.random.Generator)

    def test_parent_advances_by_exactly_one_draw(self):
        """Parent rng should advance by one integer draw regardless of n."""
        r1 = np.random.default_rng(55)
        r2 = np.random.default_rng(55)
        _ = r1.integers(0, 2**31)  # simulate the one draw split_rng makes
        split_rng(r2, 5)
        # Both rngs should now be at the same state
        assert r1.integers(0, 2**31) == r2.integers(0, 2**31)


# ---------------------------------------------------------------------------
# Stream separation via derive_seed
# ---------------------------------------------------------------------------

class TestDeriveStreamSeparation:
    def test_theta_and_estimation_streams_differ(self):
        base = 123
        coord = ("run_scaling", "some_setting")
        theta_seed = derive_seed(base, *coord, STREAM_THETA, 0)
        est_seed   = derive_seed(base, *coord, STREAM_ESTIMATION, 0)
        assert theta_seed != est_seed

    def test_theta_seeds_are_unique_per_index(self):
        base = 7
        coord = ("run_scaling", "setting_x")
        seeds = [derive_seed(base, *coord, STREAM_THETA, i) for i in range(8)]
        assert len(seeds) == len(set(seeds)), "per-index theta seeds must be unique"

    def test_estimation_seeds_are_unique_per_param(self):
        base = 7
        coord = ("run_scaling", "setting_x")
        seeds = [derive_seed(base, *coord, STREAM_ESTIMATION, i) for i in range(8)]
        assert len(seeds) == len(set(seeds))

    def test_circuit_and_kernel_streams_differ(self):
        base = 99
        coord = ("run_scaling", "coord")
        assert derive_seed(base, *coord, STREAM_CIRCUIT) != derive_seed(base, *coord, STREAM_KERNEL)

    def test_stream_seed_stable_across_base_seed_types(self):
        """derive_seed should produce identical results whether base_seed is int or np.int64."""
        base_int = 42
        base_np  = np.int64(42)
        coord = ("run_scaling", "coord")
        assert derive_seed(base_int, *coord, STREAM_THETA, 0) == derive_seed(base_np, *coord, STREAM_THETA, 0)

    @pytest.mark.parametrize("stream", [
        STREAM_CIRCUIT, STREAM_DATA, STREAM_THETA,
        STREAM_KERNEL, STREAM_ESTIMATION, STREAM_QISKIT, STREAM_FORGE,
    ])
    def test_bundle_seed_matches_derive_seed(self, stream):
        """experiment_stream_bundle should give the same value as a manual derive_seed call."""
        base = 17
        coord = ("run_scaling", "coord_abc")
        bundle = experiment_stream_bundle(base, *coord)
        manual = derive_seed(base, *coord, "stream", stream)
        assert bundle[stream] == manual
