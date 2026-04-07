from __future__ import annotations

import numpy as np

from iqp_bp.experiments.data_factory import make_dataset
from iqp_bp.experiments.run_scaling import resolve_scaling_settings


def test_make_dataset_product_bernoulli_is_binary_and_reproducible():
    cfg = {"type": "product_bernoulli", "n_samples": 32}

    data0, metadata0 = make_dataset(cfg, n=6, seed=123)
    data1, metadata1 = make_dataset(cfg, n=6, seed=123)

    assert data0.shape == (32, 6)
    assert data0.dtype == np.uint8
    assert np.array_equal(data0, data1)
    assert metadata0 == metadata1
    assert set(np.unique(data0)).issubset({0, 1})


def test_make_dataset_binary_mixture_is_binary_and_reproducible():
    cfg = {
        "type": "binary_mixture",
        "n_samples": 48,
        "binary_mixture": {
            "n_modes": 3,
            "noise": 0.2,
        },
    }

    data0, metadata0 = make_dataset(cfg, n=8, seed=77)
    data1, metadata1 = make_dataset(cfg, n=8, seed=77)

    assert data0.shape == (48, 8)
    assert data0.dtype == np.uint8
    assert np.array_equal(data0, data1)
    assert metadata0["type"] == "binary_mixture"
    assert metadata0["n_modes"] == 3
    assert metadata0["noise"] == 0.2
    assert metadata0 == metadata1
    assert set(np.unique(data0)).issubset({0, 1})
    assert np.unique(data0, axis=0).shape[0] > 1


def test_make_dataset_ising_grid_is_reproducible_and_structured():
    cfg = {
        "type": "ising",
        "n_samples": 64,
        "ising": {
            "beta": 1.5,
            "coupling_std": 1.0,
            "topology": "grid_2d",
            "burn_in_sweeps": 40,
            "thinning": 1,
            "num_chains": 2,
        },
    }

    data0, metadata0 = make_dataset(cfg, n=4, seed=19)
    data1, metadata1 = make_dataset(cfg, n=4, seed=19)

    assert data0.shape == (64, 4)
    assert data0.dtype == np.uint8
    assert np.array_equal(data0, data1)
    assert metadata0 == metadata1
    assert metadata0["type"] == "ising"
    assert metadata0["topology"] == "grid_2d"
    assert metadata0["grid_side"] == 2

    spins = 1.0 - 2.0 * data0.astype(np.float64)
    corr = np.corrcoef(spins, rowvar=False)
    off_diag = np.abs(corr[np.triu_indices(4, k=1)])
    assert np.nanmax(off_diag) > 0.05


def test_resolve_scaling_settings_expands_only_relevant_axes():
    cfg = {
        "circuit": {
            "family": ["product_state", "erdos_renyi"],
            "n_qubits": [4],
            "erdos_renyi": {"p_edge": [2.0, 4.0]},
        },
        "kernel": {
            "type": ["gaussian"],
            "bandwidth": [0.5, 1.0],
        },
        "init": {
            "scheme": ["uniform", "small_angle"],
            "small_angle": {"std": [0.1, 0.3]},
        },
        "dataset": {
            "type": "product_bernoulli",
        },
    }

    settings = resolve_scaling_settings(cfg)

    assert len(settings) == 18
    assert all(setting["dataset_type"] == "product_bernoulli" for setting in settings)

    product_uniform = [
        setting
        for setting in settings
        if setting["family"] == "product_state" and setting["init_scheme"] == "uniform"
    ]
    assert len(product_uniform) == 2
    assert {setting["bandwidth"] for setting in product_uniform} == {0.5, 1.0}
    assert all(setting["er_p_edge"] is None for setting in product_uniform)
    assert all(setting["small_angle_std"] is None for setting in product_uniform)

    er_small_angle = [
        setting
        for setting in settings
        if setting["family"] == "erdos_renyi" and setting["init_scheme"] == "small_angle"
    ]
    assert len(er_small_angle) == 8
    assert {setting["bandwidth"] for setting in er_small_angle} == {0.5, 1.0}
    assert {setting["er_p_edge"] for setting in er_small_angle} == {2.0, 4.0}
    assert {setting["small_angle_std"] for setting in er_small_angle} == {0.1, 0.3}
