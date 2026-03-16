"""Smoke tests to verify the package structure is importable."""


def test_package_version():
    import iqp_mmd
    assert iqp_mmd.__version__ == "0.1.0"


def test_import_observables():
    from iqp_mmd.observables import (
        z_ham_pow,
        moment_ham_exp_val_iqp,
        magnet_moment_iqp,
        energy_moment_iqp,
        magnet_moment_samples,
        energy_moment_samples,
    )


def test_import_circuits():
    from iqp_mmd.circuits import penn_obs, penn_op_expval, penn_mmd_loss


def test_import_models():
    from iqp_mmd.models import IqpSimulator, DeepGraphEBM, GraphEBM


def test_import_datasets():
    from iqp_mmd.datasets import load_csv_dataset, normalize_binary


def test_import_config():
    from iqp_mmd.config import DatasetPaths, load_hyperparams, DATASET_NAMES


def test_import_training():
    from iqp_mmd.training import train_iqp, prepare_iqp_training, train_rbm, train_ebm, train_graph_ebm


def test_import_metrics():
    from iqp_mmd.metrics.mmd_eval import evaluate_mmd_loss
    from iqp_mmd.metrics.kgel import evaluate_kgel
    from iqp_mmd.metrics.covariance import compute_covariance_matrix


def test_import_sampling():
    from iqp_mmd.sampling import sample_from_model


def test_config_paths():
    from iqp_mmd.config.paths import DatasetPaths
    paths = DatasetPaths(base_dir="/tmp/test")
    assert paths.datasets_dir.name == "datasets"
    assert paths.params_dir.name == "trained_parameters"


def test_config_hyperparams():
    from pathlib import Path
    from iqp_mmd.config.hyperparams import load_hyperparams

    config_path = Path(__file__).parent.parent / "configs" / "hyperparameters.yaml"
    if config_path.exists():
        hp = load_hyperparams(config_path)
        assert "IqpSimulator" in hp
        assert "2D_ising" in hp["IqpSimulator"]


def test_normalize_binary():
    import numpy as np
    from iqp_mmd.datasets.loaders import normalize_binary

    # Already {0,1}
    X = np.array([[0, 1, 0], [1, 0, 1]])
    result = normalize_binary(X)
    assert result.shape == (2, 3)

    # Convert from {-1,+1}
    X_pm = np.array([[-1, 1, -1], [1, -1, 1]])
    result = normalize_binary(X_pm)
    assert int(result[0, 0]) == 0
    assert int(result[0, 1]) == 1


def test_z_ham_pow():
    import numpy as np
    from iqp_mmd.observables.hamiltonian import z_ham_pow

    ops = np.array([[1, 0], [0, 1]])
    coeffs = np.array([1.0, 1.0])
    new_ops, new_coeffs = z_ham_pow(ops, coeffs, 2)
    assert len(new_ops) > 0
    assert len(new_ops) == len(new_coeffs)


def test_magnet_moment_samples():
    import numpy as np
    from iqp_mmd.observables.hamiltonian import magnet_moment_samples

    samples = np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
    per_sample, mean, std_err = magnet_moment_samples(samples, moment=1)
    assert per_sample.shape == (3,)
    assert mean is not None


def test_energy_moment_samples():
    import numpy as np
    from iqp_mmd.observables.hamiltonian import energy_moment_samples

    samples = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]])
    j_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    per_sample, mean, std_err = energy_moment_samples(samples, moment=1, j_matrix=j_matrix)
    assert per_sample.shape == (3,)
