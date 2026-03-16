"""Path configuration for datasets, parameters, and outputs."""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatasetPaths:
    """Centralized path management for all dataset-related files.

    Args:
        base_dir: Root directory containing datasets, training outputs, etc.
    """

    base_dir: Path

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    @property
    def datasets_dir(self) -> Path:
        return self.base_dir / "datasets"

    @property
    def params_dir(self) -> Path:
        return self.base_dir / "training" / "trained_parameters"

    @property
    def hyperparams_path(self) -> Path:
        return self.base_dir / "training" / "best_hyperparameters.yaml"

    @property
    def samples_dir(self) -> Path:
        return self.base_dir / "plots" / "samples"

    @property
    def tables_dir(self) -> Path:
        return self.base_dir / "plots" / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "plots" / "figures"

    @property
    def cov_dir(self) -> Path:
        return self.base_dir / "plots" / "cov_matrix"

    @property
    def loss_plots_dir(self) -> Path:
        return self.base_dir / "training" / "loss_plots"

    def train_path(self, dataset_name: str) -> Path:
        """Get training data path for a dataset."""
        paths = {
            "8_blobs": self.datasets_dir / "blobs/8_blobs_dataset/16_spins_8_blobs_train.csv",
            "2D_ising": self.datasets_dir / "ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv",
            "spin_glass": self.datasets_dir / "ising/spin_glass_dataset/ising_spin_glass_N_256_T_0.1_train.csv",
            "scale_free": self.datasets_dir / "ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv",
            "MNIST": self.datasets_dir / "MNIST/x_train.csv",
            "genomic-805": self.datasets_dir / "genomic/805_SNP_1000G_real_train.csv",
            "dwave": self.datasets_dir / "dwave/dwave_X_train.csv",
        }
        if dataset_name not in paths:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(paths.keys())}")
        return paths[dataset_name]

    def test_path(self, dataset_name: str) -> Path:
        """Get test data path for a dataset."""
        paths = {
            "8_blobs": self.datasets_dir / "blobs/8_blobs_dataset/16_spins_8_blobs_test.csv",
            "2D_ising": self.datasets_dir / "ising/2d_random_lattice_dataset/ising_4_4_T_3_test.csv",
            "spin_glass": self.datasets_dir / "ising/spin_glass_dataset/ising_spin_glass_N_256_T_0.1_test.csv",
            "scale_free": self.datasets_dir / "ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_test.csv",
            "MNIST": self.datasets_dir / "MNIST/x_test.csv",
            "genomic-805": self.datasets_dir / "genomic/805_SNP_1000G_real_test.csv",
            "dwave": self.datasets_dir / "dwave/dwave_X_test.csv",
        }
        if dataset_name not in paths:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(paths.keys())}")
        return paths[dataset_name]

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.params_dir, self.samples_dir, self.tables_dir, self.figures_dir, self.cov_dir, self.loss_plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
