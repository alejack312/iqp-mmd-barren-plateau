"""D-Wave quantum annealer dataset download."""

from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class DWaveDataset:
    """Container for D-Wave dataset."""

    train_samples: np.ndarray
    test_samples: np.ndarray


def download_dwave(
    num_train: int = 10000,
    output_dir: str | Path = None,
) -> DWaveDataset:
    """Download D-Wave quantum annealer samples from Zenodo.

    Downloads 484-spin samples from a 3-nearest-neighbor topology.

    Args:
        num_train: Max number of training samples.
        output_dir: If provided, save CSVs here.

    Returns:
        DWaveDataset with normalized binary samples.
    """
    import requests
    import tarfile
    import tempfile

    url = "https://zenodo.org/records/7250436/files/datasets.tar.gz?download=1"

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "datasets.tar.gz"

        print("Downloading D-Wave dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmpdir)

        X_train = np.load(f"{tmpdir}/datasets/484-z8-100mus/train-484spins-3nn-uniform-100mus.npy")
        X_test = np.load(f"{tmpdir}/datasets/484-z8-100mus/test-484spins-3nn-uniform-100mus.npy")

    X_train = X_train.reshape(X_train.shape[0], -1)[:num_train]
    X_train = ((1 + X_train) / 2).astype(int)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = ((1 + X_test) / 2).astype(int)

    dataset = DWaveDataset(train_samples=X_train, test_samples=X_test)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "dwave_X_train.csv", X_train, delimiter=",", fmt="%d")
        np.savetxt(output_dir / "dwave_X_test.csv", X_test, delimiter=",", fmt="%d")

    return dataset
