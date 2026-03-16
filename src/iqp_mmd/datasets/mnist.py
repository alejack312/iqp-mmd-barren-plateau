"""MNIST binary dataset download and preprocessing."""

from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class MNISTDataset:
    """Container for binarized MNIST dataset."""

    train_samples: np.ndarray
    test_samples: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray


def download_mnist(
    num_train: int = 50000,
    num_test: int = 10000,
    threshold: float = 0.5,
    output_dir: str | Path = None,
) -> MNISTDataset:
    """Download MNIST and binarize pixel values.

    Args:
        num_train: Number of training samples.
        num_test: Number of test samples.
        threshold: Binarization threshold (pixels > threshold become 1).
        output_dir: If provided, save CSVs here.

    Returns:
        MNISTDataset with binarized images.
    """
    import torchvision

    raw = torchvision.datasets.MNIST(root="/tmp/mnist_data", train=True, download=True)

    X_train = np.array(raw.data[:num_train]).reshape(num_train, -1) / 256.0
    X_test = np.array(raw.data[num_train : num_train + num_test]).reshape(num_test, -1) / 256.0
    y_train = np.array(raw.targets[:num_train])
    y_test = np.array(raw.targets[num_train : num_train + num_test])

    X_train = (X_train > threshold).astype(int)
    X_test = (X_test > threshold).astype(int)

    dataset = MNISTDataset(
        train_samples=X_train,
        test_samples=X_test,
        train_labels=y_train,
        test_labels=y_test,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "x_train.csv", X_train, delimiter=",", fmt="%d")
        np.savetxt(output_dir / "x_test.csv", X_test, delimiter=",", fmt="%d")
        np.savetxt(output_dir / "y_train.csv", y_train, delimiter=",", fmt="%d")
        np.savetxt(output_dir / "y_test.csv", y_test, delimiter=",", fmt="%d")

    return dataset
