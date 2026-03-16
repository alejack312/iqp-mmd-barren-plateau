"""Genomic SNP dataset download from INRIA GitLab."""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class GenomicDataset:
    """Container for genomic SNP dataset."""

    train_samples: np.ndarray
    test_samples: np.ndarray
    variant_count: int


def _load_hapt(filename: str) -> np.ndarray:
    """Load a .hapt file, dropping first two columns."""
    data = pd.read_csv(filename, delim_whitespace=True, header=None)
    return data.drop(columns=[0, 1]).values


def download_genomic(
    variant: str = "805",
    test_fraction: float = 1 / 3,
    seed: int = 666,
    output_dir: str | Path = None,
) -> GenomicDataset:
    """Download genomic SNP data from 1000 Genomes Project.

    Args:
        variant: '805' for 805 SNPs or '10k' for ~10K SNPs.
        test_fraction: Fraction of data for test split.
        seed: Random seed for train/test split.
        output_dir: If provided, save CSVs here.

    Returns:
        GenomicDataset with train/test splits.
    """
    import os
    import zipfile
    import tempfile

    np.random.seed(seed)

    base_url = "https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e"

    with tempfile.TemporaryDirectory() as tmpdir:
        if variant == "805":
            url = f"{base_url}/1000G_real_genomes/805_SNP_1000G_real.hapt"
            filename = f"{tmpdir}/805_SNP_1000G_real.hapt"
            os.system(f'wget -q -O "{filename}" "{url}"')
            data = _load_hapt(filename)
        elif variant == "10k":
            url = f"{base_url}/1000G_real_genomes/10K_SNP_1000G_real.hapt.zip?inline=false"
            zipname = f"{tmpdir}/10K_SNP_1000G_real.hapt.zip"
            os.system(f'wget -q -O "{zipname}" "{url}"')
            with zipfile.ZipFile(zipname, "r") as z:
                z.extractall(tmpdir)
            data = _load_hapt(f"{tmpdir}/10K_SNP_1000G_real.hapt")
        else:
            raise ValueError(f"Unknown variant: {variant}. Use '805' or '10k'.")

        X_train, X_test = train_test_split(data, test_size=test_fraction, random_state=seed)

    dataset = GenomicDataset(
        train_samples=X_train,
        test_samples=X_test,
        variant_count=X_train.shape[1],
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / f"{variant}_SNP_train.csv", X_train, fmt="%d", delimiter=",")
        np.savetxt(output_dir / f"{variant}_SNP_test.csv", X_test, fmt="%d", delimiter=",")

    return dataset
