"""Common dataset loading and preprocessing utilities."""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path


def load_csv_dataset(path: str | Path, delimiter: str = ",", header=None) -> np.ndarray:
    """Load a CSV dataset of binary samples.

    Args:
        path: Path to CSV file.
        delimiter: Column delimiter.
        header: Row number(s) to use as column names.

    Returns:
        Numpy array of samples.
    """
    return pd.read_csv(path, delimiter=delimiter, header=header).to_numpy()


def normalize_binary(X: np.ndarray) -> jnp.ndarray:
    """Convert {-1, +1} encoded data to {0, 1}.

    If data is already in {0, 1}, returns as-is.

    Args:
        X: Input array.

    Returns:
        Array with values in {0, 1}.
    """
    if -1 in X:
        return jnp.array((1 + X) // 2, dtype=int)
    return jnp.array(X, dtype=int)
