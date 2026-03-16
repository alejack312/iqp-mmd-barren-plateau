"""Hyperparameter search utilities and grid definitions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ShuffleSplit


def read_data(path: str | Path, labels: bool = True) -> tuple:
    """Read CSV data where each row is a sample.

    Args:
        path: Path to CSV file.
        labels: If True, last column is treated as labels.

    Returns:
        Tuple of (X, y) where y is None if labels=False.
    """
    data = pd.read_csv(path, header=None)
    if labels:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    else:
        X = data.iloc[:, :].values
        y = None
    return X, y


def construct_hyperparameter_grid(hyperparameter_settings: dict) -> dict:
    """Build a sklearn GridSearchCV parameter grid from settings dict.

    Args:
        hyperparameter_settings: Dictionary with type/dtype/val structure.

    Returns:
        Flat dictionary suitable for GridSearchCV.
    """
    grid = {}
    for name, spec in hyperparameter_settings.items():
        if spec["type"] == "list":
            dtype = spec["dtype"]
            val = spec["val"]
            if dtype == "tuple":
                grid[name] = [eval(v) for v in val]
            else:
                grid[name] = np.array(val, dtype=dtype)
    return grid


def run_hyperparameter_search(
    estimator,
    param_grid: dict,
    X: np.ndarray,
    y=None,
    cv: int = 1,
    n_jobs: int = 1,
    results_dir: str | Path = None,
    experiment_name: str = "search",
) -> dict:
    """Run GridSearchCV and optionally save results.

    Args:
        estimator: Sklearn-compatible estimator.
        param_grid: Parameter grid for search.
        X: Training data.
        y: Labels (optional).
        cv: Cross-validation folds or splitter.
        n_jobs: Parallel jobs.
        results_dir: Directory to save results CSVs.
        experiment_name: Name prefix for output files.

    Returns:
        Dictionary with 'best_params', 'cv_results', 'grid_search'.
    """
    if cv == 1:
        cv = ShuffleSplit(test_size=0.20, n_splits=1, random_state=42)

    def scorer(estimator, X, y=None):
        return estimator.score(X, y)

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        verbose=3,
        n_jobs=n_jobs,
        refit=False,
    ).fit(X, y)

    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame.from_dict(gs.cv_results_)
        df.to_csv(results_dir / f"{experiment_name}_results.csv")

        best_df = pd.DataFrame(list(gs.best_params_.items()), columns=["hyperparameter", "best_value"])
        best_df.to_csv(results_dir / f"{experiment_name}_best_hyperparameters.csv", index=False)

    return {
        "best_params": gs.best_params_,
        "cv_results": gs.cv_results_,
        "grid_search": gs,
    }
