"""CLI entry point for evaluating trained models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iqp_mmd.datasets.loaders import load_csv_dataset, normalize_binary
from iqp_mmd.config.hyperparams import load_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model with MMD loss or KGEL.")
    parser.add_argument("--metric", type=str, required=True, choices=["mmd", "kgel"])
    parser.add_argument("--samples-path", type=str, required=True, help="Path to model samples CSV")
    parser.add_argument("--test-path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--sigma", type=float, nargs="+", required=True, help="Kernel bandwidth(s)")
    parser.add_argument("--n-repeats", type=int, default=20, help="Repeats for std estimation (MMD)")
    parser.add_argument("--n-witnesses", type=int, default=10, help="Witness points (KGEL)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    X_test = normalize_binary(load_csv_dataset(args.test_path))
    model_samples = load_csv_dataset(args.samples_path, delimiter=" ").astype(int)

    results = []

    if args.metric == "mmd":
        from iqp_mmd.metrics.mmd_eval import evaluate_mmd_loss

        for sigma in args.sigma:
            result = evaluate_mmd_loss(
                ground_truth=np.array(X_test),
                model_samples=model_samples,
                sigma=sigma,
                n_repeats=args.n_repeats,
            )
            results.append({"sigma": sigma, "mmd_loss": result["mean"], "mmd_std": result["std"]})
            print(f"sigma={sigma:.4f}  MMD={result['mean']:.6f} ± {result['std']:.6f}")

    elif args.metric == "kgel":
        from iqp_mmd.metrics.kgel import evaluate_kgel

        witnesses = np.array(X_test[-args.n_witnesses:])
        ground_truth = np.array(X_test[: -args.n_witnesses])

        for sigma in args.sigma:
            result = evaluate_kgel(
                ground_truth=ground_truth,
                witnesses=witnesses,
                sigma=sigma,
                model_samples=model_samples,
            )
            results.append({"sigma": sigma, "kgel": result["kgel"]})
            print(f"sigma={sigma:.4f}  KGEL={result['kgel']:.6f}")

    if args.output and results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
