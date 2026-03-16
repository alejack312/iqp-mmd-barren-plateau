"""CLI entry point for sampling from trained models."""

import argparse
from pathlib import Path

import numpy as np

from iqp_mmd.datasets.loaders import load_csv_dataset, normalize_binary
from iqp_mmd.config.hyperparams import load_hyperparams
from iqp_mmd.sampling.sampler import sample_from_model


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained generative model.")
    parser.add_argument("--model", type=str, required=True, choices=[
        "IqpSimulator", "IqpSimulatorBitflip", "RestrictedBoltzmannMachine", "DeepEBM", "DeepGraphEBM",
    ])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--params-path", type=str, required=True, help="Path to pickled parameters")
    parser.add_argument("--hyperparams", type=str, required=True, help="Path to hyperparameters YAML")
    parser.add_argument("--ref-data", type=str, default=None, help="Reference data for model init")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--graph-path", type=str, default=None, help="Graph adjlist (for DeepGraphEBM)")
    args = parser.parse_args()

    all_hyperparams = load_hyperparams(args.hyperparams)
    hyperparams = all_hyperparams[args.model][args.dataset]

    X_ref = None
    if args.ref_data:
        X_ref = np.array(normalize_binary(load_csv_dataset(args.ref_data)))

    graph = None
    if args.graph_path:
        import networkx as nx
        graph = nx.read_adjlist(args.graph_path)

    samples = sample_from_model(
        model_name=args.model,
        params_path=args.params_path,
        hyperparams=hyperparams,
        num_samples=args.num_samples,
        X_ref=X_ref,
        graph=graph,
        output_path=args.output,
    )

    print(f"Generated {len(samples)} samples with shape {samples.shape}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
