"""CLI entry point for training models."""

import argparse
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import yaml

from iqp_mmd.datasets.loaders import load_csv_dataset, normalize_binary
from iqp_mmd.config.hyperparams import load_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Train a generative model on binary data.")
    parser.add_argument("--model", type=str, required=True, choices=[
        "IqpSimulator", "IqpSimulatorBitflip", "RestrictedBoltzmannMachine", "DeepEBM", "DeepGraphEBM",
    ])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--hyperparams", type=str, required=True, help="Path to hyperparameters YAML")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--graph-path", type=str, default=None, help="Graph adjlist path (for DeepGraphEBM)")
    parser.add_argument("--val-frac", type=float, default=None, help="Validation fraction (IQP only)")
    args = parser.parse_args()

    hyperparams = load_hyperparams(args.hyperparams)
    X_train = load_csv_dataset(args.dataset_path)
    X_train = normalize_binary(X_train)

    output_dir = Path(args.output_dir)

    if args.model in ("IqpSimulator", "IqpSimulatorBitflip"):
        from iqp_mmd.training.iqp_trainer import train_iqp

        bitflip = args.model == "IqpSimulatorBitflip"
        model_key = args.model
        result = train_iqp(
            hyperparams=hyperparams[model_key][args.dataset],
            X_train=jnp.array(X_train),
            dataset_name=args.dataset,
            bitflip=bitflip,
            val_frac=args.val_frac,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"Training complete. Final loss: {result['losses'][-1]:.6f}")

    elif args.model == "RestrictedBoltzmannMachine":
        from iqp_mmd.training.rbm_trainer import train_rbm

        result = train_rbm(
            hyperparams=hyperparams["RestrictedBoltzmannMachine"][args.dataset],
            X_train=np.array(X_train),
            dataset_name=args.dataset,
            seed=args.seed,
            output_dir=output_dir,
        )
        print("RBM training complete.")

    elif args.model == "DeepEBM":
        from iqp_mmd.training.ebm_trainer import train_ebm

        result = train_ebm(
            hyperparams=hyperparams["DeepEBM"][args.dataset],
            X_train=np.array(X_train),
            dataset_name=args.dataset,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"DeepEBM training complete. Final CD loss: {result['loss_history'][-1]:.6f}")

    elif args.model == "DeepGraphEBM":
        import networkx as nx
        from iqp_mmd.training.ebm_trainer import train_graph_ebm

        if not args.graph_path:
            parser.error("--graph-path is required for DeepGraphEBM")
        G = nx.read_adjlist(args.graph_path)

        result = train_graph_ebm(
            hyperparams=hyperparams["DeepGraphEBM"][args.dataset],
            X_train=np.array(X_train),
            dataset_name=args.dataset,
            graph=G,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"DeepGraphEBM training complete. Final CD loss: {result['loss_history'][-1]:.6f}")


if __name__ == "__main__":
    main()
