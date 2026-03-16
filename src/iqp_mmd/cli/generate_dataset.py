"""CLI entry point for generating or downloading datasets."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate or download datasets for IQP-MMD experiments.")
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "ising_lattice", "ising_network", "blobs", "mnist", "dwave", "genomic",
    ])
    parser.add_argument("--output-dir", type=str, required=True)

    # Ising lattice options
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)

    # Ising network options
    parser.add_argument("--n-nodes", type=int, default=10)
    parser.add_argument("--connectivity", type=int, default=2)

    # Blobs options
    parser.add_argument("--n-spins", type=int, default=16)
    parser.add_argument("--n-blobs", type=int, default=20)

    # Genomic options
    parser.add_argument("--variant", type=str, default="805", choices=["805", "10k"])

    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dataset == "ising_lattice":
        from iqp_mmd.datasets.ising import generate_ising_lattice

        ds = generate_ising_lattice(
            width=args.width,
            temperature=args.temperature,
            num_train=args.num_train,
            num_test=args.num_test,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"Generated Ising lattice: {ds.train_samples.shape[0]} train, {ds.test_samples.shape[0]} test")

    elif args.dataset == "ising_network":
        from iqp_mmd.datasets.ising import generate_ising_network

        ds = generate_ising_network(
            n_nodes=args.n_nodes,
            connectivity=args.connectivity,
            temperature=args.temperature,
            num_train=args.num_train,
            num_test=args.num_test,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"Generated Ising network: {ds.train_samples.shape[0]} train, {ds.test_samples.shape[0]} test")

    elif args.dataset == "blobs":
        from iqp_mmd.datasets.blobs import generate_spin_blobs

        ds = generate_spin_blobs(
            n_spins=args.n_spins,
            n_blobs=args.n_blobs,
            num_train=args.num_train,
            num_test=args.num_test,
            seed=args.seed,
            output_dir=output_dir,
        )
        print(f"Generated spin blobs: {ds.train_samples.shape[0]} train, {ds.test_samples.shape[0]} test")

    elif args.dataset == "mnist":
        from iqp_mmd.datasets.mnist import download_mnist

        ds = download_mnist(output_dir=output_dir)
        print(f"Downloaded MNIST: {ds.train_samples.shape[0]} train, {ds.test_samples.shape[0]} test")

    elif args.dataset == "dwave":
        from iqp_mmd.datasets.dwave import download_dwave

        ds = download_dwave(output_dir=output_dir)
        print(f"Downloaded D-Wave: {ds.train_samples.shape[0]} train, {ds.test_samples.shape[0]} test")

    elif args.dataset == "genomic":
        from iqp_mmd.datasets.genomic import download_genomic

        ds = download_genomic(variant=args.variant, output_dir=output_dir)
        print(f"Downloaded genomic ({args.variant}): {ds.train_samples.shape[0]} train, {ds.test_samples.shape[0]} test")


if __name__ == "__main__":
    main()
