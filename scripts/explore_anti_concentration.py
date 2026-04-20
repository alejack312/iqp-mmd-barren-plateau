"""Hands-on anti-concentration exploration for the presentation.

Builds three small probability vectors over {0,1}^n:

  1. uniform        — the reference anti-concentrated distribution
  2. delta          — a fully concentrated distribution on one bitstring
  3. random-theta   — an n=3 complete_graph IQP model with random angles

For each case it:

  * prints every field of the check_anti_concentration dict,
  * writes a side-by-side bar chart of p(x) to results/exploration/,
  * writes a side-by-side bar chart of the beta_hat(alpha) diagnostic.

Run from the repo root:

    PYTHONPATH=src python scripts/explore_anti_concentration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iqp_bp.experiments import check_anti_concentration
from iqp_bp.iqp.model import IQPModel


N_QUBITS = 3
NUM_OUTCOMES = 2**N_QUBITS
OUTPUT_DIR = Path("results/exploration")


def bitstring_labels(n: int) -> list[str]:
    """Return labels '000', '001', ..., '111' matching the exact index order."""
    return [format(i, f"0{n}b") for i in range(2**n)]


def uniform_distribution(n: int) -> np.ndarray:
    """p(x) = 1/2^n — the anti-concentrated baseline."""
    return np.full(2**n, 1.0 / (2**n), dtype=np.float64)


def delta_distribution(n: int, outcome_index: int = 0) -> np.ndarray:
    """p(x*) = 1, p(x) = 0 for x != x* — the extreme concentrated distribution."""
    p = np.zeros(2**n, dtype=np.float64)
    p[outcome_index] = 1.0
    return p


def iqp_random_distribution(n: int, seed: int = 42) -> tuple[np.ndarray, IQPModel]:
    """Build a complete_graph IQP model with uniform random angles and compute p(x)."""
    rng = np.random.default_rng(seed)
    model = IQPModel.from_family(family="complete_graph", n=n, rng=rng)
    model.theta = rng.uniform(low=-np.pi, high=np.pi, size=model.m)
    p = model.probability_vector_exact()
    return p, model


def describe_result(name: str, result: dict) -> None:
    """Print the anti-concentration dict with a short mapping to paper definitions."""
    print(f"\n=== {name} ===")
    print(f"  n                         = {result['n']}")
    print(f"  num_outcomes              = {result['num_outcomes']}")
    print(f"  uniform_probability       = {result['uniform_probability']:.6f}   (= 2^-n)")
    print(
        f"  collision_probability     = {result['collision_probability']:.6f}   "
        "(= sum_x p(x)^2, the second moment)"
    )
    print(
        f"  scaled_second_moment      = {result['scaled_second_moment']:.6f}   "
        "(= 2^n * sum p(x)^2; uniform -> 1, delta -> 2^n)"
    )
    print(
        f"  passes_second_moment      = {result['passes_second_moment_threshold']}   "
        "(scaled_second_moment >= threshold)"
    )
    print(
        f"  max_probability_scaled    = {result['max_probability_scaled']:.6f}   "
        "(= 2^n * max p; uniform -> 1, delta -> 2^n)"
    )
    print(
        f"  effective_support         = {result['effective_support']:.6f}   "
        "(= 1 / collision_probability, participation ratio)"
    )
    print(
        f"  effective_support_ratio   = {result['effective_support_ratio']:.6f}   "
        "(fraction of 2^n 'really covered' by p)"
    )
    print(f"  primary_alpha             = {result['primary_alpha']}")
    print(
        f"  primary_beta_hat          = {result['primary_beta_hat']:.6f}   "
        "(= fraction of bitstrings with p(x) >= alpha * 2^-n)"
    )
    print(f"  passes_primary_threshold  = {result['passes_primary_threshold']}")
    print("  threshold_checks:")
    for row in result["threshold_checks"]:
        print(
            f"    alpha={row['alpha']:<4}  "
            f"threshold_p={row['threshold_probability']:.6f}  "
            f"beta_hat={row['beta_hat']:.6f}  "
            f"pass={row['passes_beta_threshold']}"
        )


def plot_probability_vectors(
    vectors: dict[str, np.ndarray],
    n: int,
    output_path: Path,
) -> None:
    """Side-by-side bar charts of p(x) for each scenario."""
    labels = bitstring_labels(n)
    x_positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, len(vectors), figsize=(5 * len(vectors), 4), sharey=True)
    if len(vectors) == 1:
        axes = [axes]

    uniform_level = 1.0 / (2**n)
    for ax, (name, p) in zip(axes, vectors.items(), strict=True):
        ax.bar(x_positions, p, color="tab:blue", edgecolor="black")
        ax.axhline(
            uniform_level,
            linestyle="--",
            color="tab:red",
            label=f"uniform = 2^-n = {uniform_level:.3f}",
        )
        ax.set_title(name)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("bitstring x")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("p(x)")

    fig.suptitle(f"Output distributions over {{0,1}}^{n}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_beta_hat_curves(results: dict[str, dict], output_path: Path) -> None:
    """beta_hat(alpha) curves for each scenario on one axis."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, result in results.items():
        alphas = [row["alpha"] for row in result["threshold_checks"]]
        beta_hats = [row["beta_hat"] for row in result["threshold_checks"]]
        ax.plot(alphas, beta_hats, marker="o", linewidth=2, label=name)

    beta_min = next(iter(results.values()))["beta_min"]
    ax.axhline(beta_min, linestyle="--", color="tab:red", label=f"beta_min = {beta_min}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta_hat(alpha)")
    ax.set_title("Threshold diagnostic: beta_hat(alpha) = fraction of x with p(x) >= alpha*2^-n")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the three probability vectors.
    p_uniform = uniform_distribution(N_QUBITS)
    p_delta = delta_distribution(N_QUBITS, outcome_index=0)
    p_iqp, iqp_model = iqp_random_distribution(N_QUBITS, seed=42)

    # Show the IQP model structure so the student can see what they built.
    print("IQP model for the 'random-theta' case:")
    print(f"  family = complete_graph, n = {iqp_model.n}, m = {iqp_model.m}")
    print(f"  generator matrix G (rows = generators, cols = qubits):\n{iqp_model.G}")
    print(f"  theta = {iqp_model.theta}")

    vectors = {
        "uniform": p_uniform,
        "delta": p_delta,
        "random-theta IQP": p_iqp,
    }

    # Run the checker on each case.
    results: dict[str, dict] = {}
    for name, p in vectors.items():
        results[name] = check_anti_concentration(
            p,
            alphas=(0.25, 0.5, 1.0, 2.0, 4.0),
            primary_alpha=1.0,
            beta_min=0.25,
            second_moment_threshold=1.0,
        )

    # Print diagnostics for each case.
    for name, result in results.items():
        describe_result(name, result)

    # Save plots and JSON.
    prob_plot_path = OUTPUT_DIR / f"probability_vectors_n{N_QUBITS}.png"
    beta_plot_path = OUTPUT_DIR / f"beta_hat_curves_n{N_QUBITS}.png"
    json_path = OUTPUT_DIR / f"anti_concentration_n{N_QUBITS}.json"

    plot_probability_vectors(vectors, N_QUBITS, prob_plot_path)
    plot_beta_hat_curves(results, beta_plot_path)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {name: result for name, result in results.items()},
            handle,
            indent=2,
            sort_keys=True,
        )

    print("\nArtifacts:")
    print(f"  probability bar chart : {prob_plot_path}")
    print(f"  beta_hat curves       : {beta_plot_path}")
    print(f"  full JSON summary     : {json_path}")


if __name__ == "__main__":
    main()
