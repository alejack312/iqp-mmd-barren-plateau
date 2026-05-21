"""Headline figure for Phase 4 (Pauli Estimator Scale-Up — n=16 Validation).

Emits a 2-panel PNG that summarises the n=16 Parseval-MC validation result:

  Panel 1: scaled second moment (log-y) — exact vs estimate ±σ for 2D_ising and 8_blobs,
           with the uniform baseline (y=1) drawn as a red dashed line.
  Panel 2: per-order marginal mismatch (mean TV) across k in {1, 2, 4, 8} — grouped
           bars (Ising exact/estimate, Blobs exact/estimate); k=8 is exact-only
           (no paired estimator — validation JSON only ran k∈{1,2,4}).

Inputs (read-only):
  results/pauli_estimator_validation_n16.json
  results/iqp_mmd_ac_investigation/ising_n16_iters1000_seed666_summary_CORRECTED.json
  results/iqp_mmd_ac_investigation/spin_blobs_n16_iters1000_seed666_summary_CORRECTED.json

Output:
  results/pauli_scale/headline_scaled_second_moment.png  (dpi=150)

Run:
  C:/Python313/python.exe scripts/plot_pauli_scale_headline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_JSON = REPO_ROOT / "results" / "pauli_estimator_validation_n16.json"
ISING_CORRECTED = (
    REPO_ROOT
    / "results"
    / "iqp_mmd_ac_investigation"
    / "ising_n16_iters1000_seed666_summary_CORRECTED.json"
)
BLOBS_CORRECTED = (
    REPO_ROOT
    / "results"
    / "iqp_mmd_ac_investigation"
    / "spin_blobs_n16_iters1000_seed666_summary_CORRECTED.json"
)

OUT_DIR = REPO_ROOT / "results" / "pauli_scale"
OUT_PATH = OUT_DIR / "headline_scaled_second_moment.png"

TIER_MARK = {"pass": ("✓", "tab:green"),   # green check
             "warn": ("~",      "tab:orange"),   # orange tilde
             "fail": ("*",      "tab:red")}      # red asterisk


def _load_validation_criteria(path: Path) -> dict[tuple[str, str], dict]:
    """Return lookup {(dataset, criterion): criterion_dict}."""
    data = json.loads(path.read_text())
    return {(c["dataset"], c["criterion"]): c for c in data["criteria"]}


def _tier_colour(tier: str) -> str:
    return {"pass": "tab:blue", "warn": "tab:blue", "fail": "tab:red"}.get(tier, "tab:blue")


def panel1_scaled_ss(ax, crit: dict[tuple[str, str], dict]) -> None:
    """Panel 1: scaled second moment — exact vs estimate ±σ (log-y)."""
    datasets = [
        ("2D_ising n=16", "2D_ising"),
        ("8_blobs n=16",  "8_blobs"),
    ]

    x = np.arange(len(datasets))
    w = 0.35

    exact_vals: list[float] = []
    est_vals:   list[float] = []
    sigmas:     list[float] = []
    tiers:      list[str]   = []

    for _label, ds in datasets:
        c = crit[(ds, "val01_scaled_ss")]
        exact_vals.append(c["exact"])
        est_vals.append(c["estimate"])
        sigmas.append(c["sigma"])
        tiers.append(c["tier"])

    # Bars: exact (gray) on the left, estimate (blue/red by tier) on the right
    ax.bar(x - w / 2, exact_vals, w,
           color="tab:gray", label="Exact (ground truth)")

    # Estimate bars coloured per-dataset by tier
    est_colors = [_tier_colour(t) for t in tiers]
    ax.bar(x + w / 2, est_vals, w,
           color=est_colors, label="Estimate ± σ",
           yerr=sigmas, capsize=4, ecolor="black",
           error_kw={"elinewidth": 1.2})

    # Uniform baseline
    ax.axhline(1.0, ls="--", color="tab:red",
               label="Uniform baseline (y=1)", linewidth=1.0)

    # Tier annotations above estimate bars
    for xi, est, sig, tier in zip(x, est_vals, sigmas, tiers):
        mark, col = TIER_MARK.get(tier, ("?", "black"))
        ax.annotate(
            f"{mark} {tier}",
            xy=(xi + w / 2, est + sig),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, color=col, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in datasets])
    ax.set_yscale("log")
    ax.set_ylabel(r"$2^n \cdot \sum_x p(x)^2$ (log scale)")
    ax.set_title("Scaled Second Moment — n=16 Validation")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(loc="upper right", fontsize=8)

    # Give annotations room above the error bar tops
    ymax = max(e + s for e, s in zip(est_vals, sigmas))
    ymin = 0.5  # below uniform baseline for visual context
    ax.set_ylim(ymin, ymax * 3.0)


def panel2_marginal_mismatch(
    ax,
    crit: dict[tuple[str, str], dict],
    ising_summary: dict,
    blobs_summary: dict,
) -> None:
    """Panel 2: grouped bars of mean TV at k in {1,2,4,8}; k=8 exact-only."""
    ks = [1, 2, 4, 8]
    x = np.arange(len(ks), dtype=float)
    w = 0.18

    ising_exact = []
    ising_est   = []
    ising_sig   = []
    blobs_exact = []
    blobs_est   = []
    blobs_sig   = []

    for k in ks:
        # Exact values: always from CORRECTED summaries (single source of truth)
        ising_exact.append(
            ising_summary["marginal_mismatch_learned_vs_target"][str(k)]["mean_tv"])
        blobs_exact.append(
            blobs_summary["marginal_mismatch_learned_vs_target"][str(k)]["mean_tv"])

        # Estimates: present for k in {1,2,4} from validation JSON, absent for k=8
        if k in (1, 2, 4):
            ci = crit[("2D_ising", f"val02_mean_tv_k{k}")]
            cb = crit[("8_blobs",  f"val02_mean_tv_k{k}")]
            ising_est.append(ci["estimate"]); ising_sig.append(ci["sigma"])
            blobs_est.append(cb["estimate"]); blobs_sig.append(cb["sigma"])
        else:
            ising_est.append(np.nan); ising_sig.append(0.0)
            blobs_est.append(np.nan); blobs_sig.append(0.0)

    ising_exact = np.asarray(ising_exact)
    ising_est   = np.asarray(ising_est)
    ising_sig   = np.asarray(ising_sig)
    blobs_exact = np.asarray(blobs_exact)
    blobs_est   = np.asarray(blobs_est)
    blobs_sig   = np.asarray(blobs_sig)

    mask_paired = np.array([k != 8 for k in ks])  # True where estimate bars shown

    # Positions: Ising exact, Ising est, Blobs exact, Blobs est at offsets
    # [-1.5w, -0.5w, +0.5w, +1.5w] for paired k; for k=8 only Ising exact / Blobs exact
    # are drawn at [-0.5w, +0.5w].
    # Compute per-k positions.
    def _pos(base: float, offset: float) -> float:
        return base + offset * w

    # Paired-k group
    if mask_paired.any():
        xp = x[mask_paired]
        ax.bar(xp + (-1.5) * w, ising_exact[mask_paired], w,
               color="tab:blue", alpha=0.9, label="Ising exact")
        ax.bar(xp + (-0.5) * w, ising_est[mask_paired], w,
               color="tab:blue", alpha=0.6, hatch="///",
               yerr=ising_sig[mask_paired], capsize=3, ecolor="black",
               label="Ising estimate",
               error_kw={"elinewidth": 1.0})
        ax.bar(xp + (+0.5) * w, blobs_exact[mask_paired], w,
               color="tab:orange", alpha=0.9, label="Blobs exact")
        ax.bar(xp + (+1.5) * w, blobs_est[mask_paired], w,
               color="tab:orange", alpha=0.6, hatch="///",
               yerr=blobs_sig[mask_paired], capsize=3, ecolor="black",
               label="Blobs estimate",
               error_kw={"elinewidth": 1.0})

    # Exact-only k group (k=8)
    mask_exact_only = ~mask_paired
    if mask_exact_only.any():
        xe = x[mask_exact_only]
        ax.bar(xe + (-0.5) * w, ising_exact[mask_exact_only], w,
               color="tab:blue", alpha=0.9)
        ax.bar(xe + (+0.5) * w, blobs_exact[mask_exact_only], w,
               color="tab:orange", alpha=0.9)

        # Annotate exact-only bars
        for xi, yi_i, yi_b in zip(
            xe,
            ising_exact[mask_exact_only],
            blobs_exact[mask_exact_only],
        ):
            ax.annotate(
                "(exact only)",
                xy=(xi + (-0.5) * w, yi_i),
                xytext=(0, 8), textcoords="offset points",
                rotation=90, ha="center", va="bottom",
                fontsize=7, color="tab:blue",
            )
            ax.annotate(
                "(exact only)",
                xy=(xi + (+0.5) * w, yi_b),
                xytext=(0, 8), textcoords="offset points",
                rotation=90, ha="center", va="bottom",
                fontsize=7, color="tab:orange",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Marginal order k")
    ax.set_ylabel("Mean TV distance")
    ax.set_title("Marginal Mismatch (Mean TV) — n=16")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8)

    # Give headroom for rotated (exact only) annotations
    ymax = max(
        float(np.nanmax(ising_exact)),
        float(np.nanmax(blobs_exact)),
        float(np.nanmax(np.where(np.isfinite(ising_est), ising_est, 0))),
        float(np.nanmax(np.where(np.isfinite(blobs_est), blobs_est, 0))),
    )
    ax.set_ylim(0, ymax * 1.35)


def main() -> None:
    crit = _load_validation_criteria(VALIDATION_JSON)
    ising_summary = json.loads(ISING_CORRECTED.read_text())
    blobs_summary = json.loads(BLOBS_CORRECTED.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel1_scaled_ss(axes[0], crit)
    panel2_marginal_mismatch(axes[1], crit, ising_summary, blobs_summary)

    fig.suptitle(
        "Pauli Estimator Scale-Up — n=16 Validation Summary",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
