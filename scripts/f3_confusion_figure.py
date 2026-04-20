"""Render the F3 validation confusion matrix + init-axis breakdown as one PNG.

Usage (from repo root):
    python scripts/f3_confusion_figure.py
    python scripts/f3_confusion_figure.py --results path/to/other.jsonl --out path/to/other.png

Produces a 2-panel figure: left = confusion matrix heatmap, right = stacked
bar of PlateauAbsent/PlateauObserved by init scheme. Slide-ready at 16:9.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results/forge_plateau_diverse/results.jsonl",
        help="path to F3 plateau_agreement results.jsonl",
    )
    parser.add_argument(
        "--out",
        default="results/forge_plateau_diverse/figures/f3_confusion.png",
        help="output PNG path",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(rows)} rows from {results_path}")

    tp = sum(
        1
        for r in rows
        if r["plateau_observed"] == "PlateauObserved" and r["structurally_predicted"] is True
    )
    tn = sum(
        1
        for r in rows
        if r["plateau_observed"] == "PlateauAbsent" and r["structurally_predicted"] is False
    )
    fp = sum(
        1
        for r in rows
        if r["plateau_observed"] == "PlateauAbsent" and r["structurally_predicted"] is True
    )
    fn = sum(
        1
        for r in rows
        if r["plateau_observed"] == "PlateauObserved" and r["structurally_predicted"] is False
    )
    unknown = sum(1 for r in rows if r["structurally_predicted"] is None)

    classified = tp + tn + fp + fn
    accuracy = (tp + tn) / classified if classified else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    print(f"Confusion: TP={tp} TN={tn} FP={fp} FN={fn} unknown={unknown}")
    print(f"Accuracy={accuracy:.1%} Recall={recall:.1%} Precision={precision:.1%}")

    # Init-axis breakdown: PlateauAbsent vs PlateauObserved counts per init.
    by_init: dict[str, dict[str, int]] = {}
    for r in rows:
        init = r["init"]
        by_init.setdefault(init, {"PlateauAbsent": 0, "PlateauObserved": 0})
        by_init[init][r["plateau_observed"]] += 1

    # Keep a stable order: uniform first, small_angle second, then the rest.
    init_order = [k for k in ("uniform", "small_angle") if k in by_init]
    init_order += [k for k in by_init if k not in init_order]

    fig, (ax_cm, ax_init) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Left panel: confusion-matrix heatmap.
    cm = np.array([[tp, fn], [fp, tn]])
    ax_cm.imshow(cm, cmap="Reds")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_xticklabels(["predicted\nplateau", "predicted\nno plateau"])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(["observed\nplateau", "observed\nno plateau"])
    cm_max = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            ax_cm.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=20,
                color="white" if value > cm_max / 2 else "black",
            )
    ax_cm.set_title("Confusion matrix\n$bounded\\_degree[3] \\wedge high\\_overlap[2]$")

    # Right panel: stacked bar by init scheme.
    absent_counts = [by_init[k]["PlateauAbsent"] for k in init_order]
    observed_counts = [by_init[k]["PlateauObserved"] for k in init_order]
    ax_init.bar(
        init_order,
        absent_counts,
        label="PlateauAbsent (AC passes)",
        color="#4c72b0",
    )
    ax_init.bar(
        init_order,
        observed_counts,
        bottom=absent_counts,
        label="PlateauObserved (AC fails)",
        color="#c44e52",
    )
    # Annotate each stack segment (only when non-zero) instead of a cumulative
    # label on top — avoids legend overlap and is more informative.
    for idx in range(len(init_order)):
        absent = absent_counts[idx]
        observed = observed_counts[idx]
        if absent > 0:
            ax_init.text(idx, absent / 2, str(absent), ha="center", va="center",
                         fontsize=11, color="white")
        if observed > 0:
            ax_init.text(idx, absent + observed / 2, str(observed), ha="center",
                         va="center", fontsize=11, color="white")
    ax_init.set_ylabel("rows")
    ax_init.set_title("Plateau observations by init scheme")
    ax_init.set_ylim(0, max(a + o for a, o in zip(absent_counts, observed_counts)) * 1.15)
    ax_init.legend(loc="upper center", fontsize=9, ncol=2, frameon=True)

    fig.suptitle(
        f"F3: Structural theory vs AC-observed plateaus "
        f"(n in {{4, 6, 8}}, {len(rows)} rows)",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
