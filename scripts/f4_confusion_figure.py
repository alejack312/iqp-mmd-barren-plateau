"""Render the F4 validation confusion matrix + (family, init, n) breakdown.

Usage (from repo root):
    python scripts/f4_confusion_figure.py
    python scripts/f4_confusion_figure.py --results path/to/other.jsonl --out path/to/other.png

Produces a 2-panel figure (confusion matrix + init stacked bar) identical in
layout to the F3 script, plus a CSV table broken down by (family, init, n)
that isolates whether the F4 predicate's wins come from the init axis alone
or from the structure clause pulling its weight.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results/forge_f4_manifests/results.jsonl",
        help="path to F4 plateau_agreement results.jsonl",
    )
    parser.add_argument(
        "--out",
        default="results/forge_f4_manifests/figures/f4_confusion.png",
        help="output PNG path for the 2-panel figure",
    )
    parser.add_argument(
        "--breakdown-csv",
        default="results/forge_f4_manifests/breakdown_by_init_n_family.csv",
        help="output CSV path for the (family, init, n) breakdown",
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

    # Init-axis breakdown (same as F3): PlateauAbsent vs PlateauObserved counts per init.
    by_init: dict[str, dict[str, int]] = {}
    for r in rows:
        init = r["init"]
        by_init.setdefault(init, {"PlateauAbsent": 0, "PlateauObserved": 0})
        by_init[init][r["plateau_observed"]] += 1

    init_order = [k for k in ("uniform", "small_angle") if k in by_init]
    init_order += [k for k in by_init if k not in init_order]

    # (family, init, n) agreement breakdown — the table that tells us whether
    # the structure clause is pulling its weight beyond the init axis alone.
    breakdown: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"agree": 0, "disagree": 0, "unknown": 0, "error": 0, "timeout": 0, "skipped_no_racket": 0}
    )
    for r in rows:
        key = (str(r["family"]), str(r["init"]), int(r["n"]))
        status = str(r["agreement"])
        breakdown[key].setdefault(status, 0)
        breakdown[key][status] += 1

    breakdown_path = Path(args.breakdown_csv)
    breakdown_path.parent.mkdir(parents=True, exist_ok=True)
    with breakdown_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "family",
                "init",
                "n",
                "total",
                "agree",
                "disagree",
                "unknown",
                "error",
                "timeout",
                "skipped_no_racket",
                "agreement_rate",
            ]
        )
        for (family, init, n) in sorted(breakdown.keys()):
            counts = breakdown[(family, init, n)]
            total = sum(counts.values())
            agree = counts.get("agree", 0)
            agreement_rate = f"{agree / total:.3f}" if total else ""
            writer.writerow(
                [
                    family,
                    init,
                    n,
                    total,
                    agree,
                    counts.get("disagree", 0),
                    counts.get("unknown", 0),
                    counts.get("error", 0),
                    counts.get("timeout", 0),
                    counts.get("skipped_no_racket", 0),
                    agreement_rate,
                ]
            )
    print(f"Saved breakdown: {breakdown_path}")

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
    ax_cm.set_title("Confusion matrix\nplateau_manifests (joint Init, n, structure)")

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
    for idx in range(len(init_order)):
        absent = absent_counts[idx]
        observed = observed_counts[idx]
        if absent > 0:
            ax_init.text(
                idx, absent / 2, str(absent), ha="center", va="center", fontsize=11, color="white"
            )
        if observed > 0:
            ax_init.text(
                idx,
                absent + observed / 2,
                str(observed),
                ha="center",
                va="center",
                fontsize=11,
                color="white",
            )
    ax_init.set_ylabel("rows")
    ax_init.set_title("Plateau observations by init scheme")
    ax_init.set_ylim(0, max(a + o for a, o in zip(absent_counts, observed_counts)) * 1.15)
    ax_init.legend(loc="upper center", fontsize=9, ncol=2, frameon=True)

    fig.suptitle(
        f"F4: Joint (init, n, structure) predicate vs AC-observed plateaus "
        f"(n in {{4, 6, 8}}, {len(rows)} rows)",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
