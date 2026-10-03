#!/usr/bin/env python3
"""Render the MAF-stratified fine-mapping AUROC supplementary figure."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


RESULTS = Path(os.environ["S2F_RESULTS"])
OUT = Path(os.environ.get("S2F_FIGURE_OUTPUT", str(RESULTS / "figures")))
INPUT = RESULTS / "susie_auroc_maf_stratified_bootstrap.tsv.gz"

TOOLS = ["borzoi", "alphagenome"]
ANCESTRIES = ["AA", "CH", "NHW"]
ANCESTRY_LABELS = {
    "AA": "African American",
    "CH": "Caribbean Hispanic",
    "NHW": "Non-Hispanic White",
}
MAF_BINS = ["lt0.05", "0.05-0.2", "ge0.2"]
MAF_LABELS = {
    "lt0.05": "MAF < 0.05",
    "0.05-0.2": "0.05 ≤ MAF < 0.2",
    "ge0.2": "MAF ≥ 0.2",
}
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
COLORS = {"borzoi": "#E64B35", "alphagenome": "#4DBBD5"}
LABELS = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT, sep="\t")
    data["pip_threshold"] = pd.to_numeric(data["pip_threshold"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)
    plt.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.08, hspace=0.34, wspace=0.16)
    base = np.arange(len(THRESHOLDS)) + 1
    offset = 0.17

    for row, maf_bin in enumerate(MAF_BINS):
        for col, ancestry in enumerate(ANCESTRIES):
            ax = axes[row, col]
            subset = data[(data["maf_bin"] == maf_bin) & (data["ancestry"] == ancestry)]
            for tool in TOOLS:
                values = []
                positions = []
                for pos, threshold in zip(base, THRESHOLDS):
                    current = subset[
                        (subset["tool"] == tool)
                        & np.isclose(subset["pip_threshold"], threshold)
                    ]["auc"].dropna().to_numpy()
                    if len(current):
                        values.append(current)
                        positions.append(pos + (-offset if tool == "borzoi" else offset))
                if values:
                    artists = ax.boxplot(
                        values,
                        positions=positions,
                        widths=0.28,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                        medianprops={"color": "black", "lw": 1.2},
                        whiskerprops={"color": "#555555", "lw": 1.0},
                        capprops={"color": "#555555", "lw": 1.0},
                        boxprops={"edgecolor": "#555555", "lw": 1.0},
                    )
                    for patch in artists["boxes"]:
                        patch.set_facecolor(COLORS[tool])
                        patch.set_alpha(0.80)

            ax.axhline(0.5, color="#B2182B", linestyle="--", linewidth=1, alpha=0.65)
            ax.set_xlim(0.45, 5.55)
            ax.set_ylim(0.25, 1.0)
            ax.set_xticks(base)
            ax.set_xticklabels([f"≥{value}" for value in THRESHOLDS])
            ax.set_xlabel("PIP threshold")
            if col == 0:
                ax.set_ylabel(f"{MAF_LABELS[maf_bin]}\nAUROC")
            if row == 0:
                ax.set_title(ANCESTRY_LABELS[ancestry], fontweight="bold")
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)

    handles = [
        Patch(facecolor=COLORS[tool], edgecolor="#555555", alpha=0.8, label=LABELS[tool])
        for tool in TOOLS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    for suffix in ("pdf", "png"):
        path = OUT / f"figureS1_maf_stratified_auroc_boxplot.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=250)
        print(f"[OK] Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
