"""
Regenerate the manuscript Figure 2 AUROC boxplot from the saved 100-bootstrap TSV.

Two rows x three columns:
(A) high-PIP positives vs low-PIP (<0.01, in no credible set) comparison variants
(B) high-PIP positives vs intermediate-PIP (credible-set member) comparison variants

This script uses the stored bootstrap AUROC values directly
(susie_auroc_bootstrap_results_manuscript.tsv in $S2F_RESULTS).
"""

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

RESULTS = Path(os.environ["S2F_RESULTS"])
TSV = RESULTS / "susie_auroc_bootstrap_results_manuscript.tsv"
OUT = Path(os.environ.get("S2F_FIGURE_OUTPUT", str(RESULTS / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

TOOLS = ["borzoi", "alphagenome"]
ANCESTRIES = ["AA", "CH", "NHW"]
ANCLABELS = {"AA": "African American", "CH": "Caribbean Hispanic", "NHW": "Non-Hispanic White"}
TOOL_LABELS = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome"}
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ["standard", "rest"]
MODE_LABELS = {"standard": "vs. low PIP (<0.01)", "rest": "vs. intermediate PIP"}
PANEL_LBL = {"standard": "A", "rest": "B"}
MODEL_COLORS = {"borzoi": "#E64B35", "alphagenome": "#4DBBD5"}

# Slightly larger typography for the manuscript figure.
plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
})


def parse_auc_values(value_string: str) -> np.ndarray:
    return np.array([float(x) for x in str(value_string).split(",") if x != ""], dtype=float)


df = pd.read_csv(TSV, sep="\t")
results = {}
for _, row in df.iterrows():
    key = f"{row['tool']}_{row['ancestry']}_pip{row['pip_threshold']}_{row['mode']}"
    results[key] = parse_auc_values(row["auc_values"])

fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharey=True)
plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, hspace=0.38, wspace=0.20)

base_positions = np.arange(len(THRESHOLDS)) + 1
offset = 0.18
width = 0.30

for row_idx, mode in enumerate(MODES):
    for col_idx, anc in enumerate(ANCESTRIES):
        ax = axes[row_idx, col_idx]

        for tool in TOOLS:
            tool_positions = base_positions + (-offset if tool == "borzoi" else offset)
            box_data = []
            valid_positions = []
            for pos, threshold in zip(tool_positions, THRESHOLDS):
                key = f"{tool}_{anc}_pip{threshold}_{mode}"
                if key in results:
                    box_data.append(results[key])
                    valid_positions.append(pos)

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    positions=valid_positions,
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="#555555", lw=1.1),
                    capprops=dict(color="#555555", lw=1.1),
                    boxprops=dict(edgecolor="#555555", lw=1.1),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(MODEL_COLORS[tool])
                    patch.set_alpha(0.80)

        ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.6)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlim(0.4, len(THRESHOLDS) + 0.6)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([f"≥{t}" for t in THRESHOLDS])
        ax.tick_params(axis="both", labelsize=11)
        ax.set_xlabel("PIP Threshold", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("AUROC", fontsize=12)
        if row_idx == 0:
            ax.set_title(ANCLABELS[anc], fontsize=13, fontweight="bold", pad=8)

        if col_idx == 0:
            ax.text(
                -0.26,
                1.10,
                f"({PANEL_LBL[mode]})",
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="bottom",
                ha="left",
                clip_on=False,
            )
            ax.text(
                -0.10,
                1.10,
                MODE_LABELS[mode],
                transform=ax.transAxes,
                fontsize=12,
                fontstyle="italic",
                va="bottom",
                ha="left",
                color="#444444",
                clip_on=False,
            )

legend_handles = [
    Patch(facecolor=MODEL_COLORS[tool], edgecolor="#555555", label=TOOL_LABELS[tool], alpha=0.80)
    for tool in TOOLS
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=12)

for suffix in [".pdf", ".png"]:
    out = OUT / f"figure2_distance_matched_auroc_boxplot{suffix}"
    fig.savefig(out, bbox_inches="tight", dpi=250)
    print(f"[OK] Saved {out}")

plt.close(fig)
