#!/usr/bin/env python3
"""Create the polished primary ancestry-specific LD-reduced summary figure."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LD_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("borzoi", "alphagenome", "consensus")
METHOD_LABELS = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome", "consensus": "Consensus"}
COLORS = {"borzoi": "#3366A3", "alphagenome": "#D97706", "consensus": "#198754"}
ANCESTRIES = ("AA", "CH", "NHW")


def primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["panel"].eq("general")
        & frame["deployment"].eq("native")
        & np.isclose(frame["clump_threshold_r2"].astype(float), 0.2)
        & frame["rank_view"].eq("global")
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yields", type=Path, default=LD_ROOT / "results/native_ld_yields.tsv")
    parser.add_argument("--distributions", type=Path, default=LD_ROOT / "results/native_ld_top_vs_rest.tsv")
    parser.add_argument("--output", type=Path, default=LD_ROOT / "figures/native_ld_primary_summary.png")
    parser.add_argument("--summary", type=Path, default=LD_ROOT / "qc/native_ld_figure_summary.json")
    args = parser.parse_args()

    yields = primary(pd.read_csv(args.yields, sep="\t"))
    distributions = primary(pd.read_csv(args.distributions, sep="\t"))
    if yields.empty or distributions.empty:
        raise RuntimeError("Primary figure inputs are empty")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.dpi": 150,
    })
    figure = plt.figure(figsize=(13.2, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.1])
    top_axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    bottom_axes = [figure.add_subplot(grid[1, column]) for column in range(3)]

    directional = yields.loc[yields["endpoint"].eq("index_directionally_supported")]
    for axis, ancestry in zip(top_axes, ANCESTRIES, strict=True):
        ancestry_data = directional.loc[directional["ancestry"].eq(ancestry)]
        for method in METHODS:
            line = ancestry_data.loc[ancestry_data["method"].eq(method)].sort_values("cutoff_percent")
            axis.plot(
                line["cutoff_percent"], 100 * line["selected_yield"],
                marker="o", markersize=4, linewidth=1.8,
                color=COLORS[method], label=METHOD_LABELS[method],
            )
        axis.set_title(ancestry)
        axis.set_xlabel("Top model-score cutoff (%)")
        axis.set_ylabel("Directionally supported (%)" if ancestry == "AA" else "")
        axis.set_xscale("log")
        axis.set_xlim(0.45, 11)
        axis.set_xticks([0.5, 1, 2, 5, 10])
        axis.set_xticklabels(["0.5", "1", "2", "5", "10"])
        axis.minorticks_off()
        axis.grid(axis="y", alpha=0.25)
    top_axes[0].text(-0.17, 1.10, "A", transform=top_axes[0].transAxes, fontsize=13, fontweight="bold")
    handles, labels = top_axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    axis = bottom_axes[0]
    cutoff = directional.loc[np.isclose(directional["cutoff_percent"], 1.0)]
    xlabels = [f"{ancestry}\n{METHOD_LABELS[method]}" for ancestry in ANCESTRIES for method in METHODS]
    x = np.arange(len(xlabels))
    exact_values, regional_values, no_values = [], [], []
    for ancestry in ANCESTRIES:
        for method in METHODS:
            selector = (
                yields["ancestry"].eq(ancestry) & yields["method"].eq(method)
                & np.isclose(yields["cutoff_percent"], 1.0)
            )
            rows = yields.loc[selector].set_index("endpoint")
            exact_values.append(100 * rows.loc["index_stringent_association", "selected_yield"])
            regional_values.append(100 * rows.loc["regional_only_supported", "selected_yield"])
            no_values.append(100 * rows.loc["no_detected_regional_support", "selected_yield"])
    axis.bar(x, exact_values, color="#2A6F97", label="Exact lead pair")
    axis.bar(x, regional_values, bottom=exact_values, color="#74C0A8", label="Regional-only")
    axis.bar(x, no_values, bottom=np.asarray(exact_values) + np.asarray(regional_values), color="#D9DEE3", label="No detected support")
    axis.set_xticks(x, xlabels, rotation=55, ha="right")
    axis.set_ylabel("Top-1% lead pairs (%)")
    axis.set_title("Lead-pair and close-proxy support")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.text(-0.17, 1.08, "B", transform=axis.transAxes, fontsize=13, fontweight="bold")

    axis = bottom_axes[1]
    auc = distributions.loc[
        distributions["evidence_type"].eq("association")
        & np.isclose(distributions["cutoff_percent"], 1.0)
    ]
    offsets = {"borzoi": -0.20, "alphagenome": 0.0, "consensus": 0.20}
    ancestry_x = np.arange(3)
    for method in METHODS:
        values = [
            auc.loc[auc["ancestry"].eq(ancestry) & auc["method"].eq(method), "auc_probability_top_has_smaller_p"].iloc[0]
            for ancestry in ANCESTRIES
        ]
        axis.scatter(ancestry_x + offsets[method], values, s=38, color=COLORS[method], label=METHOD_LABELS[method], zorder=3)
    axis.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axis.set_xticks(ancestry_x, ANCESTRIES)
    axis.set_ylabel("Association probability-of-superiority")
    axis.set_title("Top-1% regular-eQTL separation")
    lower = min(0.49, float(auc["auc_probability_top_has_smaller_p"].min()) - 0.01)
    upper = max(0.55, float(auc["auc_probability_top_has_smaller_p"].max()) + 0.01)
    axis.set_ylim(lower, upper)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.text(-0.17, 1.08, "C", transform=axis.transAxes, fontsize=13, fontweight="bold")

    axis = bottom_axes[2]
    lift = cutoff.copy()
    for method in METHODS:
        subset = lift.loc[lift["method"].eq(method)].set_index("ancestry").loc[list(ANCESTRIES)]
        values = subset["lift_over_background"].to_numpy(float)
        low = subset["gene_lift_over_background_ci_low"].to_numpy(float)
        high = subset["gene_lift_over_background_ci_high"].to_numpy(float)
        axis.errorbar(
            ancestry_x + offsets[method], values,
            # Percentile bootstrap intervals need not mathematically contain
            # the original point estimate in every finite sample. Matplotlib
            # requires nonnegative error lengths, so clip only the displayed
            # whisker length while retaining the exact interval in tables.
            yerr=np.vstack([
                np.maximum(0.0, values - low),
                np.maximum(0.0, high - values),
            ]),
            fmt="o", markersize=5, capsize=2.5, linewidth=1.2,
            color=COLORS[method], label=METHOD_LABELS[method],
        )
    axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axis.set_xticks(ancestry_x, ANCESTRIES)
    axis.set_ylabel("Directional-support lift over background")
    axis.set_title("Top-1% practical enrichment")
    axis.grid(axis="y", alpha=0.25)
    axis.text(-0.17, 1.08, "D", transform=axis.transAxes, fontsize=13, fontweight="bold")

    figure.suptitle("Ancestry-specific model-first prioritization after direct LD clumping (r² ≥ 0.2)", fontsize=13, fontweight="bold", y=1.06)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "figure": str(args.output),
        "pdf": str(args.output.with_suffix(".pdf")),
        "panels": {
            "A": "Global directional-support yield curves for the ancestry-specific general benchmark",
            "B": "Exact lead-pair, regional-only, and no-detected-support composition in the global top 1%",
            "C": "Weighted probability that a global top-1% lead pair has a smaller regular-eQTL p value than a background lead pair",
            "D": "Global top-1% directional-support lift over the full LD-reduced background with gene-bootstrap 95% intervals",
        },
        "ready": args.output.is_file() and args.output.with_suffix(".pdf").is_file(),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
