#!/usr/bin/env python3
"""Plot complete top-1% lead-pair versus background eQTL evidence distributions."""

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
LABEL = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome", "consensus": "Consensus"}
COLOR = {"borzoi": "#3366A3", "alphagenome": "#D97706", "consensus": "#198754"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecdf", type=Path, default=LD_ROOT / "results/native_ld_primary_ecdf.tsv.gz")
    parser.add_argument("--output", type=Path, default=LD_ROOT / "figures/native_ld_top_vs_rest_distributions.png")
    args = parser.parse_args()
    data = pd.read_csv(args.ecdf, sep="\t")
    data = data.loc[
        data["deployment"].eq("native") & data["rank_view"].eq("global")
    ].copy()
    if data.empty:
        raise RuntimeError("Native global ECDF data are empty")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, constrained_layout=True)
    for row_index, evidence_type in enumerate(("association", "directional")):
        for column_index, ancestry in enumerate(("AA", "CH", "NHW")):
            axis = axes[row_index, column_index]
            subset = data.loc[
                data["evidence_type"].eq(evidence_type) & data["ancestry"].eq(ancestry)
            ]
            for method in METHODS:
                for group, linestyle in (("top_1_percent", "-"), ("rest_99_percent", "--")):
                    line = subset.loc[
                        subset["method"].eq(method) & subset["group"].eq(group)
                    ].sort_values("minus_log10_p_threshold")
                    axis.plot(
                        line["minus_log10_p_threshold"], line["weighted_cumulative_fraction"],
                        color=COLOR[method], linestyle=linestyle, linewidth=1.6,
                        label=f"{LABEL[method]} {'top 1%' if group == 'top_1_percent' else 'background'}",
                    )
            axis.axvline(-np.log10(2.02e-5), color="#777777", linestyle=":", linewidth=1)
            axis.set_xticks(np.arange(0, 9, 1))
            axis.tick_params(axis="x", which="both", labelbottom=True)
            axis.set_title(f"{ancestry} — {'two-sided regular-eQTL p' if evidence_type == 'association' else 'one-sided directional p'}")
            axis.set_xlabel("Evidence threshold, −log₁₀(p)")
            axis.set_ylabel("Weighted fraction at or below p" if column_index == 0 else "")
            axis.grid(alpha=0.2)
            if row_index == 0 and column_index == 0:
                axis.text(-0.16, 1.08, "A", transform=axis.transAxes, fontsize=13, fontweight="bold")
            if row_index == 1 and column_index == 0:
                axis.text(-0.16, 1.08, "B", transform=axis.transAxes, fontsize=13, fontweight="bold")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
    figure.suptitle("Top-1% lead pairs versus background eQTL evidence after ancestry-specific LD clumping", fontsize=13, fontweight="bold", y=1.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "figure": str(args.output), "pdf": str(args.output.with_suffix('.pdf')),
        "ready": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
