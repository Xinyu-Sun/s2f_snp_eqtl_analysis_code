#!/usr/bin/env python3
"""
Create Figure 2: inter-model convergence between Borzoi and AlphaGenome
predictions across ancestry groups and TSS-distance strata.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


ANCESTRIES = ["AA", "CH", "NHW"]
TSS_BINS = ["0-3kb", "3-12kb", "12-35kb", ">35kb"]
DATASET_ORDER = ["Nominal eQTLs", "Fine-mapped (PIP>=0.9)", "Fine-mapped (PIP >= 0.9)"]
DATASET_LABELS = {
    "Nominal eQTLs": "Nominal eQTLs",
    "Fine-mapped (PIP>=0.9)": "Fine-mapped (PIP >= 0.9)",
    "Fine-mapped (PIP >= 0.9)": "Fine-mapped (PIP >= 0.9)",
}


def default_input() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data_availability"
        / "figure2_model_convergence"
        / "model_convergence_results.tsv"
    )


def rounded_limits(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("Cannot determine colorbar limits from empty data")
    lower_tenths = math.floor(values.min() * 10)
    upper_tenths = math.ceil(values.max() * 10)
    if lower_tenths == upper_tenths:
        upper_tenths += 1
    return lower_tenths / 10, upper_tenths / 10


def rounded_ticks(vmin: float, vmax: float) -> list[float]:
    vmin = round(vmin, 1)
    vmax = round(vmax, 1)
    step = 0.1 if (vmax - vmin) <= 0.600001 else 0.2
    ticks = [vmin]
    next_tick = math.ceil((vmin + 1e-9) / step) * step
    while next_tick < vmax - 1e-9:
        ticks.append(next_tick)
        next_tick += step
    ticks.append(vmax)
    return sorted(set(round(tick, 1) for tick in ticks))


def annotate_missing_cells(ax: plt.Axes, data: pd.DataFrame, label: str = "n<10") -> None:
    for row_idx, row_name in enumerate(data.index):
        for col_idx, col_name in enumerate(data.columns):
            if pd.isna(data.loc[row_name, col_name]):
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#555555",
                )


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {
        "dataset",
        "ancestry",
        "tss_distance_bin",
        "spearman_r",
        "concordance",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_list}")

    df["dataset"] = df["dataset"].replace({"Fine-mapped (PIP≥0.9)": "Fine-mapped (PIP>=0.9)"})
    return df[
        df["ancestry"].isin(ANCESTRIES)
        & df["tss_distance_bin"].isin(TSS_BINS)
    ].copy()


def available_datasets(df: pd.DataFrame) -> list[str]:
    present = set(df["dataset"].unique())
    datasets = []
    for dataset in DATASET_ORDER:
        if dataset in present and dataset not in datasets:
            datasets.append(dataset)
    if not datasets:
        raise ValueError("No recognized datasets found in model convergence table")
    return datasets


def pivot_metric(df: pd.DataFrame, dataset: str, value: str) -> pd.DataFrame:
    subset = df[df["dataset"] == dataset]
    pivot = subset.pivot(index="tss_distance_bin", columns="ancestry", values=value)
    return pivot.reindex(index=TSS_BINS, columns=ANCESTRIES)


def make_figure(df: pd.DataFrame, output_dir: Path, output_prefix: str) -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    datasets = available_datasets(df)
    corr_vmin, corr_vmax = rounded_limits(df["spearman_r"])
    conc_vmin, conc_vmax = rounded_limits(df["concordance"])
    corr_ticks = rounded_ticks(corr_vmin, corr_vmax)
    conc_ticks = rounded_ticks(conc_vmin, conc_vmax)

    heatmap_cmap = plt.get_cmap("RdYlGn").copy()
    heatmap_cmap.set_bad("#e6e6e6")

    fig = plt.figure(figsize=(9.0 if len(datasets) == 2 else 4.8, 8.0))
    main_gs = plt.GridSpec(
        2,
        len(datasets) + 1,
        figure=fig,
        width_ratios=[1] * len(datasets) + [0.035],
        hspace=0.20,
        wspace=0.022,
        left=0.10,
        right=0.95,
        top=0.90,
        bottom=0.10,
    )

    corr_cbar_ax = fig.add_subplot(main_gs[0, -1])
    conc_cbar_ax = fig.add_subplot(main_gs[1, -1])
    top_axes = []
    column_axes = {}

    for col_idx, dataset in enumerate(datasets):
        ax_top = fig.add_subplot(main_gs[0, col_idx])
        pivot_corr = pivot_metric(df, dataset, "spearman_r")
        ax_top.set_facecolor("#e6e6e6")
        sns.heatmap(
            pivot_corr,
            annot=True,
            fmt=".3f",
            cmap=heatmap_cmap,
            center=0.5,
            ax=ax_top,
            cbar=(col_idx == len(datasets) - 1),
            cbar_ax=corr_cbar_ax if col_idx == len(datasets) - 1 else None,
            cbar_kws={"label": "Spearman r", "ticks": corr_ticks},
            vmin=corr_vmin,
            vmax=corr_vmax,
            linewidths=0.5,
            linecolor="white",
            square=True,
            annot_kws={"size": 11},
        )
        annotate_missing_cells(ax_top, pivot_corr)
        ax_top.set_title("Correlation\n(Borzoi vs AlphaGenome)", fontsize=11, fontweight="bold", pad=6)
        ax_top.set_xlabel("")
        ax_top.set_ylabel("TSS Distance" if col_idx == 0 else "")
        ax_top.set_xticklabels(ANCESTRIES, rotation=0)
        ax_top.set_yticklabels(ax_top.get_yticklabels(), rotation=0)
        ax_top.tick_params(axis="both", labelsize=11)
        if col_idx != 0:
            ax_top.tick_params(axis="y", labelleft=False, length=0)
        ax_top.grid(False)

        ax_bottom = fig.add_subplot(main_gs[1, col_idx])
        pivot_conc = pivot_metric(df, dataset, "concordance")
        ax_bottom.set_facecolor("#e6e6e6")
        sns.heatmap(
            pivot_conc,
            annot=True,
            fmt=".3f",
            cmap=heatmap_cmap,
            center=0.5,
            ax=ax_bottom,
            cbar=(col_idx == len(datasets) - 1),
            cbar_ax=conc_cbar_ax if col_idx == len(datasets) - 1 else None,
            cbar_kws={"label": "Concordance", "ticks": conc_ticks},
            vmin=conc_vmin,
            vmax=conc_vmax,
            linewidths=0.5,
            linecolor="white",
            square=True,
            annot_kws={"size": 11},
        )
        annotate_missing_cells(ax_bottom, pivot_conc)
        ax_bottom.set_title("Direction Concordance", fontsize=11, fontweight="bold", pad=6)
        ax_bottom.set_xlabel("")
        ax_bottom.set_ylabel("TSS Distance" if col_idx == 0 else "")
        ax_bottom.set_xticklabels(ANCESTRIES, rotation=0)
        ax_bottom.set_yticklabels(ax_bottom.get_yticklabels(), rotation=0)
        ax_bottom.tick_params(axis="both", labelsize=11)
        if col_idx != 0:
            ax_bottom.tick_params(axis="y", labelleft=False, length=0)
        ax_bottom.grid(False)

        top_axes.append((col_idx, ax_top, dataset))
        column_axes[col_idx] = (ax_top, ax_bottom)

    fig.canvas.draw()
    if len(datasets) == 2:
        left_top, left_bottom = column_axes[0]
        right_top, right_bottom = column_axes[1]
        current_gap = right_top.get_position().x0 - left_top.get_position().x1
        target_gap = 0.028
        if current_gap > target_gap:
            shift = current_gap - target_gap
            for ax in [right_top, right_bottom]:
                pos = ax.get_position()
                ax.set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])

    fig.canvas.draw()
    for col_idx, ax_top, dataset in top_axes:
        pos = ax_top.get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.068,
            f"({chr(97 + col_idx)}) {DATASET_LABELS.get(dataset, dataset)}",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    last_col = len(datasets) - 1
    for cax, ref_ax in [
        (corr_cbar_ax, top_axes[-1][1]),
        (conc_cbar_ax, column_axes[last_col][1]),
    ]:
        ref_pos = ref_ax.get_position()
        cax.set_position([ref_pos.x1 + 0.004, ref_pos.y0, 0.012, ref_pos.height])

    for cax, ticks in [(corr_cbar_ax, corr_ticks), (conc_cbar_ax, conc_ticks)]:
        cax.set_yticks(ticks)
        cax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        cax.tick_params(labelsize=9, length=3, width=0.8, pad=2)
        cax.yaxis.label.set_size(10)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        output_path = output_dir / f"{output_prefix}{suffix}"
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"[OK] Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript Figure 2.")
    parser.add_argument(
        "--model_convergence_results",
        type=Path,
        default=default_input(),
        help="Model-convergence summary TSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for regenerated figure files.",
    )
    parser.add_argument(
        "--output_prefix",
        default="figure2_model_convergence_heatmaps",
        help="Output filename prefix without extension.",
    )
    args = parser.parse_args()

    df = load_results(args.model_convergence_results)
    make_figure(df, args.output_dir, args.output_prefix)


if __name__ == "__main__":
    main()
