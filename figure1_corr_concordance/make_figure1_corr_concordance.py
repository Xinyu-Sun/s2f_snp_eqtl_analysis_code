#!/usr/bin/env python3
"""
Create Figure 1: S2F model correlation and direction concordance with eQTL
effect sizes, stratified by TSS distance and ancestry.

The script redraws the manuscript panel from summary tables generated during
the nominal and fine-mapped eQTL benchmark analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


TOOLS = ["borzoi", "alphagenome"]
TOOL_LABELS = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome"}
ANCESTRIES = ["AA", "CH", "NHW"]
TSS_ORDER = ["0-3kb", "3-12kb", "12-35kb", ">35kb"]
ANCESTRY_LABELS = {
    "AA_specific": "AA",
    "CH_specific": "CH",
    "NHW_specific": "NHW",
    "AA": "AA",
    "CH": "CH",
    "NHW": "NHW",
}


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data_availability" / "figure1_corr_concordance"


def annotate_missing_cells(ax: plt.Axes, data: pd.DataFrame, label: str = "n<10") -> None:
    """Mark masked heatmap cells so excluded bins are explicit."""
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


def normalize_ancestry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ancestry"] = out["ancestry"].map(lambda value: ANCESTRY_LABELS.get(value, value))
    return out


def read_required_tsv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    missing = required_columns.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_list}")
    return df


def prepare_nominal(nominal_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nominal = read_required_tsv(
        nominal_path,
        {"tool", "ancestry", "tss_distance_bin", "spearman_r", "concordance"},
    )
    nominal = normalize_ancestry(nominal)
    nominal = nominal[
        nominal["tool"].isin(TOOLS)
        & nominal["ancestry"].isin(ANCESTRIES)
        & nominal["tss_distance_bin"].isin(TSS_ORDER)
    ].copy()

    nominal_corr = nominal.copy()
    nominal_conc = nominal.rename(columns={"concordance": "concordance_rate"}).copy()
    return nominal_corr, nominal_conc


def prepare_finemapped(corr_path: Path, conc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    finemapped_corr = read_required_tsv(
        corr_path,
        {"tool", "ancestry", "tss_distance_bin", "spearman_r"},
    )
    finemapped_conc = read_required_tsv(
        conc_path,
        {"tool", "ancestry", "tss_distance_bin", "concordance_rate"},
    )

    finemapped_corr = normalize_ancestry(finemapped_corr)
    finemapped_conc = normalize_ancestry(finemapped_conc)

    finemapped_corr = finemapped_corr[
        finemapped_corr["tool"].isin(TOOLS)
        & finemapped_corr["ancestry"].isin(ANCESTRIES)
        & finemapped_corr["tss_distance_bin"].isin(TSS_ORDER)
    ].copy()
    finemapped_conc = finemapped_conc[
        finemapped_conc["tool"].isin(TOOLS)
        & finemapped_conc["ancestry"].isin(ANCESTRIES)
        & finemapped_conc["tss_distance_bin"].isin(TSS_ORDER)
    ].copy()
    return finemapped_corr, finemapped_conc


def pivot_panel(df: pd.DataFrame, tool: str, value_col: str) -> pd.DataFrame:
    tool_df = df[df["tool"] == tool]
    pivot = tool_df.pivot(index="tss_distance_bin", columns="ancestry", values=value_col)
    return pivot.reindex(index=TSS_ORDER, columns=ANCESTRIES)


def make_figure(
    nominal_corr: pd.DataFrame,
    nominal_conc: pd.DataFrame,
    finemapped_corr: pd.DataFrame,
    finemapped_conc: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
) -> None:
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

    heatmap_cmap = plt.get_cmap("RdYlGn").copy()
    heatmap_cmap.set_bad("#e6e6e6")

    corr_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    conc_norm = mcolors.Normalize(vmin=0.4, vmax=0.9)

    panels = {
        (0, 0): {
            "label": "a",
            "subtitle": "Nominal eQTLs",
            "df": nominal_corr,
            "value": "spearman_r",
            "norm": corr_norm,
            "fmt": ".3f",
        },
        (0, 1): {
            "label": "b",
            "subtitle": "Fine-mapped (PIP >= 0.9)",
            "df": finemapped_corr,
            "value": "spearman_r",
            "norm": corr_norm,
            "fmt": ".3f",
        },
        (1, 0): {
            "label": "c",
            "subtitle": "Nominal eQTLs",
            "df": nominal_conc,
            "value": "concordance_rate",
            "norm": conc_norm,
            "fmt": ".3f",
        },
        (1, 1): {
            "label": "d",
            "subtitle": "Fine-mapped (PIP >= 0.9)",
            "df": finemapped_conc,
            "value": "concordance_rate",
            "norm": conc_norm,
            "fmt": ".3f",
        },
    }
    row_ylabel = {0: "Spearman Correlation", 1: "Direction Concordance"}
    row_cbar_meta = {
        0: {
            "norm": corr_norm,
            "label": "Spearman r",
            "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        },
        1: {
            "norm": conc_norm,
            "label": "Concordance",
            "ticks": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    }

    fig = plt.figure(figsize=(20, 10.5))
    main_gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1, 1, 0.02],
        hspace=0.22,
        wspace=0.01,
        left=0.08,
        right=0.95,
        top=0.93,
        bottom=0.06,
    )
    pair_specs = {
        (0, 0): main_gs[0, 0].subgridspec(1, 2, wspace=0.005),
        (0, 1): main_gs[0, 1].subgridspec(1, 2, wspace=0.005),
        (1, 0): main_gs[1, 0].subgridspec(1, 2, wspace=0.005),
        (1, 1): main_gs[1, 1].subgridspec(1, 2, wspace=0.005),
    }
    row_cbar_axes = {
        0: fig.add_subplot(main_gs[0, 2]),
        1: fig.add_subplot(main_gs[1, 2]),
    }
    panel_title_anchors = []
    pair_axes = []
    left_pair_axes = {}
    right_pair_axes = {}

    for (panel_row, panel_col), meta in panels.items():
        pair_gs = pair_specs[(panel_row, panel_col)]
        ax_borzoi = fig.add_subplot(pair_gs[0, 0])
        ax_alpha = fig.add_subplot(pair_gs[0, 1])
        panel_title_anchors.append(
            (
                panel_row,
                ax_borzoi,
                ax_alpha,
                f"({meta['label']})  {meta['subtitle']}",
            )
        )
        pair_axes.append((ax_borzoi, ax_alpha))
        if panel_col == 0:
            left_pair_axes[panel_row] = (ax_borzoi, ax_alpha)
        else:
            right_pair_axes[panel_row] = (ax_borzoi, ax_alpha)

        for ax_idx, (ax, tool) in enumerate(zip([ax_borzoi, ax_alpha], TOOLS)):
            pivot = pivot_panel(meta["df"], tool, meta["value"])

            ax.set_facecolor("#e6e6e6")
            sns.heatmap(
                pivot,
                annot=True,
                fmt=meta["fmt"],
                cmap=heatmap_cmap,
                norm=meta["norm"],
                ax=ax,
                cbar=False,
                square=True,
                annot_kws={"size": 12},
                linewidths=0.5,
            )
            annotate_missing_cells(ax, pivot)
            ax.set_title(TOOL_LABELS[tool], fontsize=11, pad=3)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=11)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            if ax_idx == 1:
                ax.tick_params(axis="y", labelleft=False)

        if panel_col == 0:
            ax_borzoi.set_ylabel(row_ylabel[panel_row], fontsize=12)

    fig.canvas.draw()
    for ax_left, ax_right in pair_axes:
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        gap = pos_r.x0 - pos_l.x1
        target_gap = 0.002
        if gap > target_gap:
            delta = (gap - target_gap) / 2
            ax_left.set_position([pos_l.x0, pos_l.y0, pos_l.width + delta, pos_l.height])
            ax_right.set_position([pos_r.x0 - delta, pos_r.y0, pos_r.width + delta, pos_r.height])

    fig.canvas.draw()
    for panel_row in [0, 1]:
        left_b, left_a = left_pair_axes[panel_row]
        right_b, right_a = right_pair_axes[panel_row]
        left_edge = max(left_b.get_position().x1, left_a.get_position().x1)
        right_edge = min(right_b.get_position().x0, right_a.get_position().x0)
        gap = right_edge - left_edge
        target_group_gap = 0.050
        if gap > target_group_gap:
            shift = gap - target_group_gap
            for ax in [right_b, right_a]:
                pos = ax.get_position()
                ax.set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])

    fig.canvas.draw()
    for panel_row, ax_left, ax_right, text in panel_title_anchors:
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        y_top = max(pos_l.y1, pos_r.y1)
        y_offset = 0.028 if panel_row == 0 else 0.026
        fig.text(
            pos_l.x0,
            y_top + y_offset,
            text,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    for panel_row, meta in row_cbar_meta.items():
        cax = row_cbar_axes[panel_row]
        _, ref_right = right_pair_axes[panel_row]
        ref_pos = ref_right.get_position()
        cbar_gap = 0.004
        cbar_width = ref_pos.width * 0.10
        cax.set_position([ref_pos.x1 + cbar_gap, ref_pos.y0, cbar_width, ref_pos.height])
        sm = cm.ScalarMappable(cmap=heatmap_cmap, norm=meta["norm"])
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks(meta["ticks"])
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        cbar.ax.tick_params(labelsize=8, length=3, width=0.8, pad=2)
        cbar.outline.set_linewidth(0.6)
        cbar.set_label(meta["label"], fontsize=10, rotation=270, labelpad=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        output_path = output_dir / f"{output_prefix}{suffix}"
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"[OK] Saved {output_path}")
    plt.close(fig)


def main() -> None:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description="Generate manuscript Figure 1.")
    parser.add_argument(
        "--nominal_tool_performance",
        type=Path,
        default=data_dir / "nominal_tool_performance_results.tsv",
        help="Nominal eQTL model performance summary TSV.",
    )
    parser.add_argument(
        "--finemapped_correlations",
        type=Path,
        default=data_dir / "finemapped_stratified_correlations.tsv",
        help="Fine-mapped eQTL stratified Spearman correlation TSV.",
    )
    parser.add_argument(
        "--finemapped_concordance",
        type=Path,
        default=data_dir / "finemapped_stratified_concordance.tsv",
        help="Fine-mapped eQTL stratified direction-concordance TSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for regenerated figure files.",
    )
    parser.add_argument(
        "--output_prefix",
        default="figure1_corr_concordance_combined",
        help="Output filename prefix without extension.",
    )
    args = parser.parse_args()

    nominal_corr, nominal_conc = prepare_nominal(args.nominal_tool_performance)
    finemapped_corr, finemapped_conc = prepare_finemapped(
        args.finemapped_correlations,
        args.finemapped_concordance,
    )
    make_figure(
        nominal_corr=nominal_corr,
        nominal_conc=nominal_conc,
        finemapped_corr=finemapped_corr,
        finemapped_conc=finemapped_conc,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
