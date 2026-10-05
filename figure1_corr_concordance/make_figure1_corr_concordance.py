"""
Render manuscript Figure 1 from the nominal and fine-mapped correlation/concordance summaries.

Inputs (in $S2F_RESULTS): nominal_tool_performance_results.tsv, finemapped_stratified_correlations.tsv,
finemapped_stratified_concordance.tsv. Output (in $S2F_FIGURE_OUTPUT): fig_corr_concordance_combined.pdf/.png.

2x2 layout. Each cell = Borzoi (left) + AlphaGenome (right) heatmap.
Column headers: Borzoi | AlphaGenome at top.
Row labels on left: (A)/(C) for row 0/1, subtitle at same height.
Panel labels (A)(B)(C)(D) inside top-left of each left-hand heatmap.
"""
import pandas as pd
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path


def rounded_limits(*series: pd.Series) -> tuple[float, float]:
    """Return data-driven colorbar limits rounded outward to one decimal."""
    values = pd.concat([pd.to_numeric(s, errors="coerce") for s in series]).dropna()
    if values.empty:
        raise ValueError("Cannot determine colorbar limits from empty data")
    lower_tenths = math.floor(values.min() * 10)
    upper_tenths = math.ceil(values.max() * 10)
    if lower_tenths == upper_tenths:
        upper_tenths += 1
    return lower_tenths / 10, upper_tenths / 10


def rounded_ticks(vmin: float, vmax: float) -> list[float]:
    """Use clean one-decimal colorbar ticks, always including min/max."""
    vmin = round(vmin, 1)
    vmax = round(vmax, 1)
    step = 0.1 if (vmax - vmin) <= 0.600001 else 0.2
    ticks = [vmin]
    next_tick = math.ceil((vmin + 1e-9) / step) * step
    while next_tick < vmax - 1e-9:
        ticks.append(next_tick)
        next_tick += step
    ticks.append(vmax)
    return sorted(set(round(t, 1) for t in ticks))


def annotate_missing_cells(ax, data: pd.DataFrame, label: str = "n<10") -> None:
    """Mark masked cells so excluded bins are visually intentional."""
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


RESULTS = Path(os.environ["S2F_RESULTS"])
OUT = Path(os.environ.get("S2F_FIGURE_OUTPUT", str(RESULTS / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

NOMINAL_TOOL_PERFORMANCE = RESULTS / "nominal_tool_performance_results.tsv"
FINE_CORR = RESULTS / "finemapped_stratified_correlations.tsv"
FINE_CONC = RESULTS / "finemapped_stratified_concordance.tsv"

TOOLS              = ["borzoi", "alphagenome"]
TOOL_LABELS        = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome"}
ANCESTRIES_NOMINAL = ["AA", "CH", "NHW"]
ANCESTRIES_FINE    = ["AA", "CH", "NHW"]
TSS_ORDER          = ["0-3kb", "3-12kb", "12-35kb", ">35kb"]
ANCESTRY_LABELS    = {
    "AA_specific": "AA", "CH_specific": "CH", "NHW_specific": "NHW",
    "AA": "AA", "CH": "CH", "NHW": "NHW",
}

nominal = pd.read_csv(NOMINAL_TOOL_PERFORMANCE, sep="\t")
nominal = nominal[(nominal["ancestry"] != "All") & (nominal["tss_distance_bin"] != "All")].copy()
nom_corr = nominal.copy()
nom_conc = nominal.rename(columns={"concordance": "concordance_rate"}).copy()
fin_corr = pd.read_csv(FINE_CORR,   sep="\t")
fin_conc = pd.read_csv(FINE_CONC,   sep="\t")

# Use the same clean colorbar presentation as Figure S1. The data include a few
# tiny negative Spearman estimates and a few concordance values just above 0.9,
# but those edge cases are still printed in-cell; the shared scale is easier to
# read across manuscript figures than data-derived labels such as -0.1 or 1.0.
corr_vmin, corr_vmax = 0.0, 1.0
conc_vmin, conc_vmax = 0.4, 0.9
CORR_NORM = mcolors.Normalize(vmin=corr_vmin, vmax=corr_vmax)
CONC_NORM = mcolors.Normalize(vmin=conc_vmin, vmax=conc_vmax)
HEATMAP_CMAP = plt.get_cmap("viridis").copy()
HEATMAP_CMAP.set_bad("#e6e6e6")

# 2x2 panels: (row, col) -> plotting metadata
panels = {
    (0, 0): {
        "label": "A", "subtitle": "Nominal eQTLs", "df": nom_corr, "val": "spearman_r",
        "cmap": HEATMAP_CMAP, "norm": CORR_NORM, "cbar": "Spearman r", "fmt": ".3f", "ancestries": ANCESTRIES_NOMINAL,
    },
    (0, 1): {
        "label": "B", "subtitle": "Fine-mapped (PIP ≥ 0.9)", "df": fin_corr, "val": "spearman_r",
        "cmap": HEATMAP_CMAP, "norm": CORR_NORM, "cbar": "Spearman r", "fmt": ".3f", "ancestries": ANCESTRIES_FINE,
    },
    (1, 0): {
        "label": "C", "subtitle": "Nominal eQTLs", "df": nom_conc, "val": "concordance_rate",
        "cmap": HEATMAP_CMAP, "norm": CONC_NORM, "cbar": "Concordance", "fmt": ".3f", "ancestries": ANCESTRIES_NOMINAL,
    },
    (1, 1): {
        "label": "D", "subtitle": "Fine-mapped (PIP ≥ 0.9)", "df": fin_conc, "val": "concordance_rate",
        "cmap": HEATMAP_CMAP, "norm": CONC_NORM, "cbar": "Concordance", "fmt": ".3f", "ancestries": ANCESTRIES_FINE,
    },
}
row_ylabel = {0: "Spearman Correlation", 1: "Direction Concordance"}
row_cbar_meta = {
    0: {"cmap": HEATMAP_CMAP, "norm": CORR_NORM, "label": "Spearman r", "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
    1: {"cmap": HEATMAP_CMAP, "norm": CONC_NORM, "label": "Concordance", "ticks": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
}

# Main layout: two subplot pairs plus one colorbar column.
# Use nested grids so the Borzoi/AlphaGenome gap can be tighter than the gap
# between the two panel groups.
fig = plt.figure(figsize=(20, 10.5))
main_gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[1, 1, 0.02],
    hspace=0.22, wspace=0.01,
    left=0.08, right=0.95, top=0.93, bottom=0.06
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

for (pr, pc), meta in panels.items():
    lbl = meta["label"]
    subtitle = meta["subtitle"]
    df = meta["df"]
    val = meta["val"]
    cmap = meta["cmap"]
    norm = meta["norm"]
    fmt = meta["fmt"]
    ancs = meta["ancestries"]

    pair_gs = pair_specs[(pr, pc)]
    ax_borzoi = fig.add_subplot(pair_gs[0, 0])
    ax_alpha  = fig.add_subplot(pair_gs[0, 1])
    panel_title_anchors.append((pr, ax_borzoi, ax_alpha, f"({lbl})  {subtitle}"))
    pair_axes.append((ax_borzoi, ax_alpha))
    if pc == 0:
        left_pair_axes[pr] = (ax_borzoi, ax_alpha)
    else:
        right_pair_axes[pr] = (ax_borzoi, ax_alpha)

    for ax_idx, (ax, tool) in enumerate(zip([ax_borzoi, ax_alpha], TOOLS)):
        tool_df = df[(df["tool"] == tool) & (df["tss_distance_bin"] != "unknown")]
        pivot = tool_df.pivot(index="tss_distance_bin", columns="ancestry", values=val)
        pivot = pivot.reindex(index=TSS_ORDER, columns=ancs)
        pivot.columns = [ANCESTRY_LABELS.get(a, a) for a in pivot.columns]

        ax.set_facecolor("#e6e6e6")
        sns.heatmap(
            pivot, annot=True, fmt=fmt, cmap=cmap, norm=norm,
            ax=ax, cbar=False,
            square=True, annot_kws={"size": 12}, linewidths=0.5,
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

    if pc == 0:
        ax_borzoi.set_ylabel(row_ylabel[pr], fontsize=12)

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
for pr in [0, 1]:
    left_b, left_a = left_pair_axes[pr]
    right_b, right_a = right_pair_axes[pr]
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
for pr, ax_left, ax_right, text in panel_title_anchors:
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    y_top = max(pos_l.y1, pos_r.y1)
    y_offset = 0.028 if pr == 0 else 0.026
    fig.text(pos_l.x0, y_top + y_offset, text,
             fontsize=11, fontweight="bold", ha="left", va="bottom")

for pr, meta in row_cbar_meta.items():
    cax = row_cbar_axes[pr]
    ref_left, ref_right = right_pair_axes[pr]
    ref_pos = ref_right.get_position()
    cbar_gap = 0.004
    cbar_width = ref_pos.width * 0.10
    cax.set_position([ref_pos.x1 + cbar_gap, ref_pos.y0, cbar_width, ref_pos.height])
    sm = cm.ScalarMappable(cmap=meta["cmap"], norm=meta["norm"])
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks(meta["ticks"])
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    cbar.ax.tick_params(labelsize=8, length=3, width=0.8, pad=2)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label(meta["label"], fontsize=10, rotation=270, labelpad=10)

for suffix in [".pdf", ".png"]:
    fig.savefig(OUT / f"fig_corr_concordance_combined{suffix}", bbox_inches="tight", dpi=300)
    print(f"[OK] Saved fig_corr_concordance_combined{suffix}")
plt.close()
