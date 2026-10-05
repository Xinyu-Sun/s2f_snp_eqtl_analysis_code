"""Figure 3 (four panels), styled like Figures 1 and 2: model colors Borzoi #E64B35 / AlphaGenome #4DBBD5,
viridis heatmap with annotated cells (dark low, bright high), bold (A)-(D) labels with italic descriptors,
12-pt sans text on a 16 x 11 inch canvas.
(a) |S2F score| for low-PIP comparison variants (hatched) and high-PIP variants (PIP >= 0.9) by group.
(b) Distance-matched AUROC with comparison-only (open) and joint (filled) resampling intervals.
(c) AUROC when one group's high-PIP variants are evaluated against each group's comparison pool.
(d) Odds of a high-PIP variant (PIP >= 0.5) in the top versus bottom decile of |score| (exact conditional OR, exact 95% CI).
Usage: python a8_make_anatomy_figure.py <pairs.tsv.gz> <results_dir> <out_pdf>
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm, colors as mcolors
from fu_common import load_pairs, split_classes, GROUPS, MODELS
pairs_path, res, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL_COLORS = {"borzoi": "#E64B35", "alphagenome": "#4DBBD5"}
TOOL_LABELS = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome"}
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42})
df = load_pairs(pairs_path)
bt = pd.read_csv(f"{res}/a2_auroc_bootstrap.tsv", sep="\t"); bt = bt[(bt.pip_threshold == 0.9) & (bt.comparison == "low_pip")]
cp = pd.read_csv(f"{res}/a2_cross_pool_auroc.tsv", sep="\t"); cp = cp[(cp.pip_threshold == 0.9) & (cp.comparison == "low_pip")]
de = pd.read_csv(f"{res}/a3_decile_enrichment.tsv", sep="\t"); de = de[de.pip_threshold == 0.5]

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22, left=0.07, right=0.97, top=0.90, bottom=0.07)
ax_a, ax_b, ax_d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
gs_c = gs[1, 0].subgridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.25)
ax_c = [fig.add_subplot(gs_c[0, 0]), fig.add_subplot(gs_c[0, 1])]; cax = fig.add_subplot(gs_c[0, 2])

def block_axes(ax, labels, n_blocks=2, block_gap=1.0):
    """x positions for group labels inside model blocks; returns centres per block."""
    centres = []
    for b in range(n_blocks):
        base = b * (len(labels) + block_gap)
        centres.append([base + i for i in range(len(labels))])
    ticks = [c for blk in centres for c in blk]
    ax.set_xticks(ticks); ax.set_xticklabels(labels * n_blocks)
    for b, blk in enumerate(centres):
        ax.text(np.mean(blk), -0.17, TOOL_LABELS[list(MODELS)[b]], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12, color="#444444")
    ax.set_xlim(-0.7, ticks[-1] + 0.7)
    return centres

def panel_label(ax, letter, desc):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.048, pos.y1 + 0.028, letter, fontsize=16, fontweight="bold", ha="left", va="bottom")
    fig.text(pos.x0 - 0.005, pos.y1 + 0.028, desc, fontsize=12, style="italic", color="#444444", ha="left", va="bottom")

def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# (a) score magnitude
box_style = dict(medianprops=dict(color="black", lw=1.4), whiskerprops=dict(color="#555555", lw=1.1), capprops=dict(color="#555555", lw=1.1), boxprops=dict(edgecolor="#555555", lw=1.1), showfliers=False, patch_artist=True, widths=0.36)
centres = [[b * 4.0 + i for i in range(3)] for b in range(2)]
for b, (model, col) in enumerate(MODELS.items()):
    for i, g in enumerate(GROUPS):
        pos, low, _ = split_classes(df[df["ancestry"] == g], col, 0.9)
        for cls, sub, dx, hatch, alpha in (("low", low, -0.21, "////", 0.45), ("pos", pos, 0.21, None, 0.85)):
            v = np.abs(sub[col].values) + 1e-6
            bp = ax_a.boxplot([v], positions=[centres[b][i] + dx], **box_style)
            bp["boxes"][0].set(facecolor=MODEL_COLORS[model], alpha=alpha, hatch=hatch)
block_axes(ax_a, GROUPS)
ax_a.set_yscale("log"); ax_a.set_ylim(3e-7, 30.0); ax_a.set_ylabel("|S2F score|")
despine(ax_a); panel_label(ax_a, "(A)", "score magnitude of high-PIP and comparison variants")
ax_a.legend(handles=[Patch(facecolor="white", edgecolor="#555555", hatch="////", label="Low-PIP comparison (PIP < 0.01)"), Patch(facecolor="#bbbbbb", edgecolor="#555555", label="High-PIP (PIP ≥ 0.9)")], loc="upper left", frameon=False, fontsize=11)

# (b) AUROC: comparison-only (open) vs joint (filled)
centres = block_axes(ax_b, GROUPS)
for b, model in enumerate(MODELS):
    for i, g in enumerate(GROUPS):
        r = bt[(bt.model == model) & (bt.group == g)].iloc[0]; x = centres[b][i]; c = MODEL_COLORS[model]
        ax_b.errorbar(x - 0.16, r.auroc_comp_only_mean, yerr=[[r.auroc_comp_only_mean - r.comp_only_ci_low], [r.comp_only_ci_high - r.auroc_comp_only_mean]], fmt="s", ms=7, mfc="white", mec=c, color=c, capsize=4, lw=1.3, zorder=3)
        ax_b.errorbar(x + 0.16, r.auroc_joint_mean, yerr=[[r.auroc_joint_mean - r.joint_ci_low], [r.joint_ci_high - r.auroc_joint_mean]], fmt="o", ms=7, mfc=c, mec="#555555", mew=0.8, color=c, capsize=4, lw=1.3, zorder=3)
ax_b.axhline(0.5, color="red", ls="--", lw=1, alpha=0.6); ax_b.set_ylim(0.45, 1.04); ax_b.set_ylabel("AUROC (PIP ≥ 0.9 vs. low-PIP)")
despine(ax_b); panel_label(ax_b, "(B)", "AUROC with comparison-only or joint resampling")
ax_b.legend(handles=[Line2D([], [], marker="s", mfc="white", mec="#555555", color="#555555", ls="", ms=7, label="Comparison variants resampled"), Line2D([], [], marker="o", mfc="#bbbbbb", mec="#555555", color="#555555", ls="", ms=7, label="Positive genes and comparison variants resampled")], loc="upper left", frameon=False, fontsize=11)

# (c) cross-pool heatmaps, viridis as in Figure S1
norm = mcolors.Normalize(vmin=0.65, vmax=0.90); cmap = plt.get_cmap("viridis")
for k, model in enumerate(MODELS):
    ax = ax_c[k]
    m = cp[cp.model == model].pivot(index="positives_from", columns="comparison_pool_from", values="auroc_mean").loc[GROUPS, GROUPS].values
    ax.pcolormesh(np.arange(4), np.arange(4), m, cmap=cmap, norm=norm, edgecolors="white", linewidth=2)
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.5, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=11, color="white" if norm(m[i, j]) < 0.7 else "black")
    ax.set_xticks(np.arange(3) + 0.5); ax.set_xticklabels(GROUPS); ax.set_yticks(np.arange(3) + 0.5); ax.set_yticklabels(GROUPS if k == 0 else [])
    ax.invert_yaxis(); ax.set_title(TOOL_LABELS[model], fontsize=12, pad=4); ax.set_xlabel("Comparison pool from")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
ax_c[0].set_ylabel("High-PIP variants from")
cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax); cb.set_label("AUROC"); cb.outline.set_visible(False)
panel_label(ax_c[0], "(C)", "high-PIP variants vs. each group's comparison pool")

# (d) top-vs-bottom decile enrichment
centres = block_axes(ax_d, GROUPS)
for b, model in enumerate(MODELS):
    for i, g in enumerate(GROUPS):
        r = de[(de.model == model) & (de.group == g)].iloc[0]; c = MODEL_COLORS[model]
        # exact conditional odds ratio and exact 95% interval, as in the decile-enrichment table
        ax_d.errorbar(centres[b][i], r.exact_or, yerr=[[r.exact_or - r.exact_ci_low], [r.exact_ci_high - r.exact_or]], fmt="o", ms=7, mfc=c, mec="#555555", mew=0.8, color=c, capsize=4, lw=1.3, zorder=3)
ax_d.set_yscale("log"); ax_d.axhline(1, color="red", ls="--", lw=1, alpha=0.6); ax_d.set_ylim(0.8, 150)
ax_d.set_ylabel("Odds ratio, top vs. bottom score decile\n(high-PIP, PIP ≥ 0.5)")
despine(ax_d); panel_label(ax_d, "(D)", "enrichment of high-PIP variants among top-scored variants")

fig.legend(handles=[Patch(facecolor=MODEL_COLORS[t], edgecolor="#555555", label=TOOL_LABELS[t], alpha=0.85) for t in MODELS], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985), fontsize=12)
fig.savefig(out_pdf, bbox_inches="tight", dpi=250); fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=200); print("figure written", out_pdf)
