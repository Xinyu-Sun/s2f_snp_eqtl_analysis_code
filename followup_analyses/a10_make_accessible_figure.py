"""Figure 4 (four panels), styled like the other figures: model colors Borzoi #E64B35 / AlphaGenome #4DBBD5,
bold (A)-(D) labels with italic descriptors, 12-pt sans text on a 16 x 11 inch canvas.
(a) Share of high-PIP variants (PIP >= 0.9, solid) and low-PIP comparison variants (hatched) in FILER accessible chromatin, with Wilson 95% intervals.
(b) |S2F score| of high-PIP variants inside (solid) and outside (hatched) accessible chromatin.
(c) Distance-matched AUROC for high-PIP variants inside (filled circles) and outside (open squares) accessible chromatin.
(d) Distance-matched AUROC of accessibility alone, the model score alone, and accessibility with the score (b4 baselines).
Usage: python a10_make_accessible_figure.py <pairs.tsv.gz> <filer_category_counts.tsv.gz> <results_dir> <out_pdf>
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from fu_common import load_pairs, split_classes, GROUPS, MODELS
pairs_path, filer_path, res, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MODEL_COLORS = {"borzoi": "#E64B35", "alphagenome": "#4DBBD5"}
TOOL_LABELS = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome"}
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42})

df = load_pairs(pairs_path)
fl = pd.read_csv(filer_path, sep="\t", usecols=["chromosome", "position", "ref_allele", "alt_allele", "filer_binary_Accessible_chromatin"]).rename(columns={"filer_binary_Accessible_chromatin": "acc"})
df = df.merge(fl.drop_duplicates(["chromosome", "position", "ref_allele", "alt_allele"]), on=["chromosome", "position", "ref_allele", "alt_allele"], how="left")
au = pd.read_csv(f"{res}/a9_accessible_auroc.tsv", sep="\t"); au = au[au.pip_threshold == 0.9]
b4 = pd.read_csv(f"{res}/b4_baseline_auroc.tsv", sep="\t")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22, left=0.07, right=0.97, top=0.90, bottom=0.07)
ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))


def block_axes(ax, labels, n_blocks=2, block_gap=1.0):
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


# (a) share in accessible chromatin: low-PIP comparison (hatched) vs high-PIP (solid)
centres = block_axes(ax_a, GROUPS)
for b, (model, col) in enumerate(MODELS.items()):
    for i, g in enumerate(GROUPS):
        pos, low, _ = split_classes(df[df["ancestry"] == g], col, 0.9)
        for sub, dx, hatch, alpha in ((low, -0.21, "////", 0.45), (pos, 0.21, None, 0.85)):
            called = sub["acc"].notna(); p, lo, hi = wilson(int((sub["acc"] == 1).sum()), int(called.sum()))
            ax_a.bar(centres[b][i] + dx, p, width=0.36, color=MODEL_COLORS[model], alpha=alpha, hatch=hatch, edgecolor="#555555", lw=1.1)
            ax_a.errorbar(centres[b][i] + dx, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="#333333", capsize=3, lw=1.1, zorder=3)
ax_a.set_ylim(0, 1.0); ax_a.set_ylabel("Share in accessible chromatin")
despine(ax_a); panel_label(ax_a, "(A)", "overlap with FILER accessible-chromatin annotations")
ax_a.legend(handles=[Patch(facecolor="white", edgecolor="#555555", hatch="////", label="Low-PIP comparison (PIP < 0.01)"), Patch(facecolor="#bbbbbb", edgecolor="#555555", label="High-PIP (PIP ≥ 0.9)")], loc="upper left", frameon=False, fontsize=11)

# (b) score magnitude inside vs outside accessible chromatin (high-PIP, PIP >= 0.9)
box_style = dict(medianprops=dict(color="black", lw=1.4), whiskerprops=dict(color="#555555", lw=1.1), capprops=dict(color="#555555", lw=1.1), boxprops=dict(edgecolor="#555555", lw=1.1), showfliers=False, patch_artist=True, widths=0.36)
centres = [[b * 4.0 + i for i in range(3)] for b in range(2)]
for b, (model, col) in enumerate(MODELS.items()):
    for i, g in enumerate(GROUPS):
        pos = split_classes(df[df["ancestry"] == g], col, 0.9)[0]
        for sub, dx, hatch, alpha in ((pos[pos["acc"] == 0], -0.21, "////", 0.45), (pos[pos["acc"] == 1], 0.21, None, 0.85)):
            v = np.abs(sub[col].values) + 1e-6
            bp = ax_b.boxplot([v], positions=[centres[b][i] + dx], **box_style)
            bp["boxes"][0].set(facecolor=MODEL_COLORS[model], alpha=alpha, hatch=hatch)
block_axes(ax_b, GROUPS)
ax_b.set_yscale("log"); ax_b.set_ylim(3e-7, 30.0); ax_b.set_ylabel("|S2F score| of high-PIP variants")
despine(ax_b); panel_label(ax_b, "(B)", "score magnitude inside and outside accessible chromatin")
ax_b.legend(handles=[Patch(facecolor="white", edgecolor="#555555", hatch="////", label="Outside accessible chromatin"), Patch(facecolor="#bbbbbb", edgecolor="#555555", label="Inside accessible chromatin")], loc="upper left", frameon=False, fontsize=11)

# (c) AUROC inside (filled circle) vs outside (open square)
centres = block_axes(ax_c, GROUPS)
for b, model in enumerate(MODELS):
    for i, g in enumerate(GROUPS):
        x = centres[b][i]; c = MODEL_COLORS[model]
        for stratum, dx, fmt, mfc in (("not_accessible", -0.16, "s", "white"), ("accessible", 0.16, "o", c)):
            r = au[(au.model == model) & (au.group == g) & (au.stratum == stratum)].iloc[0]
            ax_c.errorbar(x + dx, r.auroc_mean, yerr=[[r.auroc_mean - r.ci_low], [r.ci_high - r.auroc_mean]], fmt=fmt, ms=7, mfc=mfc, mec=c if mfc == "white" else "#555555", mew=0.8 if mfc != "white" else 1.3, color=c, capsize=4, lw=1.3, zorder=3)
ax_c.axhline(0.5, color="red", ls="--", lw=1, alpha=0.6); ax_c.set_ylim(0.4, 1.0); ax_c.set_ylabel("AUROC (PIP ≥ 0.9 vs. low-PIP)")
despine(ax_c); panel_label(ax_c, "(C)", "separation of high-PIP variants inside and outside accessible chromatin")
ax_c.legend(handles=[Line2D([], [], marker="s", mfc="white", mec="#555555", color="#555555", ls="", ms=7, label="Outside accessible chromatin"), Line2D([], [], marker="o", mfc="#bbbbbb", mec="#555555", color="#555555", ls="", ms=7, label="Inside accessible chromatin")], loc="lower left", frameon=False, fontsize=11)

# (d) annotation-only baselines and the model score (b4)
PRED = [("accessible_only", -0.22, "D", "white", "Accessible-chromatin overlap alone"), ("score", 0.0, "o", "fill", "Model score alone"), ("annotation_plus_score", 0.22, "^", "fill", "Accessibility + score")]
centres = block_axes(ax_d, GROUPS)
for b, model in enumerate(MODELS):
    for i, g in enumerate(GROUPS):
        x = centres[b][i]; c = MODEL_COLORS[model]
        for pred, dx, fmt, mfc, _ in PRED:
            r = b4[(b4.model == model) & (b4.group == g) & (b4.predictor == pred)].iloc[0]
            ax_d.errorbar(x + dx, r.auroc_mean, yerr=[[r.auroc_mean - r.ci_low], [r.ci_high - r.auroc_mean]], fmt=fmt, ms=7, mfc=c if mfc == "fill" else "white", mec="#555555" if mfc == "fill" else c, mew=0.8 if mfc == "fill" else 1.3, color=c, capsize=3, lw=1.2, zorder=3)
ax_d.axhline(0.5, color="red", ls="--", lw=1, alpha=0.6); ax_d.set_ylim(0.4, 1.0); ax_d.set_ylabel("AUROC (PIP ≥ 0.9 vs. low-PIP, annotated variants)")
despine(ax_d); panel_label(ax_d, "(D)", "what the annotation alone provides, and what the score adds")
ax_d.legend(handles=[Line2D([], [], marker=f, mfc="#bbbbbb" if m == "fill" else "white", mec="#555555", color="#555555", ls="", ms=7, label=l) for _, _, f, m, l in PRED], loc="lower left", frameon=False, fontsize=11, ncol=2)

fig.legend(handles=[Patch(facecolor=MODEL_COLORS[t], edgecolor="#555555", label=TOOL_LABELS[t], alpha=0.85) for t in MODELS], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985), fontsize=12)
fig.savefig(out_pdf, bbox_inches="tight", dpi=250); fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=200); print("figure written", out_pdf)
