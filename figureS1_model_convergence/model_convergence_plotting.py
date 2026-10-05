#!/usr/bin/env python3
"""
Compare Borzoi vs AlphaGenome predictions to assess model convergence.

Computes correlation and concordance between the two models' predictions
for both nominal eQTLs and fine-mapped (PIP ≥ 0.9) variants.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# Configuration
# =============================================================================

ANCESTRIES = ["AA", "CH", "NHW"]
ANCESTRY_LABELS = {
    "AA": "African American",
    "CH": "Caribbean Hispanic",
    "NHW": "Non-Hispanic White",
}

TSS_BINS = ["0-3kb", "3-12kb", "12-35kb", ">35kb"]

# For nominal eQTLs - ancestry suffix
ANCESTRY_MAP_NOMINAL = {
    "AA_specific": "AA",
    "CH_specific": "CH",
    "NHW_specific": "NHW",
}

NOMINAL_FILE_PATTERNS = {
    "borzoi": "borzoi.blood.requested_pairs.tsv.gz",
    "alphagenome": "alphagenome.whole_blood.requested_pairs.tsv.gz",
}

FINEMAPPED_FILE_PATTERNS = {
    "borzoi": "borzoi.blood.requested_pairs.tsv.gz",
    "alphagenome": "alphagenome.whole_blood.requested_pairs.tsv.gz",
}


def rounded_limits(series: pd.Series) -> tuple[float, float]:
    """Return data-driven colorbar limits rounded outward to one decimal."""
    values = pd.to_numeric(series, errors="coerce").dropna()
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


# =============================================================================
# Data Loading
# =============================================================================

def load_nominal_paired_data(base_dir: Path) -> pd.DataFrame:
    """Load nominal eQTL data and merge Borzoi + AlphaGenome predictions."""
    tss_bins_slugs = ["0-3kb", "3-12kb", "12-35kb", "gt35kb"]
    
    all_dfs = []
    
    for ancestry_key, ancestry in ANCESTRY_MAP_NOMINAL.items():
        for tss_bin_slug in tss_bins_slugs:
            # Load Borzoi
            borzoi_file = base_dir / "outputs" / "borzoi" / ancestry_key / tss_bin_slug / NOMINAL_FILE_PATTERNS["borzoi"]
            # Load AlphaGenome
            ag_file = base_dir / "outputs" / "alphagenome" / ancestry_key / tss_bin_slug / NOMINAL_FILE_PATTERNS["alphagenome"]
            
            if not borzoi_file.exists() or not ag_file.exists():
                continue
            
            try:
                borzoi_df = pd.read_csv(borzoi_file, sep="\t")
                ag_df = pd.read_csv(ag_file, sep="\t")
                
                # Standardize column names for merging
                # Borzoi uses: chr, pos, ref, alt, gene_id
                # AlphaGenome uses: chromosome, position, ref_allele, alt_allele, gene_id
                
                borzoi_df = borzoi_df.rename(columns={
                    "chr": "chromosome",
                    "pos": "position", 
                    "ref": "ref_allele",
                    "alt": "alt_allele",
                })
                
                # Create merge key
                borzoi_df["merge_key"] = (
                    borzoi_df["chromosome"].astype(str) + "_" +
                    borzoi_df["position"].astype(str) + "_" +
                    borzoi_df["ref_allele"].astype(str) + "_" +
                    borzoi_df["alt_allele"].astype(str) + "_" +
                    borzoi_df["gene_id"].astype(str)
                )
                
                ag_df["merge_key"] = (
                    ag_df["chromosome"].astype(str) + "_" +
                    ag_df["position"].astype(str) + "_" +
                    ag_df["ref_allele"].astype(str) + "_" +
                    ag_df["alt_allele"].astype(str) + "_" +
                    ag_df["gene_id"].astype(str)
                )
                
                # Select columns
                borzoi_cols = borzoi_df[["merge_key", "logMeanSED", "beta"]].copy()
                borzoi_cols = borzoi_cols.rename(columns={"logMeanSED": "borzoi_score", "beta": "beta"})
                
                ag_cols = ag_df[["merge_key", "raw_score"]].copy()
                ag_cols = ag_cols.rename(columns={"raw_score": "alphagenome_score"})
                
                # Merge
                merged = borzoi_cols.merge(ag_cols, on="merge_key", how="inner")
                merged["ancestry"] = ancestry
                merged["tss_distance_bin"] = tss_bin_slug.replace("gt35kb", ">35kb")
                
                all_dfs.append(merged)
                
            except Exception as e:
                print(f"  [WARN] Error processing {ancestry_key}/{tss_bin_slug}: {e}")
    
    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        # Ensure numeric
        result["borzoi_score"] = pd.to_numeric(result["borzoi_score"], errors="coerce")
        result["alphagenome_score"] = pd.to_numeric(result["alphagenome_score"], errors="coerce")
        result["beta"] = pd.to_numeric(result["beta"], errors="coerce")
        print(f"  Loaded {len(result):,} paired nominal variants")
        return result
    
    return pd.DataFrame()


def load_finemapped_paired_data(base_dir: Path) -> pd.DataFrame:
    """Load fine-mapped eQTL data and merge Borzoi + AlphaGenome predictions."""
    all_dfs = []
    
    for ancestry in ANCESTRIES:
        # Load Borzoi
        borzoi_file = base_dir / "outputs_finemapped" / "borzoi" / ancestry / FINEMAPPED_FILE_PATTERNS["borzoi"]
        # Load AlphaGenome  
        ag_file = base_dir / "outputs_finemapped" / "alphagenome" / ancestry / FINEMAPPED_FILE_PATTERNS["alphagenome"]
        
        if not borzoi_file.exists() or not ag_file.exists():
            continue
        
        try:
            borzoi_df = pd.read_csv(borzoi_file, sep="\t")
            ag_df = pd.read_csv(ag_file, sep="\t")
            
            # Standardize column names
            borzoi_df = borzoi_df.rename(columns={
                "chr": "chromosome",
                "pos": "position",
                "ref": "ref_allele", 
                "alt": "alt_allele",
            })
            
            # Create merge key
            borzoi_df["merge_key"] = (
                borzoi_df["chromosome"].astype(str) + "_" +
                borzoi_df["position"].astype(str) + "_" +
                borzoi_df["ref_allele"].astype(str) + "_" +
                borzoi_df["alt_allele"].astype(str) + "_" +
                borzoi_df["gene_id"].astype(str)
            )
            
            ag_df["merge_key"] = (
                ag_df["chromosome"].astype(str) + "_" +
                ag_df["position"].astype(str) + "_" +
                ag_df["ref_allele"].astype(str) + "_" +
                ag_df["alt_allele"].astype(str) + "_" +
                ag_df["gene_id"].astype(str)
            )
            
            # Get TSS bin from borzoi data
            tss_col = "tss_distance_bin" if "tss_distance_bin" in borzoi_df.columns else None
            
            # Select columns
            borzoi_cols = ["merge_key", "logMeanSED", "beta"]
            if tss_col:
                borzoi_cols.append(tss_col)
            borzoi_subset = borzoi_df[borzoi_cols].copy()
            borzoi_subset = borzoi_subset.rename(columns={"logMeanSED": "borzoi_score"})
            
            ag_cols = ag_df[["merge_key", "raw_score"]].copy()
            ag_cols = ag_cols.rename(columns={"raw_score": "alphagenome_score"})
            
            # Merge
            merged = borzoi_subset.merge(ag_cols, on="merge_key", how="inner")
            merged["ancestry"] = ancestry
            
            # Standardize TSS bin names
            if "tss_distance_bin" in merged.columns:
                merged["tss_distance_bin"] = merged["tss_distance_bin"].replace("gt35kb", ">35kb")
            
            all_dfs.append(merged)
            
        except Exception as e:
            print(f"  [WARN] Error processing finemapped {ancestry}: {e}")
    
    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        # Ensure numeric
        result["borzoi_score"] = pd.to_numeric(result["borzoi_score"], errors="coerce")
        result["alphagenome_score"] = pd.to_numeric(result["alphagenome_score"], errors="coerce")
        result["beta"] = pd.to_numeric(result["beta"], errors="coerce")
        print(f"  Loaded {len(result):,} paired fine-mapped variants")
        return result
    
    return pd.DataFrame()


# =============================================================================
# Analysis
# =============================================================================

def compute_model_convergence(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Compute correlation and concordance between Borzoi and AlphaGenome."""
    results = []
    
    valid = df.dropna(subset=["borzoi_score", "alphagenome_score"])
    
    if len(valid) < 10:
        return pd.DataFrame()
    
    # Overall
    r_spearman, p_spearman = spearmanr(valid["borzoi_score"], valid["alphagenome_score"])
    r_pearson, p_pearson = pearsonr(valid["borzoi_score"], valid["alphagenome_score"])
    
    # Concordance (same sign)
    bz_sign = np.sign(valid["borzoi_score"])
    ag_sign = np.sign(valid["alphagenome_score"])
    concordance = (bz_sign == ag_sign).mean()
    
    results.append({
        "dataset": dataset_name,
        "ancestry": "All",
        "tss_distance_bin": "All",
        "n": len(valid),
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "concordance": concordance,
    })
    
    # By ancestry
    for ancestry in ANCESTRIES:
        subset = valid[valid["ancestry"] == ancestry]
        if len(subset) < 10:
            continue
        
        r_sp, p_sp = spearmanr(subset["borzoi_score"], subset["alphagenome_score"])
        r_pe, p_pe = pearsonr(subset["borzoi_score"], subset["alphagenome_score"])
        conc = (np.sign(subset["borzoi_score"]) == np.sign(subset["alphagenome_score"])).mean()
        
        results.append({
            "dataset": dataset_name,
            "ancestry": ancestry,
            "tss_distance_bin": "All",
            "n": len(subset),
            "spearman_r": r_sp,
            "spearman_p": p_sp,
            "pearson_r": r_pe,
            "pearson_p": p_pe,
            "concordance": conc,
        })
    
    # By TSS bin (if available)
    if "tss_distance_bin" in valid.columns:
        for tss_bin in TSS_BINS:
            subset = valid[valid["tss_distance_bin"] == tss_bin]
            if len(subset) < 10:
                continue
            
            r_sp, p_sp = spearmanr(subset["borzoi_score"], subset["alphagenome_score"])
            r_pe, p_pe = pearsonr(subset["borzoi_score"], subset["alphagenome_score"])
            conc = (np.sign(subset["borzoi_score"]) == np.sign(subset["alphagenome_score"])).mean()
            
            results.append({
                "dataset": dataset_name,
                "ancestry": "All",
                "tss_distance_bin": tss_bin,
                "n": len(subset),
                "spearman_r": r_sp,
                "spearman_p": p_sp,
                "pearson_r": r_pe,
                "pearson_p": p_pe,
                "concordance": conc,
            })
    
    # By ancestry AND TSS bin
    if "tss_distance_bin" in valid.columns:
        for ancestry in ANCESTRIES:
            for tss_bin in TSS_BINS:
                subset = valid[(valid["ancestry"] == ancestry) & (valid["tss_distance_bin"] == tss_bin)]
                if len(subset) < 10:
                    continue
                
                r_sp, p_sp = spearmanr(subset["borzoi_score"], subset["alphagenome_score"])
                r_pe, p_pe = pearsonr(subset["borzoi_score"], subset["alphagenome_score"])
                conc = (np.sign(subset["borzoi_score"]) == np.sign(subset["alphagenome_score"])).mean()
                
                results.append({
                    "dataset": dataset_name,
                    "ancestry": ancestry,
                    "tss_distance_bin": tss_bin,
                    "n": len(subset),
                    "spearman_r": r_sp,
                    "spearman_p": p_sp,
                    "pearson_r": r_pe,
                    "pearson_p": p_pe,
                    "concordance": conc,
                })
    
    return pd.DataFrame(results)


# =============================================================================
# Plotting
# =============================================================================

def set_plot_style():
    """Set consistent plot style."""
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10


def plot_convergence_heatmaps(results_df: pd.DataFrame, output_dir: Path):
    """Plot heatmaps showing model convergence by ancestry and TSS bin."""
    set_plot_style()

    df = results_df[
        (results_df["ancestry"] != "All") &
        (results_df["tss_distance_bin"] != "All")
    ].copy()

    if len(df) == 0:
        print("[WARN] No stratified data for heatmaps")
        return

    datasets = [d for d in ["Nominal eQTLs", "Fine-mapped (PIP≥0.9)"] if d in df["dataset"].unique()]
    dataset_labels = {"Fine-mapped (PIP≥0.9)": "Fine-mapped (PIP ≥ 0.9)"}
    if not datasets:
        print("[WARN] No datasets found for heatmaps")
        return

    corr_vmin, corr_vmax = rounded_limits(df["spearman_r"])
    conc_vmin, conc_vmax = rounded_limits(df["concordance"])
    corr_ticks = rounded_ticks(corr_vmin, corr_vmax)
    conc_ticks = rounded_ticks(conc_vmin, conc_vmax)
    heatmap_cmap = plt.get_cmap("viridis").copy()
    heatmap_cmap.set_bad("#e6e6e6")

    fig = plt.figure(figsize=(9.0 if len(datasets) == 2 else 4.8, 8.0))
    main_gs = plt.GridSpec(
        2, len(datasets) + 1,
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
    right_axes = {}

    for col_idx, dataset in enumerate(datasets):
        dataset_df = df[df["dataset"] == dataset]

        # Correlation heatmap
        ax_top = fig.add_subplot(main_gs[0, col_idx])
        pivot_corr = dataset_df.pivot(
            index="tss_distance_bin",
            columns="ancestry",
            values="spearman_r"
        )
        pivot_corr = pivot_corr.reindex(index=TSS_BINS, columns=ANCESTRIES)

        ax_top.set_facecolor("#e6e6e6")
        sns.heatmap(
            pivot_corr,
            annot=True,
            fmt=".3f",
            cmap=heatmap_cmap,
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

        # Concordance heatmap
        ax_bottom = fig.add_subplot(main_gs[1, col_idx])
        pivot_conc = dataset_df.pivot(
            index="tss_distance_bin",
            columns="ancestry",
            values="concordance"
        )
        pivot_conc = pivot_conc.reindex(index=TSS_BINS, columns=ANCESTRIES)

        ax_bottom.set_facecolor("#e6e6e6")
        sns.heatmap(
            pivot_conc,
            annot=True,
            fmt=".3f",
            cmap=heatmap_cmap,
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
        right_axes[col_idx] = (ax_top, ax_bottom)

    fig.canvas.draw()

    if len(datasets) == 2:
        left_top, left_bottom = right_axes[0]
        right_top, right_bottom = right_axes[1]
        left_pos = left_top.get_position()
        right_pos = right_top.get_position()
        current_gap = right_pos.x0 - left_pos.x1
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
            f"({chr(65 + col_idx)}) {dataset_labels.get(dataset, dataset)}",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    if len(datasets) >= 1:
        for cax, ref_ax in [(corr_cbar_ax, top_axes[-1][1]), (conc_cbar_ax, right_axes[len(datasets) - 1][1])]:
            ref_pos = ref_ax.get_position()
            cax.set_position([ref_pos.x1 + 0.004, ref_pos.y0, 0.012, ref_pos.height])

    for cax, ticks in [(corr_cbar_ax, corr_ticks), (conc_cbar_ax, conc_ticks)]:
        cax.set_yticks(ticks)
        cax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        cax.tick_params(labelsize=9, length=3, width=0.8, pad=2)
        cax.yaxis.label.set_size(10)

    output_path = output_dir / "model_convergence_heatmaps.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_scatter_comparison(
    nominal_df: pd.DataFrame,
    finemapped_df: pd.DataFrame,
    output_dir: Path
):
    """Plot scatter plots comparing Borzoi vs AlphaGenome predictions."""
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = [
        ("Nominal eQTLs", nominal_df),
        ("Fine-mapped (PIP≥0.9)", finemapped_df),
    ]
    
    for idx, (name, df) in enumerate(datasets):
        ax = axes[idx]
        
        valid = df.dropna(subset=["borzoi_score", "alphagenome_score"])
        
        if len(valid) == 0:
            ax.set_visible(False)
            continue
        
        # Subsample for plotting
        if len(valid) > 5000:
            plot_df = valid.sample(n=5000, random_state=42)
        else:
            plot_df = valid
        
        ax.scatter(
            plot_df["borzoi_score"],
            plot_df["alphagenome_score"],
            alpha=0.3,
            s=10,
            c="#4DBBD5",
        )
        
        # Compute stats on full data
        r_sp, _ = spearmanr(valid["borzoi_score"], valid["alphagenome_score"])
        concordance = (np.sign(valid["borzoi_score"]) == np.sign(valid["alphagenome_score"])).mean()
        
        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel("Borzoi (logMeanSED)")
        ax.set_ylabel("AlphaGenome (raw_score)")
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        
        # Add stats text
        ax.text(
            0.05, 0.95,
            f"n = {len(valid):,}\nSpearman r = {r_sp:.3f}\nConcordance = {concordance:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    
    fig.suptitle("Model Convergence: Borzoi vs AlphaGenome Predictions", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_dir / "model_convergence_scatter.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_convergence_summary_bar(results_df: pd.DataFrame, output_dir: Path):
    """Plot bar chart summary of convergence by dataset and ancestry."""
    set_plot_style()
    
    # Filter to ancestry-level results (TSS = All)
    df = results_df[
        (results_df["tss_distance_bin"] == "All") &
        (results_df["ancestry"] != "All")
    ].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correlation bar plot
    ax = axes[0]
    x = np.arange(len(ANCESTRIES))
    width = 0.35
    
    for idx, dataset in enumerate(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].set_index("ancestry")
        vals = [dataset_df.loc[a, "spearman_r"] if a in dataset_df.index else 0 for a in ANCESTRIES]
        offset = (idx - 0.5) * width
        color = "#7F7F7F" if "Nominal" in dataset else "#E64B35"
        ax.bar(x + offset, vals, width, label=dataset, color=color)
    
    ax.set_ylabel("Spearman r")
    ax.set_xlabel("Ancestry")
    ax.set_xticks(x)
    ax.set_xticklabels([ANCESTRY_LABELS[a] for a in ANCESTRIES])
    ax.set_title("Correlation: Borzoi vs AlphaGenome", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Concordance bar plot
    ax = axes[1]
    for idx, dataset in enumerate(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].set_index("ancestry")
        vals = [dataset_df.loc[a, "concordance"] if a in dataset_df.index else 0 for a in ANCESTRIES]
        offset = (idx - 0.5) * width
        color = "#7F7F7F" if "Nominal" in dataset else "#E64B35"
        ax.bar(x + offset, vals, width, label=dataset, color=color)
    
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("Concordance (Direction Agreement)")
    ax.set_xlabel("Ancestry")
    ax.set_xticks(x)
    ax.set_xticklabels([ANCESTRY_LABELS[a] for a in ANCESTRIES])
    ax.set_title("Direction Concordance: Borzoi vs AlphaGenome", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_ylim([0.4, 0.8])
    
    fig.suptitle("Model Convergence Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_dir / "model_convergence_summary.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze model convergence")
    parser.add_argument("--base_dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    
    base_dir = args.base_dir.resolve()
    output_dir = base_dir / "outputs_finemapped" / "analysis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Model Convergence Analysis")
    print("Comparing Borzoi vs AlphaGenome Predictions")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading nominal eQTL data...")
    nominal_df = load_nominal_paired_data(base_dir)
    
    print("\n[2] Loading fine-mapped eQTL data (PIP ≥ 0.9)...")
    finemapped_df = load_finemapped_paired_data(base_dir)
    
    # Compute convergence metrics
    print("\n[3] Computing convergence metrics...")
    
    results_list = []
    
    if len(nominal_df) > 0:
        nominal_results = compute_model_convergence(nominal_df, "Nominal eQTLs")
        results_list.append(nominal_results)
        print("\n  Nominal eQTLs (Overall):")
        overall = nominal_results[nominal_results["ancestry"] == "All"].iloc[0]
        print(f"    Spearman r = {overall['spearman_r']:.4f}")
        print(f"    Concordance = {overall['concordance']:.4f}")
    
    if len(finemapped_df) > 0:
        finemapped_results = compute_model_convergence(finemapped_df, "Fine-mapped (PIP≥0.9)")
        results_list.append(finemapped_results)
        print("\n  Fine-mapped (PIP≥0.9) (Overall):")
        overall = finemapped_results[finemapped_results["ancestry"] == "All"].iloc[0]
        print(f"    Spearman r = {overall['spearman_r']:.4f}")
        print(f"    Concordance = {overall['concordance']:.4f}")
    
    # Combine results
    all_results = pd.concat(results_list, ignore_index=True)
    
    # Save results
    results_path = output_dir.parent / "model_convergence_results.tsv"
    all_results.to_csv(results_path, sep="\t", index=False)
    print(f"\n[OK] Saved results: {results_path}")
    
    # Generate plots
    print("\n[4] Generating plots...")
    plot_scatter_comparison(nominal_df, finemapped_df, output_dir)
    plot_convergence_summary_bar(all_results, output_dir)
    
    if len(all_results[all_results["tss_distance_bin"] != "All"]) > 0:
        plot_convergence_heatmaps(all_results, output_dir)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Model Convergence")
    print("=" * 70)
    
    summary = all_results[all_results["tss_distance_bin"] == "All"][
        ["dataset", "ancestry", "n", "spearman_r", "concordance"]
    ].round(4)
    print(summary.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
