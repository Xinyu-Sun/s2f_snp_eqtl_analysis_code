#!/usr/bin/env python3
"""
Negative-class contamination analysis for the S2F manuscript.

This script:
1. Loads the existing scored SuSiE universe from outputs_susie/analysis/susie_s2f_combined.tsv.gz
2. Collapses to unique ancestry x gene locus x variant rows for LD contamination analysis
3. Defines the anchor variant as the top-PIP variant within each ancestry x gene locus
4. Computes genotype LD (r^2) from ancestry-specific PLINK BED/BIM/FAM files by chromosome
5. Summarizes contamination among low-PIP negatives, with manuscript summaries focused on
   loci whose top anchor PIP is at least 0.9
6. Merges contamination metrics back to model-specific scored rows
7. Reruns the distance-matched bootstrap AUROC analysis with progressively cleaner negatives:
   - PIP < 0.01
   - PIP < 0.01 and anchor r^2 < 0.2
   - PIP < 0.01 and anchor r^2 < 0.1

Assumption:
- LD is computed for all loci with an observed anchor variant so the cleaned AUROC analysis
  can reuse the full scored universe. The ancestry summary/contamination figure are reported
  on the requested high-confidence subset with top anchor PIP >= 0.9.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bed_reader import open_bed
from sklearn.metrics import auc, roc_curve


ANCESTRIES = ["AA", "CH", "NHW"]
TOOLS = ["borzoi", "alphagenome"]
PIP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
NEGATIVE_PIP_THRESHOLD = 0.01
SUMMARY_TOP_PIP_THRESHOLD = 0.9
N_DISTANCE_BINS = 10
N_BOOTSTRAP = 100

TOOL_CONFIG = {
    "borzoi": {
        "pred_col": "logMeanSED",
        "display_name": "Borzoi",
        "color": "#E64B35",
    },
    "alphagenome": {
        "pred_col": "raw_score",
        "display_name": "AlphaGenome",
        "color": "#4DBBD5",
    },
}

ANCESTRY_LABELS = {
    "AA": "African American",
    "CH": "Caribbean Hispanic",
    "NHW": "Non-Hispanic White",
}

NEGATIVE_DEFS = {
    "pip_lt_0.01": {
        "label": "PIP < 0.01",
        "color": "#6C757D",
    },
    "pip_lt_0.01_ld_lt_0.2": {
        "label": "PIP < 0.01 & r^2 < 0.2",
        "color": "#2A9D8F",
    },
    "pip_lt_0.01_ld_lt_0.1": {
        "label": "PIP < 0.01 & r^2 < 0.1",
        "color": "#264653",
    },
}

GENOTYPE_BASE = Path(
    "/mnt/vstor/Data14/xxs410_xinyu/SuShiE_work/magenta_geno_QC/"
    "_EUR_1kg_ref_highcoverage_comm_SNP_plink_MAF_01_4anc_s2"
)
GENOTYPE_GROUPS = {
    "AA": ["AFR"],
    "NHW": ["EUR"],
    "CH": ["HISP_IHG", "HISP_PER"],
}


def normalize_chrom(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("^chr", "", regex=True)
    return "chr" + s


def build_variant_key(
    chrom: pd.Series,
    pos: pd.Series,
    ref: pd.Series,
    alt: pd.Series,
    gene_id: pd.Series,
    ancestry: pd.Series,
) -> pd.Series:
    return (
        normalize_chrom(chrom)
        + "_"
        + pd.to_numeric(pos, errors="coerce").astype("Int64").astype(str)
        + "_"
        + ref.astype(str)
        + "_"
        + alt.astype(str)
        + "_"
        + gene_id.astype(str)
        + "_"
        + ancestry.astype(str)
    )


def build_plink_variant_id(chrom: pd.Series, pos: pd.Series, ref: pd.Series, alt: pd.Series) -> pd.Series:
    return (
        normalize_chrom(chrom)
        + ":"
        + pd.to_numeric(pos, errors="coerce").astype("Int64").astype(str)
        + "_"
        + ref.astype(str)
        + "_"
        + alt.astype(str)
    )


def chromosome_sort_key(chrom: str) -> tuple[int, str]:
    chrom = str(chrom).replace("chr", "")
    try:
        return (0, f"{int(chrom):02d}")
    except ValueError:
        return (1, chrom)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def squared_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 3:
        return np.nan

    x = x[mask].astype(float)
    y = y[mask].astype(float)
    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom == 0:
        return np.nan

    r = float(np.dot(x, y) / denom)
    return r * r


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_scored_susie(base_dir: Path) -> pd.DataFrame:
    combined_file = base_dir / "outputs_susie" / "analysis" / "susie_s2f_combined.tsv.gz"
    df = pd.read_csv(combined_file, sep="\t", low_memory=False)

    df = df[
        df["ancestry"].isin(ANCESTRIES)
        & df["tool"].isin(TOOLS)
    ].copy()

    for col in ["pip", "logMeanSED", "raw_score", "pos", "position"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    df["chrom_u"] = normalize_chrom(df["chromosome"].fillna(df["chr"]))
    df["pos_u"] = df["position"].fillna(df["pos"]).astype("Int64")
    df["ref_u"] = df["ref_allele"].fillna(df["ref"]).astype(str)
    df["alt_u"] = df["alt_allele"].fillna(df["alt"]).astype(str)
    df["variant_pos"] = safe_numeric(df["pos_u"])
    df["variant_locus_key"] = build_variant_key(
        df["chrom_u"],
        df["pos_u"],
        df["ref_u"],
        df["alt_u"],
        df["gene_id"],
        df["ancestry"],
    )
    df["plink_variant_id"] = build_plink_variant_id(df["chrom_u"], df["pos_u"], df["ref_u"], df["alt_u"])
    df["pred_score"] = np.where(
        df["tool"] == "borzoi",
        safe_numeric(df["logMeanSED"]),
        safe_numeric(df["raw_score"]),
    )

    return df


def load_unique_variant_locus_rows(scored_df: pd.DataFrame) -> pd.DataFrame:
    uniq = (
        scored_df[
            [
                "variant_locus_key",
                "ancestry",
                "gene_id",
                "chrom_u",
                "pos_u",
                "ref_u",
                "alt_u",
                "pip",
                "tss_distance_bin",
                "plink_variant_id",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    uniq = uniq.rename(
        columns={
            "chrom_u": "chromosome",
            "pos_u": "position",
            "ref_u": "ref",
            "alt_u": "alt",
        }
    )

    uniq = (
        uniq.sort_values(
            ["ancestry", "gene_id", "chromosome", "position", "ref", "alt", "pip"],
            ascending=[True, True, True, True, True, True, False],
        )
        .drop_duplicates(subset=["variant_locus_key"], keep="first")
        .reset_index(drop=True)
    )
    uniq["locus_key"] = uniq["ancestry"] + "|" + uniq["gene_id"] + "|" + uniq["chromosome"]
    uniq["position"] = uniq["position"].astype("Int64")
    uniq["pip"] = safe_numeric(uniq["pip"])
    return uniq


def choose_anchor_variants(variant_df: pd.DataFrame) -> pd.DataFrame:
    anchors = (
        variant_df.sort_values(
            ["locus_key", "pip", "position", "ref", "alt"],
            ascending=[True, False, True, True, True],
        )
        .drop_duplicates(subset=["locus_key"], keep="first")
        .copy()
    )
    anchors = anchors.rename(
        columns={
            "variant_locus_key": "anchor_variant_locus_key",
            "plink_variant_id": "anchor_plink_variant_id",
            "position": "anchor_position",
            "ref": "anchor_ref",
            "alt": "anchor_alt",
            "pip": "anchor_top_pip",
        }
    )
    return anchors[
        [
            "locus_key",
            "ancestry",
            "gene_id",
            "chromosome",
            "anchor_variant_locus_key",
            "anchor_plink_variant_id",
            "anchor_position",
            "anchor_ref",
            "anchor_alt",
            "anchor_top_pip",
        ]
    ]


def prepare_negative_ld_candidates(variant_df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    negatives = variant_df[variant_df["pip"] < NEGATIVE_PIP_THRESHOLD].copy()
    negatives = negatives.merge(
        anchors,
        on=["locus_key", "ancestry", "gene_id", "chromosome"],
        how="left",
    )
    negatives = negatives[negatives["variant_locus_key"] != negatives["anchor_variant_locus_key"]].copy()
    negatives["anchor_top_pip_ge_0_9"] = negatives["anchor_top_pip"] >= SUMMARY_TOP_PIP_THRESHOLD
    return negatives


@lru_cache(maxsize=None)
def load_gene_tss(base_dir: str) -> pd.DataFrame:
    gene_loc_file = Path(base_dir) / "inputs" / "hg38_gene_locations.txt"
    gene_tss = pd.read_csv(gene_loc_file, sep="\t")
    gene_tss = gene_tss.rename(columns={"ensgid": "gene_id", "TSS": "tss"})
    return gene_tss[["gene_id", "tss"]].drop_duplicates()


@lru_cache(maxsize=None)
def get_group_bim(group: str, chrom: str) -> pd.DataFrame:
    bim_path = GENOTYPE_BASE / f"{group}_{chrom}_common.bim"
    return pd.read_csv(
        bim_path,
        sep="\t",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
    )


@lru_cache(maxsize=None)
def get_group_variant_index(group: str, chrom: str) -> dict[str, int]:
    bim = get_group_bim(group, chrom)
    return {snp: idx for idx, snp in enumerate(bim["snp"].astype(str).tolist())}


@lru_cache(maxsize=None)
def get_group_bed(group: str, chrom: str):
    return open_bed(str(GENOTYPE_BASE / f"{group}_{chrom}_common.bed"))


def load_chromosome_genotypes(ancestry: str, chrom: str, variant_ids: Iterable[str]) -> dict[str, np.ndarray]:
    requested = list(dict.fromkeys(str(v) for v in variant_ids if pd.notna(v)))
    if not requested:
        return {}

    groups = GENOTYPE_GROUPS[ancestry]
    common_variant_ids = requested.copy()
    for group in groups:
        index_map = get_group_variant_index(group, chrom)
        common_variant_ids = [vid for vid in common_variant_ids if vid in index_map]

    if not common_variant_ids:
        return {}

    arrays = []
    for group in groups:
        index_map = get_group_variant_index(group, chrom)
        bed = get_group_bed(group, chrom)
        indices = [index_map[vid] for vid in common_variant_ids]
        arrays.append(bed.read(index=(slice(None), indices)))

    genotype_matrix = np.concatenate(arrays, axis=0)
    return {
        vid: genotype_matrix[:, col_idx]
        for col_idx, vid in enumerate(common_variant_ids)
    }


def compute_negative_anchor_ld(negative_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for ancestry in ANCESTRIES:
        neg_anc = negative_df[negative_df["ancestry"] == ancestry].copy()
        if neg_anc.empty:
            continue

        for chrom in sorted(neg_anc["chromosome"].dropna().unique(), key=chromosome_sort_key):
            neg_chr = neg_anc[neg_anc["chromosome"] == chrom].copy()
            if neg_chr.empty:
                continue

            variant_ids = pd.concat(
                [
                    neg_chr["plink_variant_id"],
                    neg_chr["anchor_plink_variant_id"],
                ],
                ignore_index=True,
            ).dropna().unique()
            genotype_map = load_chromosome_genotypes(ancestry, chrom, variant_ids)

            for row in neg_chr.itertuples(index=False):
                anchor_vec = genotype_map.get(row.anchor_plink_variant_id)
                neg_vec = genotype_map.get(row.plink_variant_id)
                anchor_in_plink = anchor_vec is not None
                negative_in_plink = neg_vec is not None
                ld_available = anchor_in_plink and negative_in_plink

                results.append(
                    {
                        "variant_locus_key": row.variant_locus_key,
                        "locus_key": row.locus_key,
                        "anchor_r2": squared_corr(anchor_vec, neg_vec) if ld_available else np.nan,
                        "anchor_in_plink": anchor_in_plink,
                        "negative_in_plink": negative_in_plink,
                        "ld_available": ld_available,
                    }
                )

    ld_df = pd.DataFrame(results)
    if ld_df.empty:
        return ld_df
    return ld_df.drop_duplicates(subset=["variant_locus_key"])


def build_variant_level_contamination_table(negative_df: pd.DataFrame, ld_df: pd.DataFrame) -> pd.DataFrame:
    variant_level = negative_df.merge(ld_df, on=["variant_locus_key", "locus_key"], how="left")

    variant_level["flag_r2_gt_0_2"] = variant_level["anchor_r2"] > 0.2
    variant_level["flag_r2_gt_0_5"] = variant_level["anchor_r2"] > 0.5
    variant_level["flag_r2_gt_0_8"] = variant_level["anchor_r2"] > 0.8
    variant_level["flag_r2_lt_0_2"] = variant_level["anchor_r2"] < 0.2
    variant_level["flag_r2_lt_0_1"] = variant_level["anchor_r2"] < 0.1

    cols = [
        "variant_locus_key",
        "locus_key",
        "ancestry",
        "gene_id",
        "chromosome",
        "position",
        "ref",
        "alt",
        "pip",
        "plink_variant_id",
        "anchor_variant_locus_key",
        "anchor_plink_variant_id",
        "anchor_position",
        "anchor_ref",
        "anchor_alt",
        "anchor_top_pip",
        "anchor_top_pip_ge_0_9",
        "anchor_r2",
        "anchor_in_plink",
        "negative_in_plink",
        "ld_available",
        "flag_r2_gt_0_2",
        "flag_r2_gt_0_5",
        "flag_r2_gt_0_8",
        "flag_r2_lt_0_2",
        "flag_r2_lt_0_1",
    ]
    return variant_level[cols].sort_values(["ancestry", "gene_id", "position"]).reset_index(drop=True)


def summarize_contamination_by_ancestry(variant_level: pd.DataFrame) -> pd.DataFrame:
    subset = variant_level[variant_level["anchor_top_pip_ge_0_9"]].copy()

    rows = []
    for ancestry in ANCESTRIES:
        anc = subset[subset["ancestry"] == ancestry].copy()
        ld = anc[anc["ld_available"] & anc["anchor_r2"].notna()].copy()

        rows.append(
            {
                "ancestry": ancestry,
                "ancestry_label": ANCESTRY_LABELS[ancestry],
                "summary_scope": "top_anchor_pip_ge_0.9",
                "n_loci": int(anc["locus_key"].nunique()),
                "n_negative_variants_total": int(len(anc)),
                "n_negative_variants_with_ld": int(len(ld)),
                "n_negative_variants_missing_ld": int(len(anc) - len(ld)),
                "r2_mean": float(ld["anchor_r2"].mean()) if not ld.empty else np.nan,
                "r2_median": float(ld["anchor_r2"].median()) if not ld.empty else np.nan,
                "r2_q25": float(ld["anchor_r2"].quantile(0.25)) if not ld.empty else np.nan,
                "r2_q75": float(ld["anchor_r2"].quantile(0.75)) if not ld.empty else np.nan,
                "frac_r2_gt_0_2": float((ld["anchor_r2"] > 0.2).mean()) if not ld.empty else np.nan,
                "frac_r2_gt_0_5": float((ld["anchor_r2"] > 0.5).mean()) if not ld.empty else np.nan,
                "frac_r2_gt_0_8": float((ld["anchor_r2"] > 0.8).mean()) if not ld.empty else np.nan,
                "frac_r2_lt_0_2": float((ld["anchor_r2"] < 0.2).mean()) if not ld.empty else np.nan,
                "frac_r2_lt_0_1": float((ld["anchor_r2"] < 0.1).mean()) if not ld.empty else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_contamination_summary(variant_level: pd.DataFrame, output_dir: Path) -> None:
    subset = variant_level[
        variant_level["anchor_top_pip_ge_0_9"]
        & variant_level["ld_available"]
        & variant_level["anchor_r2"].notna()
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    ancestry_colors = {"AA": "#E64B35", "CH": "#4DBBD5", "NHW": "#00A087"}
    for ancestry in ANCESTRIES:
        anc = subset[subset["ancestry"] == ancestry]["anchor_r2"].sort_values().to_numpy()
        if anc.size == 0:
            continue
        y = np.arange(1, anc.size + 1) / anc.size
        ax.step(anc, y, where="post", linewidth=2.2, color=ancestry_colors[ancestry], label=ANCESTRY_LABELS[ancestry])

    for thr in [0.2, 0.5, 0.8]:
        ax.axvline(thr, color="#666666", linestyle="--", linewidth=1)

    ax.set_xlabel("Anchor LD to negative variant (r^2)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Negative-class anchor LD in top-anchor-PIP >= 0.9 loci")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    threshold_labels = ["r^2 > 0.2", "r^2 > 0.5", "r^2 > 0.8"]
    x = np.arange(len(threshold_labels))
    width = 0.22
    offsets = {"AA": -width, "CH": 0.0, "NHW": width}
    for ancestry in ANCESTRIES:
        anc = subset[subset["ancestry"] == ancestry]["anchor_r2"]
        if anc.empty:
            values = [np.nan, np.nan, np.nan]
        else:
            values = [
                float((anc > 0.2).mean()),
                float((anc > 0.5).mean()),
                float((anc > 0.8).mean()),
            ]
        ax.bar(
            x + offsets[ancestry],
            values,
            width=width,
            color=ancestry_colors[ancestry],
            alpha=0.9,
            label=ANCESTRY_LABELS[ancestry],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of negatives")
    ax.set_title("Contamination fractions in high-confidence loci")

    fig.suptitle("Negative-class contamination from anchor LD", fontsize=14, fontweight="bold")
    fig.tight_layout()

    png_path = output_dir / "negative_ld_contamination_summary.png"
    pdf_path = output_dir / "negative_ld_contamination_summary.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=250)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def add_tss_distance(scored_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    gene_tss = load_gene_tss(str(base_dir))
    df = scored_df.merge(gene_tss, on="gene_id", how="left")
    df["tss_distance"] = np.abs(df["variant_pos"] - df["tss"])
    df = df[df["tss_distance"].notna() & (df["tss_distance"] > 0)].copy()
    df["log_distance"] = np.log10(df["tss_distance"])
    return df


def merge_contamination_to_scored_rows(scored_df: pd.DataFrame, variant_level: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        "variant_locus_key",
        "anchor_top_pip",
        "anchor_top_pip_ge_0_9",
        "anchor_r2",
        "anchor_in_plink",
        "negative_in_plink",
        "ld_available",
        "flag_r2_lt_0_2",
        "flag_r2_lt_0_1",
    ]
    return scored_df.merge(variant_level[merge_cols], on="variant_locus_key", how="left")


def balance_by_distance_bootstrap(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    random_state: int,
    n_bins: int = N_DISTANCE_BINS,
) -> pd.DataFrame:
    if positives.empty or negatives.empty:
        return pd.DataFrame()

    all_distances = pd.concat([positives["log_distance"], negatives["log_distance"]], ignore_index=True).dropna()
    if all_distances.empty:
        return pd.DataFrame()

    bin_edges = np.percentile(all_distances, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return pd.DataFrame()

    positives = positives.copy()
    negatives = negatives.copy()
    positives["distance_bin"] = pd.cut(
        positives["log_distance"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    negatives["distance_bin"] = pd.cut(
        negatives["log_distance"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )

    rng = np.random.RandomState(random_state)
    balanced = []

    bin_labels = sorted(pd.concat([positives["distance_bin"], negatives["distance_bin"]]).dropna().unique())
    for bin_idx in bin_labels:
        pos_bin = positives[positives["distance_bin"] == bin_idx]
        neg_bin = negatives[negatives["distance_bin"] == bin_idx]
        if pos_bin.empty or neg_bin.empty:
            continue

        n_pos = len(pos_bin)
        neg_sampled = neg_bin.sample(n=n_pos, replace=True, random_state=rng)
        balanced.append(pos_bin)
        balanced.append(neg_sampled)

    if not balanced:
        return pd.DataFrame()

    return pd.concat(balanced, ignore_index=True)


def _single_bootstrap_auc(payload: tuple[pd.DataFrame, pd.DataFrame, str, float, int]) -> dict | None:
    positives, negatives, pred_col, pip_threshold, random_state = payload

    balanced = balance_by_distance_bootstrap(
        positives=positives,
        negatives=negatives,
        random_state=random_state,
    )
    if len(balanced) < 10:
        return None

    balanced = balanced.dropna(subset=[pred_col, "pip"])
    if len(balanced) < 10:
        return None

    y_true = (balanced["pip"] >= pip_threshold).astype(int).to_numpy()
    y_score = np.abs(pd.to_numeric(balanced[pred_col], errors="coerce").to_numpy())
    mask = ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) < 10 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "auc": float(auc(fpr, tpr)),
        "fpr": fpr,
        "tpr": tpr,
        "n_pos_balanced_example": int(y_true.sum()),
        "n_neg_balanced_example": int(len(y_true) - y_true.sum()),
    }


def compute_bootstrap_auroc(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    pred_col: str,
    pip_threshold: float,
    n_bootstrap: int,
    seed: int,
    n_workers: int,
) -> dict | None:
    positives = positives[positives["pip"] >= pip_threshold].copy()

    n_pos_raw = len(positives)
    n_neg_raw = len(negatives)
    if n_pos_raw == 0 or n_neg_raw == 0:
        return None

    worker_inputs = [
        (
            positives,
            negatives,
            pred_col,
            pip_threshold,
            seed + bootstrap_idx,
        )
        for bootstrap_idx in range(n_bootstrap)
    ]

    if n_workers <= 1:
        bootstrap_results = [_single_bootstrap_auc(payload) for payload in worker_inputs]
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, n_bootstrap)) as executor:
            bootstrap_results = list(executor.map(_single_bootstrap_auc, worker_inputs))

    auc_values = [res["auc"] for res in bootstrap_results if res is not None]
    balanced_example = next((res for res in bootstrap_results if res is not None), None)

    if len(auc_values) < 5 or balanced_example is None:
        return None

    auc_array = np.array(auc_values, dtype=float)
    fpr = balanced_example["fpr"]
    tpr = balanced_example["tpr"]
    n_pos_bal = balanced_example["n_pos_balanced_example"]
    n_neg_bal = balanced_example["n_neg_balanced_example"]
    return {
        "auc_mean": float(np.mean(auc_array)),
        "auc_std": float(np.std(auc_array)),
        "auc_ci_low": float(np.percentile(auc_array, 2.5)),
        "auc_ci_high": float(np.percentile(auc_array, 97.5)),
        "n_bootstrap": int(len(auc_array)),
        "n_pos_raw": int(n_pos_raw),
        "n_neg_raw": int(n_neg_raw),
        "n_pos_balanced_example": int(n_pos_bal),
        "n_neg_balanced_example": int(n_neg_bal),
        "fpr": fpr,
        "tpr": tpr,
        "auc_values": auc_array,
    }


def build_negative_subset(scored_df: pd.DataFrame, negative_def: str) -> pd.DataFrame:
    negatives = scored_df[scored_df["pip"] < NEGATIVE_PIP_THRESHOLD].copy()
    if negative_def == "pip_lt_0.01":
        return negatives
    if negative_def == "pip_lt_0.01_ld_lt_0.2":
        return negatives[negatives["ld_available"].fillna(False) & (negatives["anchor_r2"] < 0.2)].copy()
    if negative_def == "pip_lt_0.01_ld_lt_0.1":
        return negatives[negatives["ld_available"].fillna(False) & (negatives["anchor_r2"] < 0.1)].copy()
    raise ValueError(f"Unknown negative definition: {negative_def}")


def run_auroc_sensitivity(
    scored_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    n_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    bootstrap_rows = []

    for tool in TOOLS:
        tool_df = scored_df[scored_df["tool"] == tool].copy()
        pred_col = TOOL_CONFIG[tool]["pred_col"]

        for ancestry in ANCESTRIES:
            anc_df = tool_df[tool_df["ancestry"] == ancestry].copy()
            if anc_df.empty:
                continue

            positives_base = anc_df[anc_df["pip"].notna()].copy()
            for negative_def in NEGATIVE_DEFS:
                negatives = build_negative_subset(anc_df, negative_def)

                for pip_threshold in PIP_THRESHOLDS:
                    result = compute_bootstrap_auroc(
                        positives=positives_base,
                        negatives=negatives,
                        pred_col=pred_col,
                        pip_threshold=pip_threshold,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                        n_workers=n_workers,
                    )

                    row = {
                        "tool": tool,
                        "ancestry": ancestry,
                        "ancestry_label": ANCESTRY_LABELS[ancestry],
                        "pip_threshold": pip_threshold,
                        "negative_definition": negative_def,
                        "negative_definition_label": NEGATIVE_DEFS[negative_def]["label"],
                    }

                    if result is None:
                        row.update(
                            {
                                "auc_mean": np.nan,
                                "auc_std": np.nan,
                                "auc_ci_low": np.nan,
                                "auc_ci_high": np.nan,
                                "n_bootstrap": 0,
                                "n_pos_raw": int((positives_base["pip"] >= pip_threshold).sum()),
                                "n_neg_raw": int(len(negatives)),
                                "n_pos_balanced_example": np.nan,
                                "n_neg_balanced_example": np.nan,
                            }
                        )
                        summary_rows.append(row)
                        continue

                    row.update(
                        {
                            "auc_mean": result["auc_mean"],
                            "auc_std": result["auc_std"],
                            "auc_ci_low": result["auc_ci_low"],
                            "auc_ci_high": result["auc_ci_high"],
                            "n_bootstrap": result["n_bootstrap"],
                            "n_pos_raw": result["n_pos_raw"],
                            "n_neg_raw": result["n_neg_raw"],
                            "n_pos_balanced_example": result["n_pos_balanced_example"],
                            "n_neg_balanced_example": result["n_neg_balanced_example"],
                        }
                    )
                    summary_rows.append(row)

                    for auc_value in result["auc_values"]:
                        bootstrap_rows.append(
                            {
                                "tool": tool,
                                "ancestry": ancestry,
                                "pip_threshold": pip_threshold,
                                "negative_definition": negative_def,
                                "auc_value": float(auc_value),
                            }
                        )

    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows)


def add_auc_delta_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline = summary_df[summary_df["negative_definition"] == "pip_lt_0.01"][
        ["tool", "ancestry", "pip_threshold", "auc_mean"]
    ].rename(columns={"auc_mean": "auc_mean_original"})

    out = summary_df.merge(baseline, on=["tool", "ancestry", "pip_threshold"], how="left")
    out["auc_delta_vs_original"] = out["auc_mean"] - out["auc_mean_original"]
    return out


def compute_gap_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    focus = summary_df[summary_df["pip_threshold"] == 0.9].copy()
    aa = focus[focus["ancestry"] == "AA"][
        ["tool", "negative_definition", "auc_mean"]
    ].rename(columns={"auc_mean": "auc_mean_aa"})
    gaps = focus.merge(aa, on=["tool", "negative_definition"], how="left")
    gaps["gap_vs_aa"] = gaps["auc_mean_aa"] - gaps["auc_mean"]
    return gaps.sort_values(["tool", "negative_definition", "ancestry"]).reset_index(drop=True)


def plot_auroc_sensitivity(summary_df: pd.DataFrame, output_dir: Path) -> None:
    focus = summary_df[summary_df["pip_threshold"] == 0.9].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex="col")
    ancestry_colors = {"AA": "#E64B35", "CH": "#4DBBD5", "NHW": "#00A087"}

    x = np.arange(len(NEGATIVE_DEFS))
    neg_keys = list(NEGATIVE_DEFS)
    tick_labels = [NEGATIVE_DEFS[key]["label"] for key in neg_keys]

    for row_idx, tool in enumerate(TOOLS):
        tool_focus = focus[focus["tool"] == tool].copy()
        display_name = TOOL_CONFIG[tool]["display_name"]

        ax_abs = axes[row_idx, 0]
        ax_delta = axes[row_idx, 1]

        for ancestry in ANCESTRIES:
            anc = (
                tool_focus[tool_focus["ancestry"] == ancestry]
                .set_index("negative_definition")
                .reindex(neg_keys)
                .reset_index()
            )
            y = anc["auc_mean"].to_numpy(dtype=float)
            yerr_low = y - anc["auc_ci_low"].to_numpy(dtype=float)
            yerr_high = anc["auc_ci_high"].to_numpy(dtype=float) - y

            ax_abs.errorbar(
                x,
                y,
                yerr=[yerr_low, yerr_high],
                color=ancestry_colors[ancestry],
                marker="o",
                linewidth=2,
                capsize=3,
                label=ANCESTRY_LABELS[ancestry],
            )

            delta = anc["auc_delta_vs_original"].to_numpy(dtype=float)
            ax_delta.plot(
                x,
                delta,
                color=ancestry_colors[ancestry],
                marker="o",
                linewidth=2,
                label=ANCESTRY_LABELS[ancestry],
            )

        ax_abs.set_title(f"{display_name}: AUROC at PIP >= 0.9", fontsize=12, fontweight="bold")
        ax_abs.set_ylabel("Bootstrap AUROC")
        ax_abs.set_ylim(0.35, 1.0)
        ax_abs.grid(alpha=0.25)

        ax_delta.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
        ax_delta.set_title(f"{display_name}: Change vs original negatives", fontsize=12, fontweight="bold")
        ax_delta.set_ylabel("Delta AUROC")
        ax_delta.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=18, ha="right")

    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Distance-matched bootstrap AUROC sensitivity to negative-set cleaning",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    png_path = output_dir / "auroc_negative_cleaning_sensitivity_pip0.9.png"
    pdf_path = output_dir / "auroc_negative_cleaning_sensitivity_pip0.9.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=250)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def build_interpretation_text(summary_df: pd.DataFrame, gap_df: pd.DataFrame) -> str:
    focus = summary_df[summary_df["pip_threshold"] == 0.9].copy()
    focus = focus[
        focus["negative_definition"].isin(["pip_lt_0.01", "pip_lt_0.01_ld_lt_0.1"])
    ].copy()

    pivot = (
        focus.pivot_table(
            index=["tool", "ancestry"],
            columns="negative_definition",
            values="auc_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot["delta_ld_lt_0.1"] = pivot["pip_lt_0.01_ld_lt_0.1"] - pivot["pip_lt_0.01"]

    mean_delta = pivot.groupby("ancestry")["delta_ld_lt_0.1"].mean()
    nhw_vs_aa = mean_delta.get("NHW", np.nan) - mean_delta.get("AA", np.nan)
    ch_vs_aa = mean_delta.get("CH", np.nan) - mean_delta.get("AA", np.nan)

    gap_focus = gap_df[
        gap_df["negative_definition"].isin(["pip_lt_0.01", "pip_lt_0.01_ld_lt_0.1"])
        & gap_df["ancestry"].isin(["CH", "NHW"])
    ].copy()
    gap_pivot = (
        gap_focus.pivot_table(
            index=["tool", "ancestry"],
            columns="negative_definition",
            values="gap_vs_aa",
            aggfunc="first",
        )
        .reset_index()
    )
    gap_pivot["gap_change"] = gap_pivot["pip_lt_0.01_ld_lt_0.1"] - gap_pivot["pip_lt_0.01"]

    lines = []
    if not np.isnan(nhw_vs_aa) and nhw_vs_aa > 0:
        lines.append("NHW gains more AUROC after LD cleaning than AA on average at PIP >= 0.9.")
    else:
        lines.append("NHW does not gain more AUROC than AA on average at PIP >= 0.9.")

    if not np.isnan(ch_vs_aa) and ch_vs_aa > 0:
        lines.append("CH also gains more AUROC than AA on average after the stricter LD < 0.1 cleaning.")
    else:
        lines.append("CH does not consistently gain more AUROC than AA after the stricter LD < 0.1 cleaning.")

    narrowed = gap_pivot.groupby("ancestry")["gap_change"].mean()
    nhw_gap_change = narrowed.get("NHW", np.nan)
    ch_gap_change = narrowed.get("CH", np.nan)
    if not np.isnan(nhw_gap_change):
        direction = "narrows" if nhw_gap_change < 0 else "widens"
        lines.append(f"The NHW-vs-AA gap {direction} on average after cleaning.")
    if not np.isnan(ch_gap_change):
        direction = "narrows" if ch_gap_change < 0 else "widens"
        lines.append(f"The CH-vs-AA gap {direction} on average after cleaning.")

    lines.append(
        "Interpret these shifts together with the contamination table counts, because only negatives with measured anchor LD can enter the cleaned sets."
    )
    return "\n".join(lines)


def write_outputs(
    variant_level: pd.DataFrame,
    ancestry_summary: pd.DataFrame,
    auroc_summary: pd.DataFrame,
    auroc_bootstrap: pd.DataFrame,
    gap_summary: pd.DataFrame,
    interpretation: str,
    output_dir: Path,
) -> None:
    variant_level.to_csv(output_dir / "variant_level_negative_ld_contamination.tsv.gz", sep="\t", index=False)
    ancestry_summary.to_csv(output_dir / "ancestry_negative_ld_contamination_summary.tsv", sep="\t", index=False)
    auroc_summary.to_csv(output_dir / "auroc_negative_cleaning_summary.tsv", sep="\t", index=False)
    auroc_bootstrap.to_csv(output_dir / "auroc_negative_cleaning_bootstrap_values.tsv.gz", sep="\t", index=False)
    gap_summary.to_csv(output_dir / "auroc_negative_cleaning_gap_summary_pip0.9.tsv", sep="\t", index=False)
    (output_dir / "negative_class_contamination_interpretation.txt").write_text(interpretation + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Negative-class contamination analysis")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "manuscript_negative_contamination" / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    print("=" * 80)
    print("Negative-class contamination analysis")
    print("=" * 80)

    print("[1/6] Loading scored SuSiE rows")
    scored_df = load_scored_susie(base_dir)
    print(f"  Scored rows retained: {len(scored_df):,}")

    print("[2/6] Building unique variant-locus universe and anchors")
    variant_df = load_unique_variant_locus_rows(scored_df)
    anchors = choose_anchor_variants(variant_df)
    negatives = prepare_negative_ld_candidates(variant_df, anchors)
    print(f"  Unique variant-locus rows: {len(variant_df):,}")
    print(f"  Unique low-PIP negatives: {len(negatives):,}")
    print(f"  High-confidence top-anchor-PIP>=0.9 negatives: {(negatives['anchor_top_pip_ge_0_9']).sum():,}")

    print("[3/6] Computing anchor LD from chromosome PLINK files")
    ld_df = compute_negative_anchor_ld(negatives)
    variant_level = build_variant_level_contamination_table(negatives, ld_df)
    ancestry_summary = summarize_contamination_by_ancestry(variant_level)
    print(f"  LD computed for {int(variant_level['ld_available'].fillna(False).sum()):,} unique negatives")

    print("[4/6] Generating contamination figure")
    plot_contamination_summary(variant_level, output_dir)

    print("[5/6] Merging LD flags back to scored rows and rerunning AUROC")
    scored_with_ld = merge_contamination_to_scored_rows(scored_df, variant_level)
    scored_with_ld = add_tss_distance(scored_with_ld, base_dir)
    auroc_summary, auroc_bootstrap = run_auroc_sensitivity(
        scored_df=scored_with_ld,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        n_workers=args.n_workers,
    )
    auroc_summary = add_auc_delta_columns(auroc_summary)
    gap_summary = compute_gap_summary(auroc_summary)
    plot_auroc_sensitivity(auroc_summary, output_dir)

    print("[6/6] Writing outputs and interpretation")
    interpretation = build_interpretation_text(auroc_summary, gap_summary)
    write_outputs(
        variant_level=variant_level,
        ancestry_summary=ancestry_summary,
        auroc_summary=auroc_summary,
        auroc_bootstrap=auroc_bootstrap,
        gap_summary=gap_summary,
        interpretation=interpretation,
        output_dir=output_dir,
    )

    print("Done.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
