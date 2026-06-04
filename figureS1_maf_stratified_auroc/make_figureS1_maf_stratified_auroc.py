#!/usr/bin/env python3
"""
MAF-stratified, distance-matched AUROC analysis for the fine-mapped eQTL benchmark.

The full-analysis mode expects a scored variant table at
<results_dir>/analysis/susie_s2f_combined.tsv.gz. The plotting-only mode can
redraw Supplementary Figure S1 directly from the packaged bootstrap-value file:

    python make_figureS1_maf_stratified_auroc.py \
      --plot_from_bootstrap_values path/to/figureS1_maf_stratified_auroc_bootstrap.tsv.gz \
      --output_dir path/to/output
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


ANCESTRIES = ["AA", "CH", "NHW"]
PIP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
NEGATIVE_PIP_THRESHOLD = 0.01
N_BOOTSTRAP = 100
MIN_BALANCED_ROWS = 6

TOOLS = ["borzoi", "alphagenome"]
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

MAF_BIN_DEFS = [
    ("lt0.05", 0.0, 0.05),
    ("0.05-0.2", 0.05, 0.2),
    ("ge0.2", 0.2, 0.5),
]
MAF_BIN_ORDER = [x[0] for x in MAF_BIN_DEFS]
MAF_BIN_LABELS = {
    "lt0.05": "MAF < 0.05",
    "0.05-0.2": "0.05 ≤ MAF < 0.2",
    "ge0.2": "MAF ≥ 0.2",
}

RAW_SUSIE_FILES = {
    "AA": "/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes/MAGENTA_eQTLs_susie_pip_w_cs_AA.txt",
    "CH": "/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes/MAGENTA_eQTLs_susie_pip_w_cs_CH.txt",
    "NHW": "/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes/MAGENTA_eQTLs_susie_pip_w_cs_NHW.txt",
}


def _normalize_chr(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("^chr", "", regex=True)


def _build_join_key(
    chrom: pd.Series,
    pos: pd.Series,
    ref: pd.Series,
    alt: pd.Series,
    gene_id: pd.Series,
    ancestry: pd.Series,
) -> pd.Series:
    return (
        _normalize_chr(chrom) + "_"
        + pd.to_numeric(pos, errors="coerce").astype("Int64").astype(str) + "_"
        + ref.astype(str) + "_"
        + alt.astype(str) + "_"
        + gene_id.astype(str) + "_"
        + ancestry.astype(str)
    )


def _compute_maf_bin(maf: pd.Series) -> pd.Series:
    binned = pd.Series(pd.NA, index=maf.index, dtype="object")
    for label, lower, upper in MAF_BIN_DEFS:
        if label == "ge0.2":
            mask = (maf >= lower) & (maf <= upper)
        else:
            mask = (maf >= lower) & (maf < upper)
        binned.loc[mask] = label
    return binned


def load_scored_susie(results_dir: Path, include_per: bool) -> pd.DataFrame:
    combined_file = results_dir / "analysis" / "susie_s2f_combined.tsv.gz"
    if not combined_file.exists():
        raise FileNotFoundError(f"Missing scored redo table: {combined_file}")

    df = pd.read_csv(combined_file, sep="\t", low_memory=False)
    ancestries = ANCESTRIES + (["PER"] if include_per else [])
    df = df[df["tool"].isin(TOOLS) & df["ancestry"].isin(ancestries)].copy()
    print(f"Loaded {len(df):,} scored rows from {combined_file}")

    for col in ["pip", "logMeanSED", "raw_score", "pos", "position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["variant_pos"] = df["pos"].fillna(df["position"])
    chrom = df["chromosome"].fillna(df["chr"])
    pos = df["position"].fillna(df["pos"])
    ref = df["ref_allele"].fillna(df["ref"])
    alt = df["alt_allele"].fillna(df["alt"])
    df["join_key"] = _build_join_key(chrom, pos, ref, alt, df["gene_id"], df["ancestry"])
    return df


def load_redo_selected_maf(results_dir: Path, include_per: bool) -> pd.DataFrame:
    """Load AF/MAF from the per-gene all-variant files used to select redo pairs."""
    selected_file = results_dir / "selected_pairs.tsv.gz"
    if not selected_file.exists():
        raise FileNotFoundError(f"Missing selected redo pairs table: {selected_file}")

    selected = pd.read_csv(
        selected_file,
        sep="\t",
        usecols=[
            "source_file",
            "chromosome",
            "position",
            "ref_allele",
            "alt_allele",
            "gene_id",
            "ancestry",
        ],
        low_memory=False,
    )
    ancestries = ANCESTRIES + (["PER"] if include_per else [])
    selected = selected[selected["ancestry"].isin(ancestries)].copy()
    selected["join_key"] = _build_join_key(
        selected["chromosome"],
        selected["position"],
        selected["ref_allele"],
        selected["alt_allele"],
        selected["gene_id"],
        selected["ancestry"],
    )

    all_dfs = []
    source_groups = list(selected.groupby(["source_file", "ancestry"], sort=True))
    for idx, ((source_file, ancestry), selected_source) in enumerate(source_groups, start=1):
        path = Path(source_file)
        if not path.exists():
            print(f"[WARN] missing source file for AF/MAF: {path}")
            continue
        selected_keys = set(selected_source["join_key"])
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip" if path.suffix == ".gz" else None,
            usecols=["chrom", "pos", "a1", "a2", "molecular_trait_id", "af"],
        )
        df = df.rename(
            columns={
                "chrom": "chromosome",
                "pos": "position",
                "a2": "ref",
                "a1": "alt",
                "molecular_trait_id": "gene_id",
            }
        )
        df["ancestry"] = ancestry
        df["af"] = pd.to_numeric(df["af"], errors="coerce")
        df["maf"] = np.minimum(df["af"], 1 - df["af"])
        df["join_key"] = _build_join_key(
            df["chromosome"],
            df["position"],
            df["ref"],
            df["alt"],
            df["gene_id"],
            df["ancestry"],
        )
        df = df[df["join_key"].isin(selected_keys)].copy()
        if len(df):
            all_dfs.append(df[["join_key", "af", "maf"]])
        if idx % 250 == 0:
            print(f"  Loaded AF/MAF from {idx:,}/{len(source_groups):,} redo source file groups")

    if not all_dfs:
        raise RuntimeError("No redo source files could be loaded for AF/MAF")

    maf_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["join_key"])
    print(f"Loaded AF/MAF for {len(maf_df):,} selected redo rows from redo source files")
    return maf_df


def load_raw_susie_maf(include_per: bool) -> pd.DataFrame:
    if include_per:
        print("[WARN] PER requested, but no raw PER SuSiE MAF file is configured; PER rows will be dropped.")

    all_dfs = []
    for ancestry, path in RAW_SUSIE_FILES.items():
        print(f"  Loading raw SuSiE AF for {ancestry}...")
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["chrom", "pos", "a1", "a2", "molecular_trait_id", "af"],
        )
        df = df.rename(
            columns={
                "chrom": "chromosome",
                "pos": "position",
                "a2": "ref",
                "a1": "alt",
                "molecular_trait_id": "gene_id",
            }
        )
        df["ancestry"] = ancestry
        df["af"] = pd.to_numeric(df["af"], errors="coerce")
        df["maf"] = np.minimum(df["af"], 1 - df["af"])
        df["join_key"] = _build_join_key(
            df["chromosome"],
            df["position"],
            df["ref"],
            df["alt"],
            df["gene_id"],
            df["ancestry"],
        )
        all_dfs.append(df[["join_key", "af", "maf"]])

    maf_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["join_key"])
    print(f"Loaded AF/MAF for {len(maf_df):,} unique SuSiE rows")
    return maf_df


def filter_excluded_positive_source(df: pd.DataFrame, exclude_new_positive_class: bool) -> pd.DataFrame:
    if not exclude_new_positive_class:
        return df
    out = df.copy()
    is_new_positive = out["redo_score_source"].isin(
        ["new_redo_prediction", "new_auroc_expanded_prediction"]
    ) & (out["pip"] >= 0.5)
    removed = int(is_new_positive.sum())
    if removed:
        print(f"[filter] excluding {removed:,} redo-only positive rows (pip >= 0.5)")
    return out.loc[~is_new_positive].copy()


def prepare_analysis_tables(
    results_dir: Path,
    include_per: bool,
    exclude_new_positive_class: bool,
):
    scored = load_scored_susie(results_dir, include_per=include_per)
    scored = filter_excluded_positive_source(scored, exclude_new_positive_class)
    maf_df = load_redo_selected_maf(results_dir, include_per=include_per)

    analysis_df = scored[
        scored["pip"].notna()
        & scored["tss_distance_bin"].notna()
        & (scored["tss_distance_bin"] != "unknown")
    ].copy()
    analysis_df = analysis_df.merge(maf_df, on="join_key", how="left")

    missing_maf = int(analysis_df["maf"].isna().sum())
    if missing_maf:
        print(f"[fallback] {missing_maf:,} scored rows missing redo-source MAF; trying top-level SuSiE AF")
        fallback_maf = load_raw_susie_maf(include_per=include_per)
        missing = analysis_df[analysis_df["maf"].isna()].drop(columns=["af", "maf"])
        filled = missing.merge(fallback_maf, on="join_key", how="left")
        keep = analysis_df[analysis_df["maf"].notna()].copy()
        analysis_df = pd.concat([keep, filled], ignore_index=True, sort=False)

    analysis_df = analysis_df[analysis_df["maf"].notna()].copy()
    analysis_df["maf_bin"] = _compute_maf_bin(analysis_df["maf"])
    analysis_df = analysis_df[analysis_df["maf_bin"].isin(MAF_BIN_ORDER)].copy()
    analysis_df["pred_score"] = np.where(
        analysis_df["tool"] == "borzoi",
        pd.to_numeric(analysis_df["logMeanSED"], errors="coerce"),
        pd.to_numeric(analysis_df["raw_score"], errors="coerce"),
    )

    negatives = analysis_df[analysis_df["pip"] < NEGATIVE_PIP_THRESHOLD].copy()
    print(f"Prepared {len(analysis_df):,} scored rows with MAF and TSS bins")
    print(f"Prepared {len(negatives):,} low-PIP negative rows with MAF and TSS bins")
    return analysis_df, negatives


def balance_by_distance_bootstrap(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(positives) == 0 or len(negatives) == 0:
        return pd.DataFrame()

    positives = positives.copy()
    negatives = negatives.copy()
    balanced_dfs = []
    rng = np.random.RandomState(random_state)

    for tss_bin in positives["tss_distance_bin"].dropna().unique():
        pos_in_bin = positives[positives["tss_distance_bin"] == tss_bin]
        neg_in_bin = negatives[negatives["tss_distance_bin"] == tss_bin]
        if len(pos_in_bin) == 0 or len(neg_in_bin) == 0:
            continue
        neg_sampled = neg_in_bin.sample(n=len(pos_in_bin), replace=True, random_state=rng)
        balanced_dfs.append(pos_in_bin)
        balanced_dfs.append(neg_sampled)

    if not balanced_dfs:
        return pd.DataFrame()
    return pd.concat(balanced_dfs, ignore_index=True)


def compute_bootstrap_auroc(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    pred_col: str,
    pip_threshold: float,
    n_bootstrap: int = N_BOOTSTRAP,
    base_seed: int = 42,
) -> Optional[dict]:
    positives = positives[positives["pip"] >= pip_threshold].copy()
    n_pos_raw = len(positives)
    n_neg_raw = len(negatives)
    if n_pos_raw == 0 or n_neg_raw == 0:
        return None

    auc_values = []
    balanced_example = None
    for bootstrap_idx in range(n_bootstrap):
        balanced = balance_by_distance_bootstrap(
            positives,
            negatives,
            random_state=base_seed + bootstrap_idx,
        )
        if len(balanced) < MIN_BALANCED_ROWS:
            continue

        balanced = balanced.dropna(subset=[pred_col, "pip"])
        if len(balanced) < MIN_BALANCED_ROWS:
            continue

        y_true = (balanced["pip"] >= pip_threshold).astype(int)
        y_score = np.abs(pd.to_numeric(balanced[pred_col], errors="coerce").values)
        mask = ~np.isnan(y_score)
        y_true = y_true[mask]
        y_score = y_score[mask]
        if len(y_true) < MIN_BALANCED_ROWS or y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_values.append(auc(fpr, tpr))
        if balanced_example is None:
            balanced_example = (int(y_true.sum()), int(len(y_true) - y_true.sum()))

    if len(auc_values) < 5 or balanced_example is None:
        return None

    auc_array = np.array(auc_values)
    n_pos_bal, n_neg_bal = balanced_example
    return {
        "auc_values": auc_array,
        "auc_mean": float(np.mean(auc_array)),
        "auc_std": float(np.std(auc_array)),
        "auc_ci_low": float(np.percentile(auc_array, 2.5)),
        "auc_ci_high": float(np.percentile(auc_array, 97.5)),
        "n_bootstrap": int(len(auc_array)),
        "n_pos_raw": int(n_pos_raw),
        "n_neg_raw": int(n_neg_raw),
        "n_pos_balanced_example": int(n_pos_bal),
        "n_neg_balanced_example": int(n_neg_bal),
    }


def summarize_positive_frequency(
    positives: pd.DataFrame,
    threshold: float,
    tool: str,
    ancestry: str,
) -> Optional[dict]:
    sub = positives[positives["pip"] >= threshold].copy()
    if sub.empty:
        return None

    af = pd.to_numeric(sub["af"], errors="coerce")
    maf = pd.to_numeric(sub["maf"], errors="coerce")
    mask = af.notna() & maf.notna()
    af = af[mask]
    maf = maf[mask]
    if len(af) == 0:
        return None

    return {
        "tool": tool,
        "ancestry": ancestry,
        "pip_threshold": threshold,
        "n_positives": int(len(af)),
        "af_mean": float(af.mean()),
        "af_median": float(af.median()),
        "af_q25": float(af.quantile(0.25)),
        "af_q75": float(af.quantile(0.75)),
        "maf_mean": float(maf.mean()),
        "maf_median": float(maf.median()),
        "maf_q25": float(maf.quantile(0.25)),
        "maf_q75": float(maf.quantile(0.75)),
        "pct_maf_lt_0_01": float((maf < 0.01).mean() * 100),
        "pct_maf_lt_0_05": float((maf < 0.05).mean() * 100),
        "pct_maf_ge_0_05": float((maf >= 0.05).mean() * 100),
    }


def plot_maf_stratified_boxplot(results: dict, output_dir: Path, suffix: str) -> None:
    fig, axes = plt.subplots(len(MAF_BIN_ORDER), len(ANCESTRIES), figsize=(16, 12), sharey=True)
    plt.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.11, hspace=0.34, wspace=0.20)

    base_positions = np.arange(len(PIP_THRESHOLDS)) + 1
    offset = 0.18
    width = 0.30

    for row_idx, maf_bin in enumerate(MAF_BIN_ORDER):
        for col_idx, ancestry in enumerate(ANCESTRIES):
            ax = axes[row_idx, col_idx]
            for tool in TOOLS:
                tool_positions = base_positions + (-offset if tool == "borzoi" else offset)
                box_data = []
                valid_positions = []
                for pos, threshold in zip(tool_positions, PIP_THRESHOLDS):
                    key = (tool, ancestry, maf_bin, threshold)
                    if key not in results:
                        continue
                    box_data.append(results[key]["auc_values"])
                    valid_positions.append(pos)

                if not box_data:
                    continue

                bp = ax.boxplot(
                    box_data,
                    positions=valid_positions,
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    medianprops=dict(color="black", lw=1.3),
                    whiskerprops=dict(color="#555555", lw=1.0),
                    capprops=dict(color="#555555", lw=1.0),
                    boxprops=dict(edgecolor="#555555", lw=1.0),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(TOOL_CONFIG[tool]["color"])
                    patch.set_alpha(0.80)

            ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.6)
            ax.set_ylim(0.3, 1.0)
            ax.set_xlim(0.4, len(PIP_THRESHOLDS) + 0.6)
            ax.set_xticks(base_positions)
            ax.set_xticklabels([f"≥{t}" for t in PIP_THRESHOLDS], fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.set_xlabel("PIP Threshold", fontsize=10)

            if col_idx == 0:
                ax.set_ylabel("AUROC", fontsize=10)
                ax.text(
                    -0.34,
                    1.10,
                    MAF_BIN_LABELS[maf_bin],
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                    clip_on=False,
                )
            if row_idx == 0:
                ax.set_title(ANCESTRY_LABELS[ancestry], fontsize=11, fontweight="bold", pad=8)

    legend_handles = [
        Patch(
            facecolor=TOOL_CONFIG[tool]["color"],
            edgecolor="#555555",
            label=TOOL_CONFIG[tool]["display_name"],
            alpha=0.80,
        )
        for tool in TOOLS
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle("MAF-Stratified, Distance-Matched AUROC", fontsize=14, fontweight="bold", y=0.975)

    stem = f"figureS1_maf_stratified_auroc_boxplot{suffix}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=250)
    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")
    plt.close(fig)


def load_results_from_bootstrap_values(path: Path) -> dict:
    """Load packaged Figure S1 bootstrap values into the plotting structure."""
    boot = pd.read_csv(path, sep="\t")
    results = {}
    group_cols = ["tool", "ancestry", "maf_bin", "pip_threshold"]
    for (tool, ancestry, maf_bin, threshold), sub in boot.groupby(group_cols, observed=True):
        sub = sub.sort_values("bootstrap_idx") if "bootstrap_idx" in sub.columns else sub
        results[(tool, ancestry, maf_bin, float(threshold))] = {
            "auc_values": pd.to_numeric(sub["auc"], errors="coerce").dropna().to_numpy()
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuscript Supplementary Figure S1 MAF-stratified AUROC plot")
    parser.add_argument("--base_dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Directory containing selected_pairs.tsv.gz and analysis/susie_s2f_combined.tsv.gz; defaults to base_dir/results.",
    )
    parser.add_argument("--file_suffix", default="", help="Suffix appended to output filenames.")
    parser.add_argument(
        "--plot_from_bootstrap_values",
        type=Path,
        default=None,
        help="Read packaged bootstrap values and only regenerate Supplementary Figure S1.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for regenerated figure files; defaults to <results_dir>/analysis/figures.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--include_per", action="store_true")
    parser.add_argument(
        "--exclude_new_positive_class",
        action="store_true",
        help="Exclude newly added positive rows (pip >= 0.5), matching the no_new_positives manuscript variant.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else base_dir / "results"
    analysis_dir = results_dir / "analysis"
    output_dir = args.output_dir.resolve() if args.output_dir else analysis_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_bits = []
    if args.file_suffix:
        suffix_bits.append(args.file_suffix.lstrip("_"))
    if args.exclude_new_positive_class:
        suffix_bits.append("no_new_positives")
    if args.include_per:
        suffix_bits.append("with_per")
    suffix = "_" + "_".join(suffix_bits) if suffix_bits else ""

    print("=" * 72)
    print("MAF-Stratified, Distance-Matched AUROC")
    print("=" * 72)

    if args.plot_from_bootstrap_values:
        results = load_results_from_bootstrap_values(args.plot_from_bootstrap_values)
        plot_maf_stratified_boxplot(results, output_dir, suffix=suffix)
        return

    positives_df, negatives_df = prepare_analysis_tables(
        results_dir,
        include_per=args.include_per,
        exclude_new_positive_class=args.exclude_new_positive_class,
    )

    plot_ancestries = ANCESTRIES
    all_ancestries = ANCESTRIES + (["PER"] if args.include_per else [])
    results = {}
    summary_rows = []
    bootstrap_rows = []
    af_summary_rows = []

    for tool in TOOLS:
        pos_tool = positives_df[positives_df["tool"] == tool].copy()
        neg_tool = negatives_df[negatives_df["tool"] == tool].copy()
        print(f"\n[{tool}] scored rows={len(pos_tool):,}, negatives={len(neg_tool):,}")

        for ancestry in all_ancestries:
            pos_anc = pos_tool[pos_tool["ancestry"] == ancestry].copy()
            neg_anc = neg_tool[neg_tool["ancestry"] == ancestry].copy()
            if pos_anc.empty:
                continue

            print(f"  {ancestry}: PIP≥0.9 positives={int((pos_anc['pip'] >= 0.9).sum())}")
            for threshold in PIP_THRESHOLDS:
                af_summary = summarize_positive_frequency(pos_anc, threshold, tool, ancestry)
                if af_summary is not None:
                    af_summary_rows.append(af_summary)

            print(f"  {ancestry} strata")
            for maf_bin in MAF_BIN_ORDER:
                pos_maf = pos_anc[pos_anc["maf_bin"] == maf_bin].copy()
                neg_maf = neg_anc[neg_anc["maf_bin"] == maf_bin].copy()
                if pos_maf.empty or neg_maf.empty:
                    print(f"    {maf_bin}: no rows")
                    continue

                print(f"    {maf_bin}:", end=" ", flush=True)
                for threshold in PIP_THRESHOLDS:
                    result = compute_bootstrap_auroc(
                        pos_maf,
                        neg_maf,
                        pred_col="pred_score",
                        pip_threshold=threshold,
                        n_bootstrap=args.n_bootstrap,
                        base_seed=args.seed,
                    )

                    pos_count = int((pos_maf["pip"] >= threshold).sum())
                    neg_count = int(len(neg_maf))
                    row = {
                        "tool": tool,
                        "ancestry": ancestry,
                        "maf_bin": maf_bin,
                        "maf_bin_label": MAF_BIN_LABELS[maf_bin],
                        "pip_threshold": threshold,
                        "n_pos_raw": pos_count,
                        "n_neg_raw": neg_count,
                    }

                    if result is None:
                        row.update(
                            {
                                "auc_mean": np.nan,
                                "auc_std": np.nan,
                                "auc_ci_low": np.nan,
                                "auc_ci_high": np.nan,
                                "n_bootstrap": 0,
                                "n_pos_balanced_example": np.nan,
                                "n_neg_balanced_example": np.nan,
                            }
                        )
                        summary_rows.append(row)
                        print(f"PIP≥{threshold}:skip(n+={pos_count},n-={neg_count})", end=" | ", flush=True)
                        continue

                    results[(tool, ancestry, maf_bin, threshold)] = result
                    row.update(
                        {
                            "auc_mean": result["auc_mean"],
                            "auc_std": result["auc_std"],
                            "auc_ci_low": result["auc_ci_low"],
                            "auc_ci_high": result["auc_ci_high"],
                            "n_bootstrap": result["n_bootstrap"],
                            "n_pos_balanced_example": result["n_pos_balanced_example"],
                            "n_neg_balanced_example": result["n_neg_balanced_example"],
                        }
                    )
                    summary_rows.append(row)

                    for idx, auc_value in enumerate(result["auc_values"]):
                        bootstrap_rows.append(
                            {
                                "tool": tool,
                                "ancestry": ancestry,
                                "maf_bin": maf_bin,
                                "maf_bin_label": MAF_BIN_LABELS[maf_bin],
                                "pip_threshold": threshold,
                                "bootstrap_idx": idx,
                                "auc": float(auc_value),
                            }
                        )

                    print(
                        f"PIP≥{threshold}:{result['auc_mean']:.3f}+/-{result['auc_std']:.3f}"
                        f"(n+={pos_count},n-={neg_count})",
                        end=" | ",
                        flush=True,
                    )
                print()

    af_summary_path = analysis_dir / f"figureS1_positive_maf_summary{suffix}.tsv"
    pd.DataFrame(af_summary_rows).to_csv(af_summary_path, sep="\t", index=False)
    print(f"\n[OK] Saved AF summary: {af_summary_path}")

    summary_path = analysis_dir / f"figureS1_maf_stratified_auroc_by_threshold{suffix}.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    print(f"[OK] Saved summary: {summary_path}")

    bootstrap_path = analysis_dir / f"figureS1_maf_stratified_auroc_bootstrap{suffix}.tsv.gz"
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, sep="\t", index=False, compression="gzip")
    print(f"[OK] Saved bootstrap values: {bootstrap_path}")

    if set(plot_ancestries).issubset({key[1] for key in results}):
        plot_maf_stratified_boxplot(results, output_dir, suffix=suffix)
    else:
        print("[WARN] Skipping plot because at least one manuscript ancestry has no plottable results.")

    print("\nDone.")


if __name__ == "__main__":
    main()
