#!/usr/bin/env python3
"""
Bootstrap AUROC summary and manuscript-style boxplot for the fine-mapped eQTL benchmark.

Manuscript version intentionally excludes PER and only reports:
  - AA: African American
  - CH: Caribbean Hispanic
  - NHW: Non-Hispanic White
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


ANCESTRIES = ["AA", "CH", "NHW"]
ANCESTRY_LABELS = {
    "AA": "African American",
    "CH": "Caribbean Hispanic",
    "NHW": "Non-Hispanic White",
}
TOOLS = ["borzoi", "alphagenome"]
TOOL_LABELS = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome"}
PIP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ["standard", "rest"]
MODE_LABELS = {
    "standard": "vs. low PIP (<0.01)",
    "rest": "vs. intermediate PIP",
}
PANEL_LABELS = {"standard": "a", "rest": "b"}
MODEL_COLORS = {"borzoi": "#E64B35", "alphagenome": "#4DBBD5"}
PRED_COLS = {"borzoi": "logMeanSED", "alphagenome": "raw_score"}
NEGATIVE_PIP_THRESHOLD = 0.01
N_DISTANCE_BINS = 10


def load_gene_tss(base_dir: Path) -> pd.DataFrame:
    gene_loc_file = base_dir / "inputs" / "hg38_gene_locations.txt"
    if not gene_loc_file.exists():
        raise FileNotFoundError(f"Gene location file not found: {gene_loc_file}")

    df = pd.read_csv(gene_loc_file, sep="\t")
    df = df.rename(columns={"chromosome_name": "chromosome", "ensgid": "gene_id", "TSS": "tss"})
    df["chromosome"] = "chr" + df["chromosome"].astype(str)
    return df[["gene_id", "chromosome", "tss"]]


def load_combined(combined_file: Path) -> pd.DataFrame:
    df = pd.read_csv(combined_file, sep="\t", low_memory=False)
    df = df[df["ancestry"].isin(ANCESTRIES) & df["tool"].isin(TOOLS)].copy()

    for col in ["pip", "logMeanSED", "raw_score", "pos", "position", "pos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["variant_pos"] = pd.to_numeric(df.get("pos"), errors="coerce").fillna(
        pd.to_numeric(df.get("position"), errors="coerce")
    )
    return df


def filter_excluded_positive_source(df: pd.DataFrame, exclude_new_positive_class: bool) -> pd.DataFrame:
    if not exclude_new_positive_class:
        return df

    out = df.copy()
    is_new_positive = out["redo_score_source"].isin(
        ["new_redo_prediction", "new_auroc_expanded_prediction"]
    ) & (out["pip"] >= 0.5)
    removed = int(is_new_positive.sum())
    if removed:
        print(f"[filter] excluding {removed} redo-only positive rows (pip >= 0.5)")
    return out.loc[~is_new_positive].copy()


def add_tss_distance(df: pd.DataFrame, gene_tss: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(gene_tss, on=["gene_id"], how="left", suffixes=("", "_gene"))
    out["tss_distance"] = np.abs(out["variant_pos"] - out["tss"])
    out["log_distance"] = np.log10(out["tss_distance"].replace(0, 1))
    return out


def balance_by_distance_bootstrap(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    random_state: int,
    n_bins: int = N_DISTANCE_BINS,
) -> pd.DataFrame:
    if positives.empty or negatives.empty:
        return pd.DataFrame()

    all_distances = pd.concat([positives["log_distance"], negatives["log_distance"]]).dropna()
    if all_distances.empty:
        return pd.DataFrame()

    bin_edges = np.percentile(all_distances, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return pd.DataFrame()

    positives = positives.copy()
    negatives = negatives.copy()
    positives["distance_bin"] = pd.cut(
        positives["log_distance"], bins=bin_edges, labels=False, include_lowest=True
    )
    negatives["distance_bin"] = pd.cut(
        negatives["log_distance"], bins=bin_edges, labels=False, include_lowest=True
    )

    rng = np.random.RandomState(random_state)
    balanced_parts = []
    n_real_bins = len(bin_edges) - 1
    for bin_idx in range(n_real_bins):
        pos_bin = positives[positives["distance_bin"] == bin_idx]
        neg_bin = negatives[negatives["distance_bin"] == bin_idx]
        if pos_bin.empty or neg_bin.empty:
            continue
        neg_sampled = neg_bin.sample(n=len(pos_bin), replace=True, random_state=rng)
        balanced_parts.append(pos_bin)
        balanced_parts.append(neg_sampled)

    if not balanced_parts:
        return pd.DataFrame()
    return pd.concat(balanced_parts, ignore_index=True)


def compute_single_bootstrap_auc(
    bootstrap_idx: int,
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    pred_col: str,
    pip_threshold: float,
    base_seed: int,
) -> float | None:
    balanced = balance_by_distance_bootstrap(
        positives=positives,
        negatives=negatives,
        random_state=base_seed + bootstrap_idx,
    )
    if balanced.empty:
        return None

    balanced = balanced.dropna(subset=[pred_col, "pip"])
    if len(balanced) < 10:
        return None

    y_true = (balanced["pip"] >= pip_threshold).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    y_score = np.abs(pd.to_numeric(balanced[pred_col], errors="coerce"))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def compute_bootstrap_distribution(
    df: pd.DataFrame,
    tool: str,
    ancestry: str,
    pip_threshold: float,
    mode: str,
    n_bootstrap: int,
    seed: int,
    n_workers: int,
) -> dict | None:
    pred_col = PRED_COLS[tool]
    subset = df[(df["tool"] == tool) & (df["ancestry"] == ancestry)].copy()
    positives = subset[subset["pip"] >= pip_threshold].copy()

    if mode == "standard":
        negatives = subset[(subset["pip"] >= 0.0) & (subset["pip"] < NEGATIVE_PIP_THRESHOLD)].copy()
    else:
        negatives = subset[(subset["pip"] >= NEGATIVE_PIP_THRESHOLD) & (subset["pip"] < pip_threshold)].copy()

    if positives.empty or negatives.empty:
        return None

    worker_fn = partial(
        compute_single_bootstrap_auc,
        positives=positives,
        negatives=negatives,
        pred_col=pred_col,
        pip_threshold=pip_threshold,
        base_seed=seed,
    )

    auc_values = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in executor.map(worker_fn, range(n_bootstrap)):
            if result is not None:
                auc_values.append(result)

    if len(auc_values) < 10:
        return None

    auc_array = np.array(auc_values)
    return {
        "auc_values": auc_array,
        "auc_mean": float(np.mean(auc_array)),
        "auc_std": float(np.std(auc_array)),
        "auc_ci_low": float(np.percentile(auc_array, 2.5)),
        "auc_ci_high": float(np.percentile(auc_array, 97.5)),
        "n_pos": int(len(positives)),
        "n_neg": int(len(negatives)),
        "n_bootstrap": int(len(auc_array)),
    }


def plot_manuscript_boxplot(results_df: pd.DataFrame, output_dir: Path, file_suffix: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    plt.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.12, hspace=0.42, wspace=0.20)

    base_positions = np.arange(len(PIP_THRESHOLDS)) + 1
    offset = 0.18
    width = 0.30

    for row_idx, mode in enumerate(MODES):
        for col_idx, ancestry in enumerate(ANCESTRIES):
            ax = axes[row_idx, col_idx]
            anc_mode = results_df[(results_df["mode"] == mode) & (results_df["ancestry"] == ancestry)]

            for tool in TOOLS:
                tool_positions = base_positions + (-offset if tool == "borzoi" else offset)
                box_data = []
                valid_positions = []
                for pos, pip_threshold in zip(tool_positions, PIP_THRESHOLDS):
                    hit = anc_mode[
                        (anc_mode["tool"] == tool) &
                        (anc_mode["pip_threshold"] == pip_threshold)
                    ]
                    if hit.empty:
                        continue
                    auc_values = np.fromstring(hit.iloc[0]["auc_values"], sep=",")
                    if len(auc_values) == 0:
                        continue
                    box_data.append(auc_values)
                    valid_positions.append(pos)

                if box_data:
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
                        patch.set_facecolor(MODEL_COLORS[tool])
                        patch.set_alpha(0.80)

            ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.6)
            ax.set_ylim(0.3, 1.06)
            ax.set_yticks(np.arange(0.3, 1.01, 0.1))
            ax.set_xlim(0.4, len(PIP_THRESHOLDS) + 0.6)
            ax.set_xticks(base_positions)
            ax.set_xticklabels([f"≥{t}" for t in PIP_THRESHOLDS])
            ax.tick_params(axis="both", labelsize=11)
            ax.set_xlabel("PIP Threshold")
            if col_idx == 0:
                ax.set_ylabel("AUROC")
            if row_idx == 0:
                ax.set_title(ANCESTRY_LABELS[ancestry], fontsize=14, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.text(
                    -0.26, 1.10, f"({PANEL_LABELS[mode]})", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", va="bottom", ha="left", clip_on=False
                )
                ax.text(
                    -0.10, 1.10, MODE_LABELS[mode], transform=ax.transAxes,
                    fontsize=14, fontstyle="italic", va="bottom", ha="left",
                    color="#444444", clip_on=False
                )

    legend_handles = [
        Patch(facecolor=MODEL_COLORS[tool], edgecolor="#555555", label=TOOL_LABELS[tool], alpha=0.80)
        for tool in TOOLS
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))

    stem = f"figure3_distance_matched_auroc_boxplot{file_suffix}"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=250)
    print(f"[OK] Saved {output_dir / (stem + '.pdf')}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuscript Figure 3 distance-matched AUROC bootstrap plot")
    parser.add_argument("--base_dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Directory containing analysis/susie_s2f_combined.tsv.gz; defaults to base_dir/results.",
    )
    parser.add_argument("--file_suffix", default="", help="Suffix appended to output filenames.")
    parser.add_argument(
        "--plot_from_results",
        type=Path,
        default=None,
        help="Read a precomputed bootstrap results TSV and only regenerate the figure.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for regenerated figure files; defaults to <results_dir>/analysis/figures.",
    )
    parser.add_argument(
        "--exclude_new_positive_class",
        action="store_true",
        help="Exclude newly added positive rows (pip >= 0.5) that were not in the earlier scored universe.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else base_dir / "results"
    analysis_dir = results_dir / "analysis"
    figure_dir = args.output_dir.resolve() if args.output_dir else analysis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    file_suffix_bits = []
    if args.file_suffix:
        file_suffix_bits.append(args.file_suffix.lstrip("_"))
    if args.exclude_new_positive_class:
        file_suffix_bits.append("no_new_positives")
    file_suffix = "_" + "_".join(file_suffix_bits) if file_suffix_bits else ""

    if args.plot_from_results:
        results_df = pd.read_csv(args.plot_from_results, sep="\t")
        plot_manuscript_boxplot(results_df, figure_dir, file_suffix)
        return

    gene_tss = load_gene_tss(base_dir)
    df = load_combined(analysis_dir / "susie_s2f_combined.tsv.gz")
    df = filter_excluded_positive_source(df, args.exclude_new_positive_class)
    df = add_tss_distance(df, gene_tss)
    df = df[df["pip"].notna() & df["log_distance"].notna()].copy()

    records = []
    for mode in MODES:
        print(f"[mode] {mode}")
        for tool in TOOLS:
            for ancestry in ANCESTRIES:
                print(f"  {tool} {ancestry}", flush=True)
                for pip_threshold in PIP_THRESHOLDS:
                    result = compute_bootstrap_distribution(
                        df=df,
                        tool=tool,
                        ancestry=ancestry,
                        pip_threshold=pip_threshold,
                        mode=mode,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed,
                        n_workers=args.n_workers,
                    )
                    if result is None:
                        continue
                    records.append(
                        {
                            "tool": tool,
                            "ancestry": ancestry,
                            "pip_threshold": pip_threshold,
                            "mode": mode,
                            "auc_mean": result["auc_mean"],
                            "auc_std": result["auc_std"],
                            "auc_ci_low": result["auc_ci_low"],
                            "auc_ci_high": result["auc_ci_high"],
                            "n_pos": result["n_pos"],
                            "n_neg": result["n_neg"],
                            "n_bootstrap": result["n_bootstrap"],
                            "auc_values": ",".join(f"{x:.8f}" for x in result["auc_values"]),
                        }
                    )
                    print(
                        f"    PIP≥{pip_threshold}: {result['auc_mean']:.3f} +/- {result['auc_std']:.3f}",
                        flush=True,
                    )

    results_df = pd.DataFrame(records)
    results_path = analysis_dir / f"figure3_distance_matched_auroc_bootstrap{file_suffix}.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    print(f"[OK] Wrote {results_path}")

    plot_manuscript_boxplot(results_df, figure_dir, file_suffix)


if __name__ == "__main__":
    main()
