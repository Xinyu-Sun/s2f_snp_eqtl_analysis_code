#!/usr/bin/env python3
"""Build the benchmark prediction tables and the summary results behind Figures 1-3, S1 and Tables S1, S8-S10.

AA and NHW rows come from the base benchmark tables built with model_scoring/. CH rows come from the
scored CH pairs (build_scoring_plan.py, collect_scores.py) of the regular-eQTL fine-mapping profile named
by the FINEMAPPING_PROFILE environment variable. Other fine-mapping profiles and interaction-eQTL results
are not mixed into the benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


ANCESTRIES = ("AA", "CH", "NHW")
TOOLS = ("borzoi", "alphagenome")
TSS_BINS = ("0-3kb", "3-12kb", "12-35kb", ">35kb")
MAF_BINS = ("lt0.05", "0.05-0.2", "ge0.2")
MAF_LABELS = {
    "lt0.05": "MAF < 0.05",
    "0.05-0.2": "0.05 ≤ MAF < 0.2",
    "ge0.2": "MAF ≥ 0.2",
}
PIP_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
BASE_SEED = 42
N_BOOTSTRAP = 100
FINEMAPPING_PROFILE = os.environ["FINEMAPPING_PROFILE"]


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def role_mask(frame: pd.DataFrame, role: str) -> pd.Series:
    return frame["analysis_role"].fillna("").str.split(",").map(lambda values: role in values)


def fisher_ci(value: float, n: int) -> tuple[float, float]:
    if n < 4 or not np.isfinite(value) or abs(value) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(value)
    half = norm.ppf(0.975) / np.sqrt(n - 3)
    return (float(np.tanh(z - half)), float(np.tanh(z + half)))


def standardize_base_nominal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame = frame[frame["population"].isin(["AA", "NHW"])].copy()
    frame["ancestry"] = frame["population"]
    frame = frame.rename(
        columns={
            "borzoi_logMeanSED": "borzoi_score",
            "alphagenome_raw_score": "alphagenome_score",
            "alt_allele_freq": "af",
        }
    )
    return numeric(frame, ["beta", "borzoi_score", "alphagenome_score", "distance_to_tss", "af"])


def standardize_ch(master: pd.DataFrame, role: str) -> pd.DataFrame:
    frame = master[role_mask(master, role)].copy()
    frame["ancestry"] = "CH"
    frame["population"] = "CH"
    if "alt_allele_freq" in frame:
        frame["af"] = pd.to_numeric(frame["alt_allele_freq"], errors="coerce")
    if "maf" not in frame and "af" in frame:
        frame["maf"] = np.minimum(frame["af"], 1 - frame["af"])
    return numeric(
        frame,
        [
            "beta", "borzoi_score", "alphagenome_score", "distance_to_tss",
            "af", "maf", "pip", "CS", "position", "tss",
        ],
    )


def standardize_base_finemap(path: Path, keep_ancestries: tuple[str, ...] = ("AA", "NHW")) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame = frame[frame["ancestry"].isin(keep_ancestries)].copy()
    frame = frame.rename(
        columns={
            "borzoi_logMeanSED": "borzoi_score",
            "alphagenome_raw_score": "alphagenome_score",
            "alt_allele_freq": "af",
        }
    )
    return numeric(frame, ["beta", "pip", "CS", "borzoi_score", "alphagenome_score", "position", "af"])


def grouped_subsets(frame: pd.DataFrame):
    yield "All", "All", frame
    for ancestry in ANCESTRIES:
        yield ancestry, "All", frame[frame["ancestry"].eq(ancestry)]
    for distance_bin in TSS_BINS:
        yield "All", distance_bin, frame[frame["tss_distance_bin"].eq(distance_bin)]
    for ancestry in ANCESTRIES:
        for distance_bin in TSS_BINS:
            yield ancestry, distance_bin, frame[
                frame["ancestry"].eq(ancestry) & frame["tss_distance_bin"].eq(distance_bin)
            ]


def performance_table(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry, distance_bin, subset in grouped_subsets(frame):
            valid = subset.dropna(subset=[score, "beta"])
            if len(valid) < 10:
                continue
            sp, spp = spearmanr(valid[score], valid["beta"])
            pe, pep = pearsonr(valid[score], valid["beta"])
            rows.append(
                {
                    "dataset": dataset,
                    "tool": tool,
                    "ancestry": ancestry,
                    "tss_distance_bin": distance_bin,
                    "n": len(valid),
                    "spearman_r": sp,
                    "spearman_p": spp,
                    "pearson_r": pe,
                    "pearson_p": pep,
                    "concordance": float((np.sign(valid[score]) == np.sign(valid["beta"])).mean()),
                }
            )
    return pd.DataFrame(rows)


def finemapped_stratified(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    conc_rows = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry in ANCESTRIES:
            for distance_bin in TSS_BINS:
                valid = frame[
                    frame["ancestry"].eq(ancestry)
                    & frame["tss_distance_bin"].eq(distance_bin)
                ].dropna(subset=[score, "beta"])
                if len(valid) < 10:
                    continue
                sp, spp = spearmanr(valid[score], valid["beta"])
                ci = fisher_ci(float(sp), len(valid))
                corr_rows.append(
                    {
                        "tool": tool,
                        "ancestry": ancestry,
                        "tss_distance_bin": distance_bin,
                        "n": len(valid),
                        "spearman_r": sp,
                        "spearman_p": spp,
                        "spearman_ci_lower": ci[0],
                        "spearman_ci_upper": ci[1],
                    }
                )
                concordant = int((np.sign(valid[score]) == np.sign(valid["beta"])).sum())
                rate = concordant / len(valid)
                se = math.sqrt(rate * (1 - rate) / len(valid))
                conc_rows.append(
                    {
                        "tool": tool,
                        "ancestry": ancestry,
                        "tss_distance_bin": distance_bin,
                        "n": len(valid),
                        "concordant": concordant,
                        "discordant": len(valid) - concordant,
                        "concordance_rate": rate,
                        "ci_lower": max(0.0, rate - 1.96 * se),
                        "ci_upper": min(1.0, rate + 1.96 * se),
                    }
                )
    return pd.DataFrame(corr_rows), pd.DataFrame(conc_rows)


def convergence_table(nominal: pd.DataFrame, high_pip: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in [("Nominal eQTLs", nominal), ("Fine-mapped (PIP≥0.9)", high_pip)]:
        for ancestry, distance_bin, subset in grouped_subsets(frame):
            valid = subset.dropna(subset=["borzoi_score", "alphagenome_score"])
            if len(valid) < 10:
                continue
            sp, spp = spearmanr(valid["borzoi_score"], valid["alphagenome_score"])
            pe, pep = pearsonr(valid["borzoi_score"], valid["alphagenome_score"])
            rows.append(
                {
                    "dataset": dataset,
                    "ancestry": ancestry,
                    "tss_distance_bin": distance_bin,
                    "n": len(valid),
                    "spearman_r": sp,
                    "spearman_p": spp,
                    "pearson_r": pe,
                    "pearson_p": pep,
                    "concordance": float(
                        (np.sign(valid["borzoi_score"]) == np.sign(valid["alphagenome_score"])).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_log_distance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "distance_to_tss" in out:
        distance = pd.to_numeric(out["distance_to_tss"], errors="coerce").abs()
    elif "tss" in out and "position" in out:
        distance = (
            pd.to_numeric(out["position"], errors="coerce")
            - pd.to_numeric(out["tss"], errors="coerce")
        ).abs()
    else:
        distance = pd.Series(np.nan, index=out.index, dtype=float)
    out["log_distance"] = np.log10(distance.where(distance > 0, 1.0))
    return out


def distance_balanced(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    seed: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    all_distances = pd.concat([positives["log_distance"], negatives["log_distance"]]).dropna()
    if all_distances.empty:
        return pd.DataFrame()
    edges = np.percentile(all_distances, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return pd.DataFrame()
    pos = positives.copy()
    neg = negatives.copy()
    pos["distance_bin"] = pd.cut(pos["log_distance"], edges, labels=False, include_lowest=True)
    neg["distance_bin"] = pd.cut(neg["log_distance"], edges, labels=False, include_lowest=True)
    rng = np.random.RandomState(seed)
    pieces = []
    for bin_index in range(len(edges) - 1):
        pos_bin = pos[pos["distance_bin"].eq(bin_index)]
        neg_bin = neg[neg["distance_bin"].eq(bin_index)]
        if pos_bin.empty or neg_bin.empty:
            continue
        pieces.extend([pos_bin, neg_bin.sample(n=len(pos_bin), replace=True, random_state=rng)])
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def bootstrap_auc(
    frame: pd.DataFrame,
    score: str,
    threshold: float,
    mode: str,
    positive_mask: pd.Series | None = None,
) -> dict[str, object] | None:
    valid = frame.dropna(subset=[score, "pip", "log_distance"]).copy()
    positives = valid[valid["pip"].ge(threshold)].copy()
    if positive_mask is not None:
        positives = positives[positive_mask.reindex(positives.index).fillna(False)]
    if mode == "standard":
        negatives = valid[valid["pip"].ge(0) & valid["pip"].lt(0.01)].copy()
    elif mode == "rest":
        negatives = valid[valid["pip"].ge(0.01) & valid["pip"].lt(threshold)].copy()
    else:
        raise ValueError(mode)
    if positives.empty or negatives.empty:
        return None
    aucs = []
    example = None
    for index in range(N_BOOTSTRAP):
        balanced = distance_balanced(positives, negatives, BASE_SEED + index)
        if balanced.empty:
            continue
        y = balanced["pip"].ge(threshold).astype(int)
        if y.nunique() != 2:
            continue
        aucs.append(float(roc_auc_score(y, balanced[score].abs())))
        if example is None:
            example = (int(y.sum()), int((1 - y).sum()))
    if len(aucs) < 10:
        return None
    values = np.asarray(aucs)
    return {
        "auc_mean": float(values.mean()),
        "auc_std": float(values.std()),
        "auc_ci_low": float(np.percentile(values, 2.5)),
        "auc_ci_high": float(np.percentile(values, 97.5)),
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "n_bootstrap": len(values),
        "auc_values": ",".join(f"{value:.8f}" for value in values),
        "n_pos_balanced_example": example[0] if example else 0,
        "n_neg_balanced_example": example[1] if example else 0,
    }


def ch_auroc_rows(ch_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for threshold in PIP_THRESHOLDS:
            for mode in ("standard", "rest"):
                result = bootstrap_auc(ch_pool, score, threshold, mode)
                if result:
                    row = {
                        "tool": tool,
                        "ancestry": "CH",
                        "pip_threshold": threshold,
                        "mode": mode,
                    }
                    row.update({key: result[key] for key in [
                        "auc_mean", "auc_std", "auc_ci_low", "auc_ci_high",
                        "n_pos", "n_neg", "n_bootstrap", "auc_values",
                    ]})
                    rows.append(row)
    return pd.DataFrame(rows)


def ch_maf_rows(ch_pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    bootstraps = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for maf_group in MAF_BINS:
            subset = ch_pool[ch_pool["maf_bin"].eq(maf_group)].copy()
            for threshold in PIP_THRESHOLDS:
                result = bootstrap_auc(subset, score, threshold, "standard")
                if not result:
                    continue
                summaries.append(
                    {
                        "tool": tool,
                        "ancestry": "CH",
                        "maf_bin": maf_group,
                        "maf_bin_label": MAF_LABELS[maf_group],
                        "pip_threshold": threshold,
                        "n_pos_raw": result["n_pos"],
                        "n_neg_raw": result["n_neg"],
                        "auc_mean": result["auc_mean"],
                        "auc_std": result["auc_std"],
                        "auc_ci_low": result["auc_ci_low"],
                        "auc_ci_high": result["auc_ci_high"],
                        "n_bootstrap": result["n_bootstrap"],
                        "n_pos_balanced_example": result["n_pos_balanced_example"],
                        "n_neg_balanced_example": result["n_neg_balanced_example"],
                    }
                )
                for index, value in enumerate(result["auc_values"].split(",")):
                    bootstraps.append(
                        {
                            "tool": tool,
                            "ancestry": "CH",
                            "maf_bin": maf_group,
                            "maf_bin_label": MAF_LABELS[maf_group],
                            "pip_threshold": threshold,
                            "bootstrap_idx": index,
                            "auc": float(value),
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(bootstraps)


def pair_key_from_members(members: pd.DataFrame) -> pd.Series:
    parsed = members["variant_id"].astype(str).str.extract(
        r"^(chr[^:]+):(\d+)_([^_]+)_([^_]+)$"
    )
    if parsed.isna().any().any():
        raise RuntimeError("Could not parse one or more credible-set variant IDs")
    return (
        parsed[0] + ":" + parsed[1] + ":" + parsed[2] + ":" + parsed[3]
        + ":" + members["gene_id"].astype(str)
    )


def release_cs_sizes(members_path: Path) -> pd.DataFrame:
    members = pd.read_csv(members_path, sep="\t", low_memory=False)
    members = members[
        members["mode"].eq("regular") & members["profile"].eq(FINEMAPPING_PROFILE)
    ].copy()
    members["pair_key"] = pair_key_from_members(members)
    members["cs_size"] = members.groupby(["gene_id", "cs"])["variant_id"].transform("nunique")
    return members[["pair_key", "gene_id", "cs", "cs_size", "pip"]].drop_duplicates("pair_key")


def ch_cs_sensitivity(ch_pool: pd.DataFrame, cs_sizes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = ch_pool.merge(cs_sizes[["pair_key", "cs_size"]], on="pair_key", how="left", validate="many_to_one")
    result_rows = []
    size_rows = []
    for tool in TOOLS:
        score = f"{tool}_score"
        positives = frame[frame["pip"].ge(0.9)].dropna(subset=[score]).copy()
        size_rows.append(
            {
                "tool": tool,
                "ancestry": "CH",
                "n_positive": len(positives),
                "n_singleton": int(positives["cs_size"].eq(1).sum()),
                "pct_singleton": float(100 * positives["cs_size"].eq(1).mean()),
                "median_cs_size": float(positives["cs_size"].median()),
                "p90_cs_size": float(positives["cs_size"].quantile(0.9)),
                "max_cs_size": int(positives["cs_size"].max()),
            }
        )
        for mode in ("standard", "rest"):
            for label, mask in [
                ("all", pd.Series(True, index=frame.index)),
                ("singleton", frame["cs_size"].eq(1)),
            ]:
                result = bootstrap_auc(frame, score, 0.9, mode, positive_mask=mask)
                if result:
                    result_rows.append(
                        {
                            "tool": tool,
                            "ancestry": "CH",
                            "mode": mode,
                            "positive_subset": label,
                            "pip_threshold": 0.9,
                            "n_pos": result["n_pos"],
                            "n_comparison": result["n_neg"],
                            "auc_mean": result["auc_mean"],
                            "auc_std": result["auc_std"],
                            "auc_ci_low": result["auc_ci_low"],
                            "auc_ci_high": result["auc_ci_high"],
                            "n_bootstrap": result["n_bootstrap"],
                            "auc_values": result["auc_values"],
                        }
                    )
    return pd.DataFrame(result_rows), pd.DataFrame(size_rows)


def threshold_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry in ANCESTRIES:
            group = frame[frame["ancestry"].eq(ancestry)].dropna(subset=[score])
            for threshold in PIP_THRESHOLDS:
                selected = group[group["pip"].ge(threshold)]
                row = {"tool": tool, "Group": ancestry, "Threshold": f"≥{threshold}", "Total": len(selected)}
                for distance_bin in TSS_BINS:
                    count = int(selected["tss_distance_bin"].eq(distance_bin).sum())
                    percent = 100 * count / len(selected) if len(selected) else np.nan
                    row[distance_bin] = f"{count} ({percent:.1f}%)" if len(selected) else "0"
                rows.append(row)
    return pd.DataFrame(rows)


def positive_maf_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    observed_ancestries = [
        ancestry for ancestry in ANCESTRIES if frame["ancestry"].eq(ancestry).any()
    ]
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry in observed_ancestries:
            group = frame[frame["ancestry"].eq(ancestry)].dropna(subset=[score, "af", "maf"])
            for threshold in PIP_THRESHOLDS:
                selected = group[group["pip"].ge(threshold)]
                af = selected["af"]
                maf = selected["maf"]
                rows.append(
                    {
                        "tool": tool,
                        "ancestry": ancestry,
                        "pip_threshold": threshold,
                        "n_positives": len(selected),
                        "af_mean": af.mean(),
                        "af_median": af.median(),
                        "af_q25": af.quantile(0.25),
                        "af_q75": af.quantile(0.75),
                        "maf_mean": maf.mean(),
                        "maf_median": maf.median(),
                        "maf_q25": maf.quantile(0.25),
                        "maf_q75": maf.quantile(0.75),
                        "pct_maf_lt_0_01": 100 * maf.lt(0.01).mean() if len(maf) else np.nan,
                        "pct_maf_lt_0_05": 100 * maf.lt(0.05).mean() if len(maf) else np.nan,
                        "pct_maf_ge_0_05": 100 * maf.ge(0.05).mean() if len(maf) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip" if path.suffix == ".gz" else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-scored", type=Path, required=True)
    parser.add_argument("--ch-members", type=Path, required=True)
    parser.add_argument("--base-nominal", type=Path, required=True)
    parser.add_argument("--base-high-pip", type=Path, required=True)
    parser.add_argument("--base-auroc-pool", type=Path, required=True)
    parser.add_argument("--base-auroc-results", type=Path, required=True)
    parser.add_argument("--base-maf-summary", type=Path, required=True)
    parser.add_argument("--base-maf-bootstrap", type=Path, required=True)
    parser.add_argument("--base-positive-maf-summary", type=Path, required=True)
    parser.add_argument("--base-cs-results", type=Path, required=True)
    parser.add_argument("--base-cs-sizes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master_scored, sep="\t", low_memory=False)
    finemap_mask = master["analysis_role"].astype(str).str.contains(
        "finemapping_benchmark", na=False
    )
    finemap_provenance = master.loc[finemap_mask, ["mode", "profile"]]
    wrong_profile = ~(
        finemap_provenance["mode"].eq("regular")
        & finemap_provenance["profile"].eq(FINEMAPPING_PROFILE)
    )
    if finemap_provenance.empty or wrong_profile.any():
        observed = finemap_provenance.value_counts(dropna=False).to_dict()
        raise RuntimeError(
            f"CH fine-mapping rows must use only the regular/{FINEMAPPING_PROFILE} profile; "
            f"observed: {observed}"
        )
    ch_nominal = standardize_ch(master, "nominal_benchmark")
    ch_finemap = add_log_distance(standardize_ch(master, "finemapping_benchmark"))

    base_nominal = standardize_base_nominal(args.base_nominal)
    nominal = pd.concat([base_nominal, ch_nominal], ignore_index=True, sort=False)
    write(nominal, args.output_dir / "nominal_eqtl_10k_predictions_wide.tsv.gz")

    base_high = standardize_base_finemap(args.base_high_pip)
    high = pd.concat(
        [base_high, ch_finemap[ch_finemap["pip"].ge(0.9)]],
        ignore_index=True,
        sort=False,
    )
    write(high, args.output_dir / "finemapped_pip_ge_0.9_predictions_wide.tsv.gz")

    base_pool = add_log_distance(standardize_base_finemap(args.base_auroc_pool))
    pool = pd.concat([base_pool, ch_finemap], ignore_index=True, sort=False)
    write(pool, args.output_dir / "finemapped_auroc_selected_pairs_predictions_wide.tsv.gz")

    nominal_performance = performance_table(nominal, "Nominal eQTLs")
    fine_performance = performance_table(high, "Fine-mapped (PIP≥0.9)")
    fine_corr, fine_conc = finemapped_stratified(high)
    convergence = convergence_table(nominal, high)
    write(nominal_performance, args.output_dir / "nominal_tool_performance_results.tsv")
    write(fine_performance, args.output_dir / "finemapped_tool_performance_results.tsv")
    write(fine_corr, args.output_dir / "finemapped_stratified_correlations.tsv")
    write(fine_conc, args.output_dir / "finemapped_stratified_concordance.tsv")
    write(convergence, args.output_dir / "model_convergence_results.tsv")

    base_auroc = pd.read_csv(args.base_auroc_results, sep="\t")
    auroc = pd.concat([base_auroc[~base_auroc["ancestry"].eq("CH")], ch_auroc_rows(ch_finemap)])
    auroc = auroc.sort_values(["tool", "ancestry", "mode", "pip_threshold"], kind="mergesort")
    write(auroc, args.output_dir / "susie_auroc_bootstrap_results_manuscript.tsv")

    base_maf_summary = pd.read_csv(args.base_maf_summary, sep="\t")
    base_maf_bootstrap = pd.read_csv(args.base_maf_bootstrap, sep="\t")
    ch_maf_summary, ch_maf_bootstrap = ch_maf_rows(ch_finemap)
    maf_summary = pd.concat([base_maf_summary[~base_maf_summary["ancestry"].eq("CH")], ch_maf_summary])
    maf_bootstrap = pd.concat([base_maf_bootstrap[~base_maf_bootstrap["ancestry"].eq("CH")], ch_maf_bootstrap])
    write(maf_summary, args.output_dir / "susie_auroc_maf_stratified_results.tsv")
    write(maf_bootstrap, args.output_dir / "susie_auroc_maf_stratified_bootstrap.tsv.gz")

    cs_sizes = release_cs_sizes(args.ch_members)
    ch_cs_results, ch_cs_sizes = ch_cs_sensitivity(ch_finemap, cs_sizes)
    base_cs_results = pd.read_csv(args.base_cs_results, sep="\t")
    base_cs_sizes = pd.read_csv(args.base_cs_sizes, sep="\t")
    cs_results = pd.concat([base_cs_results[~base_cs_results["ancestry"].eq("CH")], ch_cs_results])
    cs_size_summary = pd.concat([base_cs_sizes[~base_cs_sizes["ancestry"].eq("CH")], ch_cs_sizes])
    write(cs_results, args.output_dir / "credible_set_size_sensitivity_bootstrap.tsv")
    write(cs_size_summary, args.output_dir / "positive_credible_set_size_summary.tsv")

    write(threshold_counts(pool), args.output_dir / "finemapped_eqtl_pair_counts.tsv")
    base_positive_maf = pd.read_csv(args.base_positive_maf_summary, sep="\t")
    ch_positive_maf = positive_maf_summary(ch_finemap)
    positive_maf = pd.concat(
        [base_positive_maf[~base_positive_maf["ancestry"].eq("CH")], ch_positive_maf],
        ignore_index=True,
    )
    positive_maf = positive_maf.reindex(columns=base_positive_maf.columns)
    key_columns = ["tool", "ancestry", "pip_threshold"]
    expected_rows = len(TOOLS) * len(ANCESTRIES) * len(PIP_THRESHOLDS)
    if len(positive_maf) != expected_rows or positive_maf.duplicated(key_columns).any():
        duplicates = positive_maf.loc[
            positive_maf.duplicated(key_columns, keep=False), key_columns
        ].to_dict("records")
        raise RuntimeError(
            "The positive-variant MAF table must contain exactly one row per tool/ancestry/PIP threshold; "
            f"observed {len(positive_maf)} rows (expected {expected_rows}), "
            f"duplicates={duplicates}"
        )
    write(positive_maf, args.output_dir / "positive_variant_maf_distribution.tsv")

    selection_summary = (
        ch_finemap.assign(
            selection_class=np.select(
                [
                    ch_finemap["pip"].ge(0.5),
                    ch_finemap["pip"].lt(0.01),
                ],
                ["positive_pip_ge_0.5", "sampled_negative_pip_lt_0.01"],
                default="sampled_intermediate_pip_0.01_to_0.5",
            )
        )
        .groupby("selection_class", observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    selection_summary.insert(0, "ancestry", "CH")
    write(selection_summary, args.output_dir / "ch_selected_pair_summary.tsv")

    summary = {
        "ch_nominal_rows": len(ch_nominal),
        "ch_high_pip_rows": len(ch_finemap[ch_finemap["pip"].ge(0.9)]),
        "ch_finemap_pool_rows": len(ch_finemap),
        "ch_high_pip_with_borzoi": int(
            ch_finemap[ch_finemap["pip"].ge(0.9)]["borzoi_score"].notna().sum()
        ),
        "ch_high_pip_with_alphagenome": int(
            ch_finemap[ch_finemap["pip"].ge(0.9)]["alphagenome_score"].notna().sum()
        ),
        "profile": f"regular_{FINEMAPPING_PROFILE}",
        "bootstrap_seed": BASE_SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
