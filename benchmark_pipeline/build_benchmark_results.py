#!/usr/bin/env python3
"""Build the benchmark prediction tables and the summary results behind Figures 1, 2, S1, S2 and Tables S1, S8-S10.

AA and NHW rows come from the base benchmark tables built with model_scoring/. CH rows come from the
scored CH pairs (build_scoring_plan.py, collect_scores.py) of the regular-eQTL fine-mapping profile named
by the FINEMAPPING_PROFILE environment variable. Other fine-mapping profiles and interaction-eQTL results
are not mixed into the benchmark.

Gene TSS (hg38_gene_locations.txt) and exact-cohort PLINK allele frequencies are filled for rows that do not
carry them. The fine-mapping pool then follows one comparison rule in every group: low-PIP comparison variants
have PIP < 0.01 and belong to no credible set; intermediate-PIP comparison variants are credible-set members with
0.01 <= PIP < 0.5 (0.01 <= PIP < t at threshold t); positives (PIP >= 0.5) are kept. Distance-matched AUROC,
its MAF-stratified version and the singleton credible-set sensitivity are computed with one routine for all
three groups (100 iterations, seeds 42-141).
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
    """log10 of the absolute TSS distance, floored at 1 bp; missing where the distance is unknown."""
    out = frame.copy()
    distance = pd.Series(np.nan, index=out.index, dtype=float)
    if "distance_to_tss" in out:
        distance = pd.to_numeric(out["distance_to_tss"], errors="coerce").abs()
    if "tss" in out and "position" in out:
        from_tss = (
            pd.to_numeric(out["position"], errors="coerce")
            - pd.to_numeric(out["tss"], errors="coerce")
        ).abs()
        distance = distance.where(distance.notna(), from_tss)
    out["log_distance"] = np.log10(distance.clip(lower=1.0))
    return out


def maf_bin(maf: pd.Series) -> pd.Series:
    values = pd.to_numeric(maf, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    result.loc[(values >= 0) & (values < 0.05)] = "lt0.05"
    result.loc[(values >= 0.05) & (values < 0.2)] = "0.05-0.2"
    result.loc[(values >= 0.2) & (values <= 0.5)] = "ge0.2"
    return result


def fill_tss_and_maf(frame: pd.DataFrame, gene_locations: Path, plink_maf: Path) -> pd.DataFrame:
    """Fill gene TSS, absolute TSS distance, ALT frequency, MAF and MAF bin where a row lacks them.

    TSS comes from hg38_gene_locations.txt; frequencies come from the exact-QTL-cohort PLINK table, joined on
    group and chr:pos_REF_ALT so that the ALT frequency is oriented to the scored allele.
    """
    out = frame.copy().reset_index(drop=True)
    for column in ("tss", "distance_to_tss", "alt_allele_freq", "af", "maf", "maf_bin"):
        if column not in out:
            out[column] = np.nan
    genes = pd.read_csv(gene_locations, sep="\t", dtype={"ensgid": str})
    genes = genes.rename(columns={"ensgid": "gene_id", "TSS": "tss_reference"})
    genes["tss_reference"] = pd.to_numeric(genes["tss_reference"], errors="coerce")
    genes = genes.dropna(subset=["tss_reference"]).drop_duplicates("gene_id")
    tss = out["gene_id"].astype(str).map(genes.set_index("gene_id")["tss_reference"])
    out["tss"] = pd.to_numeric(out["tss"], errors="coerce").where(lambda s: s.notna(), tss)
    position = pd.to_numeric(out["position"], errors="coerce")
    distance = pd.to_numeric(out["distance_to_tss"], errors="coerce").abs()
    out["distance_to_tss"] = distance.where(distance.notna(), (position - out["tss"]).abs())

    freq = pd.read_csv(
        plink_maf, sep="\t", usecols=["ancestry", "plink_variant_id", "a1", "a1_frequency", "maf"]
    )
    if freq.duplicated(["ancestry", "plink_variant_id"]).any():
        raise RuntimeError("The PLINK MAF table has duplicate group/variant rows")
    variant = (
        out["chromosome"].astype(str) + ":" + position.astype("Int64").astype(str)
        + "_" + out["ref_allele"].astype(str) + "_" + out["alt_allele"].astype(str)
    )
    joined = pd.DataFrame({"ancestry": out["ancestry"], "plink_variant_id": variant}).merge(
        freq, on=["ancestry", "plink_variant_id"], how="left", validate="many_to_one"
    )
    a1_frequency = pd.to_numeric(joined["a1_frequency"], errors="coerce")
    alt_frequency = pd.Series(
        np.where(
            joined["a1"].eq(out["alt_allele"]),
            a1_frequency,
            np.where(joined["a1"].eq(out["ref_allele"]), 1 - a1_frequency, np.nan),
        ),
        index=out.index,
    )
    for column, values in (("alt_allele_freq", alt_frequency), ("af", alt_frequency), ("maf", joined["maf"])):
        current = pd.to_numeric(out[column], errors="coerce")
        out[column] = current.where(current.notna(), pd.to_numeric(values, errors="coerce"))
    out["maf_bin"] = out["maf_bin"].where(out["maf_bin"].notna(), maf_bin(out["maf"]))
    if "pair_key" not in out:
        out["pair_key"] = np.nan
    key = (
        out["chromosome"].astype(str) + ":" + position.astype("Int64").astype(str) + ":"
        + out["ref_allele"].astype(str) + ":" + out["alt_allele"].astype(str) + ":" + out["gene_id"].astype(str)
    )
    out["pair_key"] = out["pair_key"].where(out["pair_key"].notna(), key)
    return out


def comparison_class(frame: pd.DataFrame) -> pd.Series:
    """positive (PIP >= 0.5), low (PIP < 0.01) or intermediate (0.01 <= PIP < 0.5)."""
    pip = pd.to_numeric(frame["pip"], errors="coerce")
    return pd.Series(
        np.select([pip.ge(0.5), pip.lt(0.01)], ["positive", "low"], default="intermediate"),
        index=frame.index,
    )


def apply_comparison_rule(frame: pd.DataFrame) -> pd.DataFrame:
    """Low-PIP comparison variants: PIP < 0.01 and in no credible set. Intermediate-PIP comparison variants:
    credible-set members with 0.01 <= PIP < 0.5. Positives (PIP >= 0.5) are kept."""
    member = pd.to_numeric(frame["CS"], errors="coerce").fillna(0).ne(0)
    cls = comparison_class(frame)
    keep = cls.eq("positive") | (cls.eq("low") & ~member) | (cls.eq("intermediate") & member)
    return frame[keep].copy()


def pool_sizes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cls = comparison_class(frame)
    member = pd.to_numeric(frame["CS"], errors="coerce").fillna(0).ne(0)
    for ancestry in ANCESTRIES:
        for tool in TOOLS:
            scored = frame["ancestry"].eq(ancestry) & frame[f"{tool}_score"].notna()
            for name in ("positive", "low", "intermediate"):
                mask = scored & cls.eq(name)
                rows.append(
                    {
                        "ancestry": ancestry,
                        "tool": tool,
                        "pip_class": name,
                        "n_pairs": int(mask.sum()),
                        "n_credible_set_members": int((mask & member).sum()),
                    }
                )
    return pd.DataFrame(rows)


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


def auroc_rows(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry in ANCESTRIES:
            group = pool[pool["ancestry"].eq(ancestry)]
            for threshold in PIP_THRESHOLDS:
                for mode in ("standard", "rest"):
                    result = bootstrap_auc(group, score, threshold, mode)
                    if result:
                        row = {
                            "tool": tool,
                            "ancestry": ancestry,
                            "pip_threshold": threshold,
                            "mode": mode,
                        }
                        row.update({key: result[key] for key in [
                            "auc_mean", "auc_std", "auc_ci_low", "auc_ci_high",
                            "n_pos", "n_neg", "n_bootstrap", "auc_values",
                        ]})
                        rows.append(row)
    return pd.DataFrame(rows)


def maf_rows(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    bootstraps = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry, maf_group in [(a, m) for a in ANCESTRIES for m in MAF_BINS]:
            subset = pool[pool["ancestry"].eq(ancestry) & pool["maf_bin"].eq(maf_group)].copy()
            for threshold in PIP_THRESHOLDS:
                result = bootstrap_auc(subset, score, threshold, "standard")
                if not result:
                    continue
                summaries.append(
                    {
                        "tool": tool,
                        "ancestry": ancestry,
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
                            "ancestry": ancestry,
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
    members["ancestry"] = "CH"
    return members[["ancestry", "pair_key", "gene_id", "cs", "cs_size", "pip"]].drop_duplicates("pair_key")


def base_cs_sizes(path: Path, ancestry: str) -> pd.DataFrame:
    """Credible-set size (distinct variants per gene and credible-set index) from a SuSiE credible-set table
    (MAGENTA_eQTLs_susie_pip_w_cs_<group>.txt; a2 = REF, a1 = ALT)."""
    table = pd.read_csv(path, sep="\t", low_memory=False)
    table["CS"] = pd.to_numeric(table["CS"], errors="coerce")
    table = table[table["CS"].notna() & table["CS"].ne(0)].copy()
    table["gene_id"] = table["molecular_trait_id"].astype(str)
    chromosome = table["chrom"].astype(str).str.replace("^chr", "", regex=True)
    table["pair_key"] = (
        "chr" + chromosome + ":" + pd.to_numeric(table["pos"]).astype(int).astype(str) + ":"
        + table["a2"].astype(str) + ":" + table["a1"].astype(str) + ":" + table["gene_id"]
    )
    table["cs_size"] = table.groupby(["gene_id", "CS"])["variant_id"].transform("nunique")
    table["ancestry"] = ancestry
    table = table.rename(columns={"CS": "cs"})
    return table[["ancestry", "pair_key", "gene_id", "cs", "cs_size", "pip"]].drop_duplicates("pair_key")


def cs_sensitivity(pool: pd.DataFrame, cs_sizes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    size_rows = []
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry in ANCESTRIES:
            sizes = cs_sizes[cs_sizes["ancestry"].eq(ancestry)][["pair_key", "cs_size"]]
            frame = pool[pool["ancestry"].eq(ancestry)].merge(
                sizes, on="pair_key", how="left", validate="many_to_one"
            )
            result, size = cs_rows(frame, tool, score, ancestry)
            result_rows.extend(result)
            size_rows.append(size)
    return pd.DataFrame(result_rows), pd.DataFrame(size_rows)


def cs_rows(frame: pd.DataFrame, tool: str, score: str, ancestry: str) -> tuple[list[dict], dict]:
    positives = frame[frame["pip"].ge(0.9)].dropna(subset=[score]).copy()
    if positives["cs_size"].isna().any():
        raise RuntimeError(f"{ancestry} {tool}: positives without a credible-set size")
    size_row = {
        "tool": tool,
        "ancestry": ancestry,
        "n_positive": len(positives),
        "n_singleton": int(positives["cs_size"].eq(1).sum()),
        "pct_singleton": float(100 * positives["cs_size"].eq(1).mean()),
        "median_cs_size": float(positives["cs_size"].median()),
        "p90_cs_size": float(positives["cs_size"].quantile(0.9)),
        "max_cs_size": int(positives["cs_size"].max()),
    }
    result_rows = []
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
                        "ancestry": ancestry,
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
    return result_rows, size_row


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
    for tool in TOOLS:
        score = f"{tool}_score"
        for ancestry in ANCESTRIES:
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
    parser.add_argument(
        "--base-credible-sets", nargs=2, type=Path, required=True, metavar=("AA_TABLE", "NHW_TABLE"),
        help="MAGENTA_eQTLs_susie_pip_w_cs_AA.txt and MAGENTA_eQTLs_susie_pip_w_cs_NHW.txt",
    )
    parser.add_argument("--gene-locations", type=Path, required=True, help="hg38_gene_locations.txt")
    parser.add_argument("--plink-maf", type=Path, required=True, help="benchmark_plink_maf_exact_qtl_cohorts.tsv.gz")
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
    ch_finemap = standardize_ch(master, "finemapping_benchmark")

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

    base_pool = standardize_base_finemap(args.base_auroc_pool)
    selected = pd.concat([base_pool, ch_finemap], ignore_index=True, sort=False)
    selected = add_log_distance(fill_tss_and_maf(selected, args.gene_locations, args.plink_maf))
    missing = selected[["log_distance", "maf"]].isna().groupby(selected["ancestry"]).sum()
    if missing.to_numpy().any():
        raise RuntimeError(f"Pool rows without TSS distance or MAF:\n{missing}")
    pool = apply_comparison_rule(selected)
    sizes = pool_sizes(selected).merge(
        pool_sizes(pool), on=["ancestry", "tool", "pip_class"], suffixes=("_selected", "_analyzed")
    )
    write(sizes, args.output_dir / "comparison_pool_sizes.tsv")
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

    auroc = auroc_rows(pool).sort_values(["tool", "ancestry", "mode", "pip_threshold"], kind="mergesort")
    write(auroc, args.output_dir / "susie_auroc_bootstrap_results_manuscript.tsv")

    maf_summary, maf_bootstrap = maf_rows(pool)
    write(maf_summary, args.output_dir / "susie_auroc_maf_stratified_results.tsv")
    write(maf_bootstrap, args.output_dir / "susie_auroc_maf_stratified_bootstrap.tsv.gz")

    cs_sizes = pd.concat(
        [
            base_cs_sizes(args.base_credible_sets[0], "AA"),
            release_cs_sizes(args.ch_members),
            base_cs_sizes(args.base_credible_sets[1], "NHW"),
        ],
        ignore_index=True,
    )
    cs_results, cs_size_summary = cs_sensitivity(pool, cs_sizes)
    write(cs_results, args.output_dir / "credible_set_size_sensitivity_bootstrap.tsv")
    write(cs_size_summary, args.output_dir / "positive_credible_set_size_summary.tsv")

    write(threshold_counts(pool), args.output_dir / "finemapped_eqtl_pair_counts.tsv")
    positive_maf = positive_maf_summary(pool)
    key_columns = ["tool", "ancestry", "pip_threshold"]
    expected_rows = len(TOOLS) * len(ANCESTRIES) * len(PIP_THRESHOLDS)
    if len(positive_maf) != expected_rows or positive_maf.duplicated(key_columns).any():
        raise RuntimeError(
            "The positive-variant MAF table must contain exactly one row per tool/ancestry/PIP threshold; "
            f"observed {len(positive_maf)} rows (expected {expected_rows})"
        )
    write(positive_maf, args.output_dir / "positive_variant_maf_distribution.tsv")

    summary = {
        "ch_nominal_rows": len(ch_nominal),
        "ch_high_pip_rows": len(ch_finemap[ch_finemap["pip"].ge(0.9)]),
        "pool_rows_selected": len(selected),
        "pool_rows_analyzed": len(pool),
        "pool_rows_analyzed_by_group": pool["ancestry"].value_counts().sort_index().to_dict(),
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
