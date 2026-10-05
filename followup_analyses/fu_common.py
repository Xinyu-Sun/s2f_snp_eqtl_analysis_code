"""Shared loaders and the distance-matched AUROC used by the follow-up analyses.

Reproduces the manuscript design (scripts/analyze_susie_auroc_bootstrap.py and the benchmark builder):
positives = PIP >= t with a model score; comparison variants = PIP < 0.01 (low) or 0.01 <= PIP < t (intermediate);
log10(|TSS distance|, floored at 1 bp) split into 10 equal-count bins whose edges are the deciles of the pooled
positive and comparison distances; within each bin and iteration, positives are kept and comparison pairs are
sampled with replacement to the positive count; AUROC on |score|.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GROUPS = ["AA", "CH", "NHW"]
MODELS = {"borzoi": "borzoi_score", "alphagenome": "alphagenome_score"}
N_BINS = 10
TSS_BINS = ["0-3kb", "3-12kb", "12-35kb", ">35kb"]


def load_pairs(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["ancestry"].isin(GROUPS)].copy()
    for c in ["pip", "beta", "std_error", "distance_to_tss", "borzoi_score", "alphagenome_score", "maf"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["abs_z"] = (df["beta"] / df["std_error"]).abs()
    # The benchmark table carries TSS distance and cohort MAF for every row; rows that lack them are filled from the
    # pipeline's gene TSS table (FU_GENE_TSS) and the exact-cohort PLINK MAF table (FU_MAF, joined on chr:pos_REF_ALT).
    tss_path = os.environ.get("FU_GENE_TSS")
    if tss_path:
        g = pd.read_csv(tss_path, sep=None, engine="python")
        g = g.rename(columns={"ensgid": "gene_id", "TSS": "tss_ref"})
        g["gene_id"] = g["gene_id"].astype(str).str.split(".").str[0]
        g["tss_ref"] = pd.to_numeric(g["tss_ref"], errors="coerce")
        g = g.dropna(subset=["tss_ref"]).drop_duplicates("gene_id")[["gene_id", "tss_ref"]]
        df = df.merge(g, on="gene_id", how="left")
        need = df["distance_to_tss"].isna() & df["tss_ref"].notna()
        df.loc[need, "distance_to_tss"] = (df.loc[need, "position"].astype(float) - df.loc[need, "tss_ref"]).abs()
    maf_path = os.environ.get("FU_MAF")
    if maf_path:
        m = pd.read_csv(maf_path, sep="\t", usecols=["ancestry", "plink_variant_id", "maf"]).rename(columns={"maf": "maf_cohort"})
        m = m.drop_duplicates(["ancestry", "plink_variant_id"])
        df["plink_variant_id"] = df["chromosome"].astype(str) + ":" + df["position"].astype(str) + "_" + df["ref_allele"] + "_" + df["alt_allele"]
        df = df.merge(m, on=["ancestry", "plink_variant_id"], how="left")
        df["maf"] = df["maf"].where(df["maf"].notna(), df["maf_cohort"])
    # pipeline convention: log10 of the absolute distance, floored at 1 bp
    df["log_dist"] = np.log10(df["distance_to_tss"].abs().clip(lower=1))
    df["variant_key"] = df["chromosome"].astype(str) + ":" + df["position"].astype(str) + "_" + df["ref_allele"] + "_" + df["alt_allele"]
    return df


def split_classes(df, model_col, t=0.9):
    d = df.dropna(subset=[model_col, "log_dist", "pip"])
    pos = d[d["pip"] >= t]
    low = d[d["pip"] < 0.01]
    mid = d[(d["pip"] >= 0.01) & (d["pip"] < t)]
    return pos, low, mid


def distance_bins(pos, comp):
    """Equal-count bin edges: deciles of the pooled log-distance of positives and comparison variants."""
    allv = np.concatenate([pos["log_dist"].values, comp["log_dist"].values])
    allv = allv[np.isfinite(allv)]
    edges = np.unique(np.percentile(allv, np.linspace(0, 100, N_BINS + 1)))
    return edges


def _bin_index(values, edges):
    """Right-closed bins with the lowest edge included (pandas.cut(..., include_lowest=True)); -1 if outside."""
    idx = np.digitize(values, edges, right=True) - 1
    idx[values == edges[0]] = 0
    idx[(values < edges[0]) | (values > edges[-1])] = -1
    return idx


def matched_auroc(pos, comp, model_col, rng, edges=None, resample_pos_genes=False, pos_weight_col=None):
    """One distance-matched AUROC draw. If resample_pos_genes, positive genes are
    resampled with replacement first (cluster bootstrap over genes). If pos_weight_col is given,
    positives are weighted by that column (comparison variants have weight 1)."""
    if len(pos) == 0 or len(comp) == 0:
        return np.nan, 0
    if resample_pos_genes:
        genes = pos["gene_id"].unique()
        picked = rng.choice(genes, size=len(genes), replace=True)
        counts = pd.Series(picked).value_counts()
        pos = pos.loc[pos["gene_id"].isin(counts.index)]
        pos = pos.loc[pos.index.repeat(pos["gene_id"].map(counts).astype(int))]
    if edges is None:
        edges = distance_bins(pos, comp)
    if len(edges) < 2:
        return np.nan, 0
    pb = _bin_index(pos["log_dist"].values, edges)
    cb = _bin_index(comp["log_dist"].values, edges)
    comp_scores = np.abs(comp[model_col].values)
    pos_scores = np.abs(pos[model_col].values)
    pos_w = pos[pos_weight_col].values if pos_weight_col else np.ones(len(pos))
    ys, ss, ws = [], [], []
    for b in range(len(edges) - 1):
        pidx = np.where(pb == b)[0]
        cidx = np.where(cb == b)[0]
        if len(pidx) == 0 or len(cidx) == 0:
            continue
        draw = rng.choice(cidx, size=len(pidx), replace=True)
        ss.extend(pos_scores[pidx]); ys.extend([1] * len(pidx)); ws.extend(pos_w[pidx])
        ss.extend(comp_scores[draw]); ys.extend([0] * len(pidx)); ws.extend([1.0] * len(pidx))
    if len(set(ys)) < 2:
        return np.nan, 0
    return roc_auc_score(ys, ss, sample_weight=ws if pos_weight_col else None), int(sum(ys))


def group_rng(seed, group):
    """Independent random stream for each group."""
    return np.random.default_rng([seed, GROUPS.index(group)])


def bootstrap_p(d):
    """Two-sided bootstrap p for a difference: twice the smaller share of replicates on either side of zero,
    with the plus-one adjustment, so the smallest attainable value is 2 / (B + 1)."""
    d = np.asarray(d, dtype=float); d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    return float(min(1.0, 2 * min(((d <= 0).sum() + 1) / (len(d) + 1), ((d >= 0).sum() + 1) / (len(d) + 1))))


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return (np.nan, np.nan)
    return (np.percentile(x, lo), np.percentile(x, hi))
