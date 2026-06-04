#!/usr/bin/env python3
"""
Prepare an expanded SuSiE prediction set for AUROC analyses.

This is a larger, AUROC-focused rerun. It uses the exact scored-positive
universe from the original outputs_susie table (PIP >= 0.5, known TSS bin),
then samples low-PIP and intermediate-PIP background variants within ancestry
x TSS x MAF strata so both vanilla distance-matched and MAF-stratified AUROC
have a better negative/intermediate universe.

Outputs default to results_auroc_expanded/ and do not overwrite results/.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
COMMON_BASE = Path("/mnt/vstor/Data14/xxs410_xinyu/seq2func_related/common_eqtl_snp_proj")
OLD_COMBINED = COMMON_BASE / "outputs_susie" / "analysis" / "susie_s2f_combined.tsv.gz"
CURRENT_REDO_COMBINED = BASE_DIR / "results" / "analysis" / "susie_s2f_combined.tsv.gz"
TOOLS = ["borzoi", "alphagenome"]
ANCESTRIES = ["AA", "CH", "NHW"]
RAW_SUSIE_FILES = {
    "AA": "/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes/MAGENTA_eQTLs_susie_pip_w_cs_AA.txt",
    "CH": "/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes/MAGENTA_eQTLs_susie_pip_w_cs_CH.txt",
    "NHW": "/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes/MAGENTA_eQTLs_susie_pip_w_cs_NHW.txt",
}
MAF_BIN_DEFS = [
    ("lt0.05", 0.0, 0.05),
    ("0.05-0.2", 0.05, 0.2),
    ("ge0.2", 0.2, 0.5),
]
PIP_CLASSES = ["low", "intermediate"]

sys.path.append(str(BASE_DIR / "code"))
from prepare_base_finemapped_eqtl_batches import (  # noqa: E402
    DEFAULT_SEED,
    add_tss_info,
    list_susie_files,
    read_susie_header,
    _normalize_chr,
    _variant_key_cols,
    _write_vcf,
    _safe_slug,
)


def compute_maf_bin(maf: pd.Series) -> pd.Series:
    binned = pd.Series(pd.NA, index=maf.index, dtype="object")
    for label, lower, upper in MAF_BIN_DEFS:
        if label == "ge0.2":
            mask = (maf >= lower) & (maf <= upper)
        else:
            mask = (maf >= lower) & (maf < upper)
        binned.loc[mask] = label
    return binned


def load_all_variant_susie_with_af(ancestries: list[str]) -> pd.DataFrame:
    keep_cols = [
        "chrom", "pos", "a1", "a2", "molecular_trait_id", "variant_id",
        "pvalue", "beta", "se", "qvalue", "pip", "CS", "af",
    ]
    dfs = []
    for ancestry in ancestries:
        files = list_susie_files(ancestry)
        print(f"[load] {ancestry}: {len(files):,} per-gene all-variant files")
        anc_parts = []
        n_missing = 0
        for i, path in enumerate(files, start=1):
            if not path.exists():
                n_missing += 1
                continue
            header = read_susie_header(path)
            usecols = [c for c in keep_cols if c in header]
            df = pd.read_csv(path, sep="\t", usecols=usecols, compression="gzip")
            if df.empty:
                continue
            df["source_file"] = str(path)
            anc_parts.append(df)
            if i % 250 == 0:
                print(f"  {ancestry}: read {i:,}/{len(files):,} files")

        if not anc_parts:
            continue
        if n_missing:
            print(f"  {ancestry}: skipped {n_missing:,} missing expected files")

        anc_df = pd.concat(anc_parts, ignore_index=True)
        anc_df = anc_df.rename(
            columns={
                "chrom": "chromosome",
                "pos": "position",
                "a2": "ref_allele",
                "a1": "alt_allele",
                "molecular_trait_id": "gene_id",
                "se": "std_error",
            }
        )
        anc_df["ancestry"] = ancestry
        anc_df["population"] = ancestry
        anc_df["chromosome"] = anc_df["chromosome"].map(_normalize_chr)
        anc_df["position"] = pd.to_numeric(anc_df["position"], errors="coerce").astype("Int64")
        anc_df["pip"] = pd.to_numeric(anc_df["pip"], errors="coerce")
        anc_df["CS"] = pd.to_numeric(anc_df.get("CS", 0), errors="coerce").fillna(0).astype(int)
        anc_df = anc_df.dropna(subset=["position", "pip", "gene_id", "ref_allele", "alt_allele", "af"])
        anc_df["position"] = anc_df["position"].astype("int64")
        dfs.append(anc_df)
        print(f"  {ancestry}: loaded {len(anc_df):,} variant-gene rows")

    if not dfs:
        raise RuntimeError("No SuSiE all-variant files were loaded")

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=_variant_key_cols() + ["pip"], keep="first")
    print(f"[load] combined {len(combined):,} rows after duplicate cleanup ({before:,} before)")
    return combined


def classify_for_auroc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["af"] = pd.to_numeric(out["af"], errors="coerce")
    out["maf"] = np.minimum(out["af"], 1 - out["af"])
    out["maf_bin"] = compute_maf_bin(out["maf"])
    in_cs = out["CS"].fillna(0).astype(int).ne(0)
    out["pip_class"] = "drop"
    out.loc[(out["pip"] >= 0.5) & in_cs, "pip_class"] = "positive"
    out.loc[out["pip"] < 0.01, "pip_class"] = "low"
    out.loc[(out["pip"] >= 0.01) & (out["pip"] < 0.5), "pip_class"] = "intermediate"
    out = out[out["pip_class"] != "drop"].copy()
    out = out[out["maf_bin"].notna() & out["tss_distance_bin"].ne("unknown")].copy()
    out["cs_state"] = np.where(out["CS"].fillna(0).astype(int) != 0, "cs", "non_cs")
    return out


def coalesce_scored_column(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in df.columns and fallback in df.columns:
        return df[preferred].where(df[preferred].notna(), df[fallback])
    if preferred in df.columns:
        return df[preferred]
    return df[fallback]


def normalize_scored_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = pd.DataFrame(
        {
            "tool": out["tool"],
            "chromosome": coalesce_scored_column(out, "chromosome", "chr"),
            "position": coalesce_scored_column(out, "position", "pos"),
            "ref_allele": coalesce_scored_column(out, "ref_allele", "ref"),
            "alt_allele": coalesce_scored_column(out, "alt_allele", "alt"),
            "gene_id": out["gene_id"],
            "ancestry": out["ancestry"],
        }
    )
    keys["chromosome"] = keys["chromosome"].map(_normalize_chr)
    keys["position"] = pd.to_numeric(keys["position"], errors="coerce")
    keys = keys.dropna(subset=["position", "gene_id", "ancestry", "tool"])
    keys["position"] = keys["position"].astype("int64")
    return keys.drop_duplicates()


def normalize_scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()
    out = pd.DataFrame(
        {
            "tool": raw["tool"],
            "chromosome": coalesce_scored_column(raw, "chromosome", "chr"),
            "position": coalesce_scored_column(raw, "position", "pos"),
            "ref_allele": coalesce_scored_column(raw, "ref_allele", "ref"),
            "alt_allele": coalesce_scored_column(raw, "alt_allele", "alt"),
            "gene_id": raw["gene_id"],
            "ancestry": raw["ancestry"],
            "pip": raw["pip"],
            "CS": raw["CS"],
            "tss_distance_bin": raw["tss_distance_bin"],
        }
    )
    out["chromosome"] = out["chromosome"].map(_normalize_chr)
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    out = out.dropna(subset=["position", "gene_id", "ancestry", "tool"])
    out["position"] = out["position"].astype("int64")
    return out


def build_join_key(df: pd.DataFrame) -> pd.Series:
    chrom = df["chromosome"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["position"], errors="coerce").astype("Int64").astype(str)
    return (
        chrom + "_"
        + pos + "_"
        + df["ref_allele"].astype(str) + "_"
        + df["alt_allele"].astype(str) + "_"
        + df["gene_id"].astype(str) + "_"
        + df["ancestry"].astype(str)
    )


def load_raw_maf() -> pd.DataFrame:
    parts = []
    for ancestry, path in RAW_SUSIE_FILES.items():
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["chrom", "pos", "a1", "a2", "molecular_trait_id", "af"],
        )
        df = df.rename(
            columns={
                "chrom": "chromosome",
                "pos": "position",
                "a2": "ref_allele",
                "a1": "alt_allele",
                "molecular_trait_id": "gene_id",
            }
        )
        df["ancestry"] = ancestry
        df["chromosome"] = df["chromosome"].map(_normalize_chr)
        df["af"] = pd.to_numeric(df["af"], errors="coerce")
        df["maf"] = np.minimum(df["af"], 1 - df["af"])
        df["join_key"] = build_join_key(df)
        parts.append(df[["join_key", "af", "maf"]])
    return pd.concat(parts, ignore_index=True).drop_duplicates("join_key")


def load_original_positive_tool_keys() -> pd.DataFrame:
    """Load exact original scored positives used by the manuscript TSS count table."""
    df = pd.read_csv(OLD_COMBINED, sep="\t", low_memory=False)
    df = df[df["tool"].isin(TOOLS) & df["ancestry"].isin(ANCESTRIES)].copy()
    df["pip"] = pd.to_numeric(df["pip"], errors="coerce")
    df["CS"] = pd.to_numeric(df["CS"], errors="coerce").fillna(0).astype(int)
    df = df[
        (df["pip"] >= 0.5)
        & df["tss_distance_bin"].notna()
        & df["tss_distance_bin"].ne("unknown")
    ].copy()
    # The original high-PIP known-TSS rows are already credible-set rows, but
    # keep this guard so the definition is explicit.
    df = df[df["CS"].ne(0)].copy()
    keys = normalize_scored_keys(df)
    print(f"[positive source] original scored positive tool rows: {len(keys):,}")
    return keys


def load_original_positive_pairs() -> pd.DataFrame:
    df = pd.read_csv(OLD_COMBINED, sep="\t", low_memory=False)
    df = df[df["tool"].isin(TOOLS) & df["ancestry"].isin(ANCESTRIES)].copy()
    df["pip"] = pd.to_numeric(df["pip"], errors="coerce")
    df["CS"] = pd.to_numeric(df["CS"], errors="coerce").fillna(0).astype(int)
    df = df[
        (df["pip"] >= 0.5)
        & df["tss_distance_bin"].notna()
        & df["tss_distance_bin"].ne("unknown")
        & df["CS"].ne(0)
    ].copy()
    pos = normalize_scored_rows(df)
    pos = pos.drop_duplicates(subset=_variant_key_cols(), keep="first")
    pos["join_key"] = build_join_key(pos)
    pos = pos.merge(load_raw_maf(), on="join_key", how="left")
    missing_maf = int(pos["maf"].isna().sum())
    if missing_maf:
        raise RuntimeError(f"{missing_maf} original positive pairs are missing raw MAF")
    pos["maf_bin"] = compute_maf_bin(pos["maf"])
    pos = pos[pos["maf_bin"].notna()].copy()
    pos["pip_class"] = "positive"
    pos["redo_label"] = "positive_pip_ge_0.5_in_original_table"
    pos["population"] = pos["ancestry"]
    pos["variant_id"] = (
        pos["chromosome"].astype(str) + ":"
        + pos["position"].astype("int64").astype(str) + "_"
        + pos["ref_allele"].astype(str) + "_"
        + pos["alt_allele"].astype(str)
    )
    for col in ["pvalue", "beta", "std_error", "qvalue", "source_file", "tss", "distance_to_tss", "cs_state"]:
        if col not in pos.columns:
            pos[col] = pd.NA
    pos["cs_state"] = "cs"
    return pos


def load_scored_tool_keys(extra_combined: Path | None) -> pd.DataFrame:
    pieces = []
    for path in [OLD_COMBINED, CURRENT_REDO_COMBINED, extra_combined]:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df = df[df["tool"].isin(TOOLS) & df["ancestry"].isin(ANCESTRIES)].copy()
        pieces.append(normalize_scored_keys(df))
        print(f"[reuse] loaded scored keys from {path}: {len(pieces[-1]):,}")
    if not pieces:
        return pd.DataFrame(columns=["tool"] + _variant_key_cols())
    return pd.concat(pieces, ignore_index=True).drop_duplicates()


def add_reuse_counts(df: pd.DataFrame, scored_tool_keys: pd.DataFrame) -> pd.DataFrame:
    reuse_counts = (
        scored_tool_keys.groupby(_variant_key_cols(), as_index=False)
        .agg(reuse_tool_count=("tool", "nunique"))
    )
    out = df.merge(reuse_counts, on=_variant_key_cols(), how="left")
    out["reuse_tool_count"] = out["reuse_tool_count"].fillna(0).astype(int)
    return out


def reduce_unique_pairs(df: pd.DataFrame) -> pd.DataFrame:
    cols = _variant_key_cols() + [
        "variant_id", "pvalue", "beta", "std_error", "qvalue", "pip", "CS",
        "source_file", "population", "tss", "distance_to_tss", "tss_distance_bin",
        "af", "maf", "maf_bin", "cs_state", "pip_class", "reuse_tool_count",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out = out.sort_values(
        ["pip", "reuse_tool_count", "cs_state"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return out.drop_duplicates(subset=_variant_key_cols(), keep="first")


def sample_background(
    df: pd.DataFrame,
    *,
    positives: pd.DataFrame,
    pip_class: str,
    ratio: int,
    min_per_stratum: int,
    max_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    strata_cols = ["ancestry", "tss_distance_bin", "maf_bin"]
    pos_counts = positives.groupby(strata_cols, observed=True).size().reset_index(name="n_pos")

    pool = df[df["pip_class"] == pip_class].copy()
    for row in pos_counts.itertuples(index=False):
        subset = pool[
            (pool["ancestry"] == row.ancestry)
            & (pool["tss_distance_bin"] == row.tss_distance_bin)
            & (pool["maf_bin"] == row.maf_bin)
        ].copy()
        if subset.empty:
            continue

        target = max(int(row.n_pos) * ratio, min_per_stratum)
        target = min(target, max_per_stratum, len(subset))

        subset = subset.sort_values(
            ["reuse_tool_count", "cs_state", "pip"],
            ascending=[False, True, True if pip_class == "low" else False],
            kind="mergesort",
        )

        reused = subset[subset["reuse_tool_count"] > 0]
        unscored = subset[subset["reuse_tool_count"] == 0]
        chosen = []
        n_reuse = min(len(reused), target)
        if n_reuse:
            chosen.append(reused.iloc[:n_reuse])

        remaining = target - n_reuse
        if remaining > 0 and len(unscored):
            weights = np.where(unscored["cs_state"].eq("non_cs"), 2.0, 1.0)
            idx = rng.choice(
                unscored.index.to_numpy(),
                size=min(remaining, len(unscored)),
                replace=False,
                p=weights / weights.sum(),
            )
            chosen.append(unscored.loc[idx])

        if chosen:
            parts.append(pd.concat(chosen, ignore_index=True))

    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=_variant_key_cols(), keep="first")


@dataclass(frozen=True)
class ChunkConfig:
    results_dir: Path
    pairs_per_chunk: int
    seed: int


def write_prediction_inputs(missing_pairs: pd.DataFrame, cfg: ChunkConfig) -> pd.DataFrame:
    inputs_dir = cfg.results_dir / "inputs"
    job_rows = []
    for ancestry, anc_df in missing_pairs.groupby("ancestry", sort=True):
        anc_df = anc_df.sort_values(_variant_key_cols(), kind="mergesort").reset_index(drop=True)
        anc_df["chunk_index"] = (anc_df.index // cfg.pairs_per_chunk).astype(int)
        n_chunks = int(anc_df["chunk_index"].max()) + 1 if len(anc_df) else 0
        print(f"[chunks] {ancestry}: {len(anc_df):,} missing pairs -> {n_chunks:,} chunks per tool")

        for chunk_index in range(n_chunks):
            chunk = anc_df[anc_df["chunk_index"] == chunk_index].drop(columns=["chunk_index"]).copy()
            vars_df = (
                chunk[["chromosome", "position", "ref_allele", "alt_allele"]]
                .drop_duplicates()
                .sort_values(["chromosome", "position", "ref_allele", "alt_allele"], kind="mergesort")
            )
            for tool in TOOLS:
                chunk_dir = inputs_dir / tool / _safe_slug(ancestry) / f"chunk_{chunk_index:04d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                pairs_path = chunk_dir / "pairs.tsv"
                vcf_path = chunk_dir / "variants.vcf"
                chunk.to_csv(pairs_path, sep="\t", index=False)
                _write_vcf(vars_df, vcf_path)
                job_rows.append(
                    {
                        "tool": tool,
                        "ancestry": ancestry,
                        "chunk_index": chunk_index,
                        "pairs_tsv": str(pairs_path.resolve()),
                        "variants_vcf": str(vcf_path.resolve()),
                        "n_pairs": int(len(chunk)),
                        "n_unique_variants": int(len(vars_df)),
                        "seed": cfg.seed,
                    }
                )

    jobs = pd.DataFrame(job_rows)
    manifest_dir = cfg.results_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(manifest_dir / "jobs.tsv", sep="\t", index=False)
    (manifest_dir / "seed.txt").write_text(f"{cfg.seed}\n")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=BASE_DIR / "results_auroc_expanded")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED + 101)
    parser.add_argument("--low_ratio", type=int, default=50)
    parser.add_argument("--intermediate_ratio", type=int, default=25)
    parser.add_argument("--min_low_per_stratum", type=int, default=200)
    parser.add_argument("--min_intermediate_per_stratum", type=int, default=100)
    parser.add_argument("--max_low_per_stratum", type=int, default=3000)
    parser.add_argument("--max_intermediate_per_stratum", type=int, default=1500)
    parser.add_argument("--pairs_per_chunk", type=int, default=500)
    parser.add_argument("--extra_combined", type=Path, default=None)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "analysis").mkdir(parents=True, exist_ok=True)

    all_variants = load_all_variant_susie_with_af(ANCESTRIES)
    all_variants = add_tss_info(all_variants)
    all_variants = classify_for_auroc(all_variants)
    scored_tool_keys = load_scored_tool_keys(args.extra_combined)
    all_variants = add_reuse_counts(all_variants, scored_tool_keys)
    all_variants = reduce_unique_pairs(all_variants)

    positive_tool_keys = load_original_positive_tool_keys()
    positives = load_original_positive_pairs()
    positives = add_reuse_counts(positives.drop(columns=["reuse_tool_count"], errors="ignore"), scored_tool_keys)

    low = sample_background(
        all_variants,
        positives=positives,
        pip_class="low",
        ratio=args.low_ratio,
        min_per_stratum=args.min_low_per_stratum,
        max_per_stratum=args.max_low_per_stratum,
        seed=args.seed,
    )
    intermediate = sample_background(
        all_variants,
        positives=positives,
        pip_class="intermediate",
        ratio=args.intermediate_ratio,
        min_per_stratum=args.min_intermediate_per_stratum,
        max_per_stratum=args.max_intermediate_per_stratum,
        seed=args.seed + 10_000,
    )

    selected = pd.concat([positives, low, intermediate], ignore_index=True)
    selected = selected.drop_duplicates(subset=_variant_key_cols(), keep="first")
    selected["redo_label"] = selected["pip_class"].map(
        {
            "positive": "positive_pip_ge_0.5_in_original_table",
            "low": "sampled_negative_pip_lt_0.01",
            "intermediate": "sampled_intermediate_pip_0.01_to_0.5",
        }
    )
    selected = selected.sort_values(_variant_key_cols(), kind="mergesort").reset_index(drop=True)
    selected.to_csv(args.results_dir / "selected_pairs.tsv.gz", sep="\t", index=False, compression="gzip")
    positive_tool_keys.to_csv(
        args.results_dir / "analysis" / "original_positive_tool_keys.tsv",
        sep="\t",
        index=False,
    )

    all_tool_pairs = pd.concat([selected.assign(tool=tool) for tool in TOOLS], ignore_index=True)
    reuse = all_tool_pairs.merge(scored_tool_keys, on=["tool"] + _variant_key_cols(), how="inner")
    missing_tool_pairs = all_tool_pairs.merge(
        reuse.assign(already_scored=1),
        on=["tool"] + _variant_key_cols(),
        how="left",
    )
    missing_tool_pairs = missing_tool_pairs[missing_tool_pairs["already_scored"].isna()].drop(columns=["already_scored"])

    reusable_pair_keys = reuse[_variant_key_cols()].drop_duplicates()
    missing_pair_keys = missing_tool_pairs[_variant_key_cols()].drop_duplicates()
    predicted_pairs = selected.merge(reusable_pair_keys, on=_variant_key_cols(), how="inner")
    missing_pairs = selected.merge(missing_pair_keys, on=_variant_key_cols(), how="inner")

    predicted_pairs.to_csv(args.results_dir / "predicted_pairs_reused.tsv.gz", sep="\t", index=False, compression="gzip")
    missing_pairs.to_csv(args.results_dir / "missing_pairs_for_prediction.tsv.gz", sep="\t", index=False, compression="gzip")
    jobs = write_prediction_inputs(
        missing_pairs,
        ChunkConfig(args.results_dir, args.pairs_per_chunk, args.seed),
    )

    summary = selected.groupby(["ancestry", "redo_label"], observed=True).size().reset_index(name="n")
    summary.to_csv(args.results_dir / "analysis" / "selected_pair_summary.tsv", sep="\t", index=False)

    strata_summary = (
        selected.groupby(["ancestry", "redo_label", "tss_distance_bin", "maf_bin"], observed=True)
        .size()
        .reset_index(name="n")
    )
    strata_summary.to_csv(args.results_dir / "analysis" / "selected_pair_strata_summary.tsv", sep="\t", index=False)

    reused_tool_keys = reuse[["tool"] + _variant_key_cols()].drop_duplicates()
    tool_summary = all_tool_pairs.merge(
        reused_tool_keys.assign(reused=1),
        on=["tool"] + _variant_key_cols(),
        how="left",
    )
    tool_summary["reused"] = tool_summary["reused"].fillna(0).astype(int)
    tool_summary = (
        tool_summary.groupby(["tool", "ancestry", "redo_label"], observed=True)
        .agg(requested_pairs=("gene_id", "size"), reused_pairs=("reused", "sum"))
        .reset_index()
    )
    tool_summary["new_pairs"] = tool_summary["requested_pairs"] - tool_summary["reused_pairs"]
    tool_summary.to_csv(args.results_dir / "analysis" / "prediction_request_summary.tsv", sep="\t", index=False)

    print("\n[summary]")
    print(summary.to_string(index=False))
    print("\n[prediction requests]")
    print(tool_summary.to_string(index=False))
    print(f"\n[reuse] selected pairs with any old/current score: {len(predicted_pairs):,}")
    print(f"[todo] pairs needing at least one prediction: {len(missing_pairs):,}")
    print(f"[todo] tool jobs written: {len(jobs):,} -> {args.results_dir / 'manifests/jobs.tsv'}")
    for tool in TOOLS:
        n = int((jobs["tool"] == tool).sum()) if not jobs.empty else 0
        print(f"  {tool}: SLURM/local task ids 1-{n}" if n else f"  {tool}: no new jobs")


if __name__ == "__main__":
    main()
