#!/usr/bin/env python3
"""
Prepare fine-mapped eQTL benchmark inputs from per-gene all-variant PIP files.

The original `outputs_susie` input used top-level MAGENTA_eQTLs_susie_pip_w_cs
files, which only contain variants with CS != 0. This script instead reads
the per-gene SuSiE PIP files (all variants), keeps all PIP >= 0.5
variants, samples additional low-PIP negatives, samples intermediate-PIP
variants for manuscript-style AUROC comparisons, and writes prediction chunks.

Outputs live under the result directory:
  results/selected_pairs.tsv.gz
  results/predicted_pairs_reused.tsv.gz
  results/missing_pairs_for_prediction.tsv.gz
  results/inputs/{tool}/{ancestry}/chunk_####/{pairs.tsv, variants.vcf}
  results/manifests/jobs.tsv
"""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TOOLS = ["borzoi", "alphagenome"]
DEFAULT_ANCESTRIES = ["AA", "CH", "NHW", "PER"]
DEFAULT_SEED = 20260423
SUSIE_ROOT = Path("/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes")
EGENE_LIST_DIR = SUSIE_ROOT / "eGenes"
EGENE_LISTS = {
    "AA": EGENE_LIST_DIR / "aa_no_int_Nominal_eGenes.tsv",
    "CH": EGENE_LIST_DIR / "ch_no_int_Nominal_eGenes.tsv",
    "NHW": EGENE_LIST_DIR / "nhw_no_int_Nominal_eGenes.tsv",
    "PER": EGENE_LIST_DIR / "per_no_int_Nominal_eGenes.tsv",
}
OLD_COMBINED = Path(
    "/mnt/vstor/Data14/xxs410_xinyu/seq2func_related/common_eqtl_snp_proj/"
    "outputs_susie/analysis/susie_s2f_combined.tsv.gz"
)
GENE_LOC_FILE = Path(
    "/mnt/vstor/Data14/xxs410_xinyu/seq2func_related/common_eqtl_snp_proj/"
    "inputs/hg38_gene_locations.txt"
)


def _safe_slug(s: str) -> str:
    return str(s).strip().replace("/", "_").replace(" ", "_").replace(">", "gt").replace("<", "lt")


def _normalize_chr(chromosome: str) -> str:
    chrom = str(chromosome)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def _variant_key_cols() -> list[str]:
    return ["chromosome", "position", "ref_allele", "alt_allele", "gene_id", "ancestry"]


def _write_vcf(df_vars: pd.DataFrame, out_vcf: Path) -> None:
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    with out_vcf.open("w") as f:
        f.write("##fileformat=VCFv4.2\n")
        for row in df_vars.itertuples(index=False):
            vcf_id = f"{row.chromosome}_{int(row.position)}_{row.ref_allele}_{row.alt_allele}"
            f.write(f"{row.chromosome}\t{int(row.position)}\t{vcf_id}\t{row.ref_allele}\t{row.alt_allele}\t.\t.\n")


def tss_distance_bin(distance: float) -> str:
    if pd.isna(distance):
        return "unknown"
    distance = abs(float(distance))
    if distance <= 3000:
        return "0-3kb"
    if distance <= 12000:
        return "3-12kb"
    if distance <= 35000:
        return "12-35kb"
    return ">35kb"


def list_susie_files(ancestry: str) -> list[Path]:
    list_path = EGENE_LISTS.get(ancestry)
    if list_path is not None and list_path.exists():
        eg = pd.read_csv(list_path, sep="\t", dtype=str)
        files = []
        for row in eg.itertuples(index=False):
            hgnc = str(row.hgnc_symbol)
            gene = str(row.molecular_trait_id)
            chrom = str(row.chr)
            anc = str(row.ancestry)
            if hgnc.lower() == "nan" or hgnc == "":
                hgnc = "NA"
            if gene in {"NA", "nan", ""} or chrom in {"NA", "nan", ""}:
                continue
            files.append(
                SUSIE_ROOT
                / anc
                / f"{hgnc}_{gene}_chr_{chrom}_{anc}"
                / f"MAGENTA_eQTL_{gene}_{hgnc}_chr_{chrom}_{anc}_susie_pip_w_cs_corrected_n.txt.gz"
            )
        return files
    return sorted((SUSIE_ROOT / ancestry).glob("*/*_susie_pip_w_cs_corrected_n.txt.gz"))


def read_susie_header(path: Path) -> list[str]:
    with gzip.open(path, "rt") as f:
        return f.readline().rstrip("\n").split("\t")


def load_all_variant_susie(ancestries: list[str]) -> pd.DataFrame:
    keep_cols = [
        "chrom", "pos", "a1", "a2", "molecular_trait_id", "variant_id",
        "pvalue", "beta", "se", "qvalue", "pip", "CS",
    ]
    dfs = []

    for ancestry in ancestries:
        files = list_susie_files(ancestry)
        print(f"[load] {ancestry}: {len(files):,} per-gene all-variant files")
        if not files:
            continue

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
        anc_df = anc_df.dropna(subset=["position", "pip", "gene_id", "ref_allele", "alt_allele"])
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


def add_tss_info(df: pd.DataFrame) -> pd.DataFrame:
    gene_loc = pd.read_csv(GENE_LOC_FILE, sep="\t")
    gene_tss = gene_loc[["ensgid", "TSS"]].rename(columns={"ensgid": "gene_id", "TSS": "tss"})
    out = df.merge(gene_tss, on="gene_id", how="left")
    out["distance_to_tss"] = (out["position"] - out["tss"]).abs()
    out["tss_distance_bin"] = out["distance_to_tss"].map(tss_distance_bin)
    return out


def load_old_prediction_keys() -> pd.DataFrame:
    if not OLD_COMBINED.exists():
        print(f"[warn] old combined prediction file not found: {OLD_COMBINED}")
        return pd.DataFrame(columns=_variant_key_cols() + ["tool"])

    cols = ["tool", "chr", "pos", "ref", "alt", "gene_id", "ancestry",
            "chromosome", "position", "ref_allele", "alt_allele"]
    old = pd.read_csv(OLD_COMBINED, sep="\t", usecols=lambda c: c in cols, low_memory=False)

    is_borzoi = old["chr"].notna() if "chr" in old.columns else pd.Series(False, index=old.index)
    old["chromosome_key"] = np.where(is_borzoi, old.get("chr"), old.get("chromosome"))
    old["position_key"] = np.where(is_borzoi, old.get("pos"), old.get("position"))
    old["ref_key"] = np.where(is_borzoi, old.get("ref"), old.get("ref_allele"))
    old["alt_key"] = np.where(is_borzoi, old.get("alt"), old.get("alt_allele"))

    key = old[["tool", "chromosome_key", "position_key", "ref_key", "alt_key", "gene_id", "ancestry"]].copy()
    key = key.rename(
        columns={
            "chromosome_key": "chromosome",
            "position_key": "position",
            "ref_key": "ref_allele",
            "alt_key": "alt_allele",
        }
    )
    key["chromosome"] = key["chromosome"].map(_normalize_chr)
    key["position"] = pd.to_numeric(key["position"], errors="coerce")
    key = key.dropna(subset=["position", "gene_id", "ancestry", "tool"])
    key["position"] = key["position"].astype("int64")
    key = key.drop_duplicates()
    print(f"[reuse] loaded {len(key):,} old scored tool/pair keys")
    return key


def select_pairs(
    df: pd.DataFrame,
    *,
    pip_positive_min: float,
    negative_pip_max: float,
    negative_ratio: float,
    max_negatives_per_ancestry: int | None,
    intermediate_pip_min: float,
    intermediate_pip_max: float,
    intermediate_ratio: float,
    max_intermediate_per_ancestry: int | None,
    seed: int,
) -> pd.DataFrame:
    positives = df[df["pip"] >= pip_positive_min].copy()
    neg_pool = df[df["pip"] < negative_pip_max].copy()
    intermediate_pool = df[(df["pip"] >= intermediate_pip_min) & (df["pip"] < intermediate_pip_max)].copy()
    print(f"[select] positives PIP >= {pip_positive_min}: {len(positives):,}")
    print(f"[select] negative pool PIP < {negative_pip_max}: {len(neg_pool):,}")
    print(
        "[select] intermediate pool "
        f"{intermediate_pip_min} <= PIP < {intermediate_pip_max}: {len(intermediate_pool):,}"
    )

    rng_seed = seed
    sampled_negatives = []
    sampled_intermediates = []
    for ancestry, anc_pos in positives.groupby("ancestry", sort=True):
        anc_neg = neg_pool[neg_pool["ancestry"] == ancestry]
        anc_intermediate = intermediate_pool[intermediate_pool["ancestry"] == ancestry]

        target_neg = int(np.ceil(len(anc_pos) * negative_ratio))
        if max_negatives_per_ancestry is not None:
            target_neg = min(target_neg, max_negatives_per_ancestry)
        target_neg = min(target_neg, len(anc_neg))

        if not anc_neg.empty and target_neg > 0:
            strata = ["tss_distance_bin"]
            if "CS" in anc_neg.columns:
                anc_neg = anc_neg.copy()
                anc_neg["cs_state"] = np.where(anc_neg["CS"].astype(int) != 0, "cs", "non_cs")
                strata.append("cs_state")

            frac_parts = []
            for _, group in anc_neg.groupby(strata, dropna=False, sort=True):
                n_group = len(group)
                n_take = max(1, int(round(target_neg * n_group / len(anc_neg))))
                n_take = min(n_take, n_group)
                frac_parts.append(group.sample(n=n_take, random_state=rng_seed))
                rng_seed += 1
            sampled = pd.concat(frac_parts, ignore_index=True)
            if len(sampled) > target_neg:
                sampled = sampled.sample(n=target_neg, random_state=seed + len(sampled))
            sampled_negatives.append(sampled)
            n_sampled_neg = len(sampled)
        else:
            n_sampled_neg = 0

        target_intermediate = int(np.ceil(len(anc_pos) * intermediate_ratio))
        if max_intermediate_per_ancestry is not None:
            target_intermediate = min(target_intermediate, max_intermediate_per_ancestry)
        target_intermediate = min(target_intermediate, len(anc_intermediate))

        if not anc_intermediate.empty and target_intermediate > 0:
            anc_intermediate = anc_intermediate.copy()
            anc_intermediate["pip_bin"] = pd.cut(
                anc_intermediate["pip"],
                bins=[intermediate_pip_min, 0.1, 0.2, 0.3, 0.4, intermediate_pip_max],
                labels=["0.01-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"],
                include_lowest=True,
                right=False,
            ).astype(str)
            strata = ["pip_bin", "tss_distance_bin"]
            if "CS" in anc_intermediate.columns:
                anc_intermediate["cs_state"] = np.where(
                    anc_intermediate["CS"].astype(int) != 0, "cs", "non_cs"
                )
                strata.append("cs_state")

            frac_parts = []
            for _, group in anc_intermediate.groupby(strata, dropna=False, sort=True):
                n_group = len(group)
                n_take = max(1, int(round(target_intermediate * n_group / len(anc_intermediate))))
                n_take = min(n_take, n_group)
                frac_parts.append(group.sample(n=n_take, random_state=rng_seed))
                rng_seed += 1
            sampled = pd.concat(frac_parts, ignore_index=True)
            if len(sampled) > target_intermediate:
                sampled = sampled.sample(n=target_intermediate, random_state=seed + 10_000 + len(sampled))
            sampled_intermediates.append(sampled)
            n_sampled_intermediate = len(sampled)
        else:
            n_sampled_intermediate = 0

        print(
            f"  {ancestry}: positives={len(anc_pos):,}, "
            f"sampled_low_negatives={n_sampled_neg:,} from {len(anc_neg):,}, "
            f"sampled_intermediate={n_sampled_intermediate:,} from {len(anc_intermediate):,}"
        )

    negatives = pd.concat(sampled_negatives, ignore_index=True) if sampled_negatives else pd.DataFrame(columns=df.columns)
    intermediates = pd.concat(sampled_intermediates, ignore_index=True) if sampled_intermediates else pd.DataFrame(columns=df.columns)
    selected = pd.concat([positives, negatives, intermediates], ignore_index=True)
    selected["selection_label"] = np.select(
        [
            selected["pip"] >= pip_positive_min,
            selected["pip"] < negative_pip_max,
            (selected["pip"] >= intermediate_pip_min) & (selected["pip"] < intermediate_pip_max),
        ],
        [
            "positive_pip_ge_0.5",
            "sampled_negative_pip_lt_0.01",
            "sampled_intermediate_pip_0.01_to_0.5",
        ],
        default="other",
    )
    selected = selected.drop_duplicates(subset=_variant_key_cols(), keep="first")
    selected = selected.sort_values(_variant_key_cols(), kind="mergesort").reset_index(drop=True)
    print(f"[select] selected {len(selected):,} unique variant-gene-ancestry rows")
    return selected


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
        if anc_df.empty:
            continue
        anc_df["chunk_index"] = (anc_df.index // cfg.pairs_per_chunk).astype(int)
        n_chunks = int(anc_df["chunk_index"].max()) + 1
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--ancestries", nargs="+", default=DEFAULT_ANCESTRIES)
    ap.add_argument("--pip_positive_min", type=float, default=0.5)
    ap.add_argument("--negative_pip_max", type=float, default=0.01)
    ap.add_argument("--negative_ratio", type=float, default=5.0)
    ap.add_argument("--max_negatives_per_ancestry", type=int, default=25000)
    ap.add_argument("--intermediate_pip_min", type=float, default=0.01)
    ap.add_argument("--intermediate_pip_max", type=float, default=0.5)
    ap.add_argument("--intermediate_ratio", type=float, default=5.0)
    ap.add_argument("--max_intermediate_per_ancestry", type=int, default=25000)
    ap.add_argument("--pairs_per_chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "analysis").mkdir(parents=True, exist_ok=True)

    all_variants = add_tss_info(load_all_variant_susie(args.ancestries))
    selected = select_pairs(
        all_variants,
        pip_positive_min=args.pip_positive_min,
        negative_pip_max=args.negative_pip_max,
        negative_ratio=args.negative_ratio,
        max_negatives_per_ancestry=args.max_negatives_per_ancestry,
        intermediate_pip_min=args.intermediate_pip_min,
        intermediate_pip_max=args.intermediate_pip_max,
        intermediate_ratio=args.intermediate_ratio,
        max_intermediate_per_ancestry=args.max_intermediate_per_ancestry,
        seed=args.seed,
    )
    selected.to_csv(args.results_dir / "selected_pairs.tsv.gz", sep="\t", index=False, compression="gzip")

    old_keys = load_old_prediction_keys()
    tool_key = selected[_variant_key_cols()].drop_duplicates()
    all_tool_pairs = pd.concat([tool_key.assign(tool=tool) for tool in TOOLS], ignore_index=True)
    reuse = all_tool_pairs.merge(old_keys, on=["tool"] + _variant_key_cols(), how="inner")
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
        ChunkConfig(results_dir=args.results_dir, pairs_per_chunk=args.pairs_per_chunk, seed=args.seed),
    )

    summary = selected.groupby(["ancestry", "selection_label"], observed=True).size().reset_index(name="n")
    summary.to_csv(args.results_dir / "analysis" / "selected_pair_summary.tsv", sep="\t", index=False)
    print("\n[summary]")
    print(summary.to_string(index=False))
    print(f"\n[reuse] selected pairs with any old score: {len(predicted_pairs):,}")
    print(f"[todo] pairs needing prediction: {len(missing_pairs):,}")
    print(f"[todo] tool jobs written: {len(jobs):,} -> {args.results_dir / 'manifests/jobs.tsv'}")
    for tool in TOOLS:
        n = int((jobs["tool"] == tool).sum()) if not jobs.empty else 0
        print(f"  {tool}: SLURM/local task ids 1-{n}" if n else f"  {tool}: no new jobs")


if __name__ == "__main__":
    main()
