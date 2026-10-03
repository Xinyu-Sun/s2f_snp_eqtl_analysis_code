#!/usr/bin/env python3
"""Prepare outcome-blind native-pair, sample, and chromosome LD manifests."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LD_ROOT.parent
ANCESTRIES = ("AA", "CH", "NHW")
PLINK_TEMPLATES = {
    "AA": "/mnt/vstor/Data14/metabrain_lasso/rna_seq_norm/aa_233/geno/imp_fil/geno_final/plink_by_chr/MAGENTA_AA_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.no_outlier_MAC10.plink_qc.{chromosome}",
    "CH": "/mnt/vstor/Data14/metabrain_lasso/rna_seq_norm/MAGENTA_Hispanic/geno/imp_fil/geno_final/plink_by_chr/MAGENTA_HISP_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.no_outlier_MAC10.plink_qc.{chromosome}",
    "NHW": "/mnt/vstor/Data14/metabrain_lasso/rna_seq_norm/nhw_239/geno/imp_fil/geno_final/plink_by_chr/MAGENTA_NHW_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.no_outlier_MAC10.plink_qc.{chromosome}",
}
SAMPLE_LISTS = {
    "AA": Path("/mnt/vstor/Data14/xxs410_xinyu/SuShiE_work/magenta_geno_QC/AA_sample_int_w_exp_tpm.txt"),
    "CH": Path("/mnt/vstor/Data14/xxs410_xinyu/SuShiE_work/magenta_geno_QC/HISP_IHG_sample_int_w_exp_tpm.txt"),
    "NHW": Path("/mnt/vstor/Data14/xxs410_xinyu/SuShiE_work/magenta_geno_QC/NHW_sample_int_w_exp_tpm.txt"),
}
PER_LIST = Path("/mnt/vstor/Data14/xxs410_xinyu/SuShiE_work/magenta_geno_QC/HISP_PER_sample_int_w_exp_tpm.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    if mode is not None:
        os.chmod(temporary, mode)
    temporary.replace(path)


def atomic_dataframe_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
    temporary.replace(path)


def normalized_sample_core(value: str) -> str:
    return re.sub(r"^[A-Za-z_]+", "", value)


def read_keep_iids(path: Path) -> list[str]:
    result = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields:
            result.append(fields[1] if len(fields) > 1 else fields[0])
    if len(result) != len(set(result)):
        raise RuntimeError(f"Duplicate sample IDs in {path}")
    return result


def read_fam(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["fid", "iid", "father", "mother", "sex", "phenotype"],
        dtype=str,
    )
    if frame["iid"].duplicated().any():
        raise RuntimeError(f"Duplicate PLINK IID in {path}")
    return frame


def map_samples(ancestry: str, fam: Path, output: Path) -> dict[str, object]:
    requested = read_keep_iids(SAMPLE_LISTS[ancestry])
    fam_frame = read_fam(fam)
    plink_samples = fam_frame["iid"].astype(str).to_list()
    requested_by_core = {normalized_sample_core(value): value for value in requested}
    plink_by_core = {normalized_sample_core(value): value for value in plink_samples}
    if len(requested_by_core) != len(requested) or len(plink_by_core) != len(plink_samples):
        raise RuntimeError(f"Nonunique normalized sample core for {ancestry}")
    matched_cores = sorted(set(requested_by_core) & set(plink_by_core))
    matched_iids = [plink_by_core[core] for core in matched_cores]
    fam_by_iid = fam_frame.set_index("iid", drop=False)
    per_overlap = 0
    if ancestry == "CH":
        per_cores = {normalized_sample_core(value) for value in read_keep_iids(PER_LIST)}
        per_overlap = len(set(matched_cores) & per_cores)
        if per_overlap:
            raise RuntimeError("PER membership leaked into the CH-IHG sample set")
    keep_text = "".join(
        f"{fam_by_iid.loc[iid, 'fid']}\t{iid}\n" for iid in matched_iids
    )
    atomic_text(output, keep_text, mode=0o600)
    return {
        "ancestry": ancestry,
        "requested_expression_samples": len(requested),
        "plink_samples": len(plink_samples),
        "matched_samples": len(matched_iids),
        "unmatched_expression_samples": len(requested) - len(matched_iids),
        "normalized_core_unique": True,
        "per_overlap_in_ch": per_overlap,
        "sample_file": str(output),
        "sample_file_sha256": sha256_file(output),
        "sample_identifiers_reported": False,
    }


def panel_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "general": frame["general"].eq(1),
        "ad_tier1_excluding_apoe": frame["ad_tier1"].eq(1) & frame["apoe_locus"].eq(0),
        "ad_tier1_apoe_only": frame["ad_tier1"].eq(1) & frame["apoe_locus"].eq(1),
        "ad_tier2": frame["ad_tier2"].eq(1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=PROJECT_ROOT / "results/analysis_pairs.tsv.gz")
    parser.add_argument("--output-root", type=Path, default=LD_ROOT)
    args = parser.parse_args()

    columns = [
        "pair_key", "chromosome", "position", "ref_allele", "alt_allele", "gene_id",
        "tss", "abs_distance_to_tss", "general", "ad_tier1", "ad_tier2", "apoe_locus",
        "AA_present", "CH_present", "NHW_present",
        "AA_maf", "CH_maf", "NHW_maf", "AA_pvalue", "CH_pvalue", "NHW_pvalue",
        "AA_z", "CH_z", "NHW_z", "AA_qvalue", "CH_qvalue", "NHW_qvalue",
        "borzoi_score", "alphagenome_score", "general_sampling_stratum",
        "general_gene_inclusion_probability", "general_inverse_probability_weight",
    ]
    frame = pd.read_csv(args.pairs, sep="\t", usecols=columns, low_memory=False)
    if frame["pair_key"].duplicated().any():
        raise RuntimeError("Duplicate exact pair keys in analysis_pairs")
    finite_scores = np.isfinite(frame["borzoi_score"]) & np.isfinite(frame["alphagenome_score"])
    valid_snv = (
        frame["ref_allele"].astype(str).str.fullmatch("[ACGT]")
        & frame["alt_allele"].astype(str).str.fullmatch("[ACGT]")
        & frame["ref_allele"].ne(frame["alt_allele"])
    )
    valid_distance = frame["abs_distance_to_tss"].le(100000)
    eligible = finite_scores & valid_snv & valid_distance
    excluded = {
        "nonfinite_common_model_score": int((~finite_scores).sum()),
        "not_biallelic_acgt_snv": int((~valid_snv).sum()),
        "outside_100kb": int((~valid_distance).sum()),
        "excluded_union": int((~eligible).sum()),
    }
    frame = frame.loc[eligible].copy()
    frame["chromosome"] = frame["chromosome"].astype(int)
    frame["position"] = frame["position"].astype(int)
    frame["variant_id"] = (
        "chr" + frame["chromosome"].astype(str) + ":" + frame["position"].astype(str)
        + "_" + frame["ref_allele"].astype(str) + "_" + frame["alt_allele"].astype(str)
    )

    output_root = args.output_root
    (output_root / "manifests/variants").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests/pairs").mkdir(parents=True, exist_ok=True)
    (output_root / "work/extract_ids").mkdir(parents=True, exist_ok=True)
    (output_root / "work/sample_keep").mkdir(parents=True, exist_ok=True)
    (output_root / "qc").mkdir(parents=True, exist_ok=True)

    sample_qc = []
    census_rows = []
    chromosome_jobs = []
    masks = panel_masks(frame)

    for ancestry in ANCESTRIES:
        fam22 = Path(PLINK_TEMPLATES[ancestry].format(chromosome=22) + ".fam")
        sample_file = output_root / f"work/sample_keep/{ancestry}.samples.txt"
        sample_qc.append(map_samples(ancestry, fam22, sample_file))
        native = frame.loc[frame[f"{ancestry}_present"].eq(1)].copy()
        for panel, mask in masks.items():
            panel_native = native.loc[mask.loc[native.index]]
            census_rows.append({
                "ancestry": ancestry,
                "panel": panel,
                "native_pairs": int(len(panel_native)),
                "genes": int(panel_native["gene_id"].nunique()),
                "unique_variants": int(panel_native["variant_id"].nunique()),
                "three_way_shared_pairs": int(
                    panel_native[["AA_present", "CH_present", "NHW_present"]].eq(1).all(axis=1).sum()
                ),
            })

        variant_info = native[[
            "variant_id", "chromosome", "position", "ref_allele", "alt_allele", f"{ancestry}_maf"
        ]].copy()
        variant_info = variant_info.rename(columns={f"{ancestry}_maf": "eqtl_maf"})
        maf_spread = variant_info.groupby("variant_id", sort=False)["eqtl_maf"].agg(lambda x: x.max() - x.min())
        if (maf_spread.fillna(0) > 1e-10).any():
            raise RuntimeError(f"Inconsistent ancestry MAF across genes for {ancestry}")
        variant_info = variant_info.drop_duplicates("variant_id").sort_values(
            ["chromosome", "position", "ref_allele", "alt_allele"]
        )
        for chromosome in range(1, 23):
            chrom = variant_info.loc[variant_info["chromosome"].eq(chromosome)].copy()
            if chrom.empty:
                raise RuntimeError(f"No target variants for {ancestry} chromosome {chromosome}")
            chrom_pairs = native.loc[native["chromosome"].eq(chromosome), [
                "pair_key", "variant_id", "chromosome", "position", "ref_allele", "alt_allele",
                "gene_id", "tss", "abs_distance_to_tss", "general", "ad_tier1", "ad_tier2",
                "apoe_locus", "AA_present", "CH_present", "NHW_present", f"{ancestry}_maf",
                f"{ancestry}_pvalue", f"{ancestry}_z", f"{ancestry}_qvalue",
                "borzoi_score", "alphagenome_score", "general_sampling_stratum",
                "general_gene_inclusion_probability", "general_inverse_probability_weight",
            ]].copy()
            chrom_pairs = chrom_pairs.rename(columns={
                f"{ancestry}_maf": "maf", f"{ancestry}_pvalue": "pvalue",
                f"{ancestry}_z": "z", f"{ancestry}_qvalue": "qvalue",
            })
            chrom_pairs["three_way_shared"] = chrom_pairs[
                ["AA_present", "CH_present", "NHW_present"]
            ].eq(1).all(axis=1).astype(int)
            pair_path = output_root / f"manifests/pairs/{ancestry}.chr{chromosome}.pairs.tsv.gz"
            variant_path = output_root / f"manifests/variants/{ancestry}.chr{chromosome}.variants.tsv.gz"
            extract_path = output_root / f"work/extract_ids/{ancestry}.chr{chromosome}.ids.txt"
            atomic_dataframe_gzip(chrom_pairs.sort_values(["gene_id", "position", "pair_key"]), pair_path)
            atomic_dataframe_gzip(chrom, variant_path)
            atomic_text(extract_path, "\n".join(chrom["variant_id"].astype(str)) + "\n")
            bfile_prefix = Path(PLINK_TEMPLATES[ancestry].format(chromosome=chromosome))
            for extension in ("bed", "bim", "fam"):
                if not Path(str(bfile_prefix) + f".{extension}").is_file():
                    raise RuntimeError(f"Missing PLINK input: {bfile_prefix}.{extension}")
            if sha256_file(Path(str(bfile_prefix) + ".fam")) != sha256_file(fam22):
                raise RuntimeError(f"PLINK FAM differs by chromosome for {ancestry} chr{chromosome}")
            chromosome_jobs.append({
                "job_index": len(chromosome_jobs),
                "ancestry": ancestry,
                "chromosome": chromosome,
                "bfile_prefix": str(bfile_prefix),
                "sample_file": str(sample_file),
                "pair_manifest": str(pair_path),
                "variant_manifest": str(variant_path),
                "extract_ids_file": str(extract_path),
                "expected_variants": int(len(chrom)),
                "output_prefix": str(output_root / f"work/ld/{ancestry}.chr{chromosome}"),
                "qc_output": str(output_root / f"qc/ld_chromosomes/{ancestry}.chr{chromosome}.json"),
            })

    census = pd.DataFrame(census_rows)
    jobs = pd.DataFrame(chromosome_jobs)
    census_path = output_root / "qc/native_ld_input_census.tsv"
    jobs_path = output_root / "manifests/ld_chromosome_jobs.tsv"
    census.to_csv(census_path, sep="\t", index=False)
    jobs.to_csv(jobs_path, sep="\t", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pairs": str(args.pairs),
        "source_pairs_sha256": sha256_file(args.pairs),
        "source_pair_rows": int(len(frame)),
        "exclusions_before_native_split": excluded,
        "sample_qc": sample_qc,
        "ch_definition": "IHG_only",
        "per_excluded": True,
        "chromosome_jobs": int(len(jobs)),
        "expected_jobs": 66,
        "census": str(census_path),
        "census_sha256": sha256_file(census_path),
        "job_manifest": str(jobs_path),
        "job_manifest_sha256": sha256_file(jobs_path),
        "sample_ids_emitted_to_summary": False,
        "ready": len(jobs) == 66 and all(row["matched_samples"] >= 200 for row in sample_qc),
    }
    summary_path = output_root / "qc/native_ld_input_summary.json"
    atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
