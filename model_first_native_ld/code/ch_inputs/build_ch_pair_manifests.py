#!/usr/bin/env python3
"""Build the CH pair manifests of the locked 500-gene model-first benchmark.

The gene panel, model scores, clumping rules, and AA/NHW inputs come from the locked design.
CH regular-eQTL association statistics and CH pair availability come from the scored CH pairs
of the CH eQTL release. CH LD edges in the design directory are used only if exact variant
coverage and allele-frequency QC pass; otherwise CH LD is computed in the CH QTL cohort
(prepare_ch_ld_inputs.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def role_mask(frame: pd.DataFrame, role: str) -> pd.Series:
    return frame["analysis_role"].fillna("").str.split(",").map(lambda values: role in values)


def chromosome_number(value: object) -> int:
    text = str(value).removeprefix("chr")
    return int(float(text))


def load_locked_metadata(path: Path, keys: set[str]) -> pd.DataFrame:
    columns = [
        "pair_key", "general", "ad_tier1", "ad_tier2", "apoe_locus",
        "AA_present", "NHW_present", "general_sampling_stratum",
        "general_gene_inclusion_probability", "general_inverse_probability_weight",
    ]
    pieces = []
    for chunk in pd.read_csv(path, sep="\t", usecols=columns, chunksize=250_000):
        selected = chunk[chunk["pair_key"].isin(keys)]
        if not selected.empty:
            pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def gene_design_metadata(path: Path, genes: set[str]) -> pd.DataFrame:
    columns = [
        "gene_id", "general_sampling_stratum", "general_gene_inclusion_probability",
        "general_inverse_probability_weight",
    ]
    pieces = []
    for chunk in pd.read_csv(path, sep="\t", usecols=columns, chunksize=250_000):
        selected = chunk[chunk["gene_id"].isin(genes)]
        if not selected.empty:
            pieces.append(selected)
    combined = pd.concat(pieces, ignore_index=True).drop_duplicates("gene_id")
    if set(combined["gene_id"]) != genes:
        missing = sorted(genes - set(combined["gene_id"]))
        raise RuntimeError(f"Missing locked design metadata for {len(missing)} genes")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-scored", type=Path, required=True)
    parser.add_argument("--locked-analysis-pairs", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument(
        "--allow-ld-recompute",
        action="store_true",
        help=(
            "Write CH pair manifests and the locked-LD coverage audit "
            "without failing when CH LD must be computed in the CH QTL cohort."
        ),
    )
    args = parser.parse_args()

    master = pd.read_csv(args.master_scored, sep="\t", low_memory=False)
    ch_pairs = master[role_mask(master, "model_first_fixed_panel")].copy()
    ch_pairs["position"] = pd.to_numeric(ch_pairs["position"], errors="coerce")
    ch_pairs["distance_to_tss"] = pd.to_numeric(ch_pairs["distance_to_tss"], errors="coerce")
    ch_pairs["borzoi_score"] = pd.to_numeric(ch_pairs["borzoi_score"], errors="coerce")
    ch_pairs["alphagenome_score"] = pd.to_numeric(ch_pairs["alphagenome_score"], errors="coerce")
    valid_allele = ch_pairs["ref_allele"].astype(str).str.fullmatch("[ACGT]") & ch_pairs[
        "alt_allele"
    ].astype(str).str.fullmatch("[ACGT]")
    ch_pairs = ch_pairs[
        valid_allele
        & ch_pairs["distance_to_tss"].le(100_000)
        & np.isfinite(ch_pairs["borzoi_score"])
        & np.isfinite(ch_pairs["alphagenome_score"])
    ].copy()
    if ch_pairs["pair_key"].duplicated().any():
        raise RuntimeError("CH panel pairs are not unique")

    keys = set(ch_pairs["pair_key"].astype(str))
    genes = set(ch_pairs["gene_id"].astype(str))
    locked = load_locked_metadata(args.locked_analysis_pairs, keys)
    design = gene_design_metadata(args.locked_analysis_pairs, genes)
    ch_pairs = ch_pairs.merge(locked, on="pair_key", how="left", validate="one_to_one")
    ch_pairs = ch_pairs.merge(
        design,
        on="gene_id",
        how="left",
        suffixes=("", "_gene"),
        validate="many_to_one",
    )
    for column in [
        "general_sampling_stratum", "general_gene_inclusion_probability",
        "general_inverse_probability_weight",
    ]:
        ch_pairs[column] = ch_pairs[column].where(
            ch_pairs[column].notna(), ch_pairs[f"{column}_gene"]
        )
    ch_pairs["general"] = 1
    ch_pairs["ad_tier1"] = ch_pairs["ad_tier1"].fillna(0).astype(int)
    ch_pairs["ad_tier2"] = ch_pairs["ad_tier2"].fillna(0).astype(int)
    ch_pairs["apoe_locus"] = ch_pairs["apoe_locus"].fillna(0).astype(int)
    ch_pairs["AA_present"] = ch_pairs["AA_present"].fillna(0).astype(int)
    ch_pairs["NHW_present"] = ch_pairs["NHW_present"].fillna(0).astype(int)
    ch_pairs["CH_present"] = 1
    ch_pairs["three_way_shared"] = (
        ch_pairs["AA_present"].eq(1) & ch_pairs["NHW_present"].eq(1)
    ).astype(int)
    ch_pairs["chromosome"] = ch_pairs["chromosome"].map(chromosome_number)
    ch_pairs["position"] = ch_pairs["position"].astype(int)
    ch_pairs["tss"] = pd.to_numeric(ch_pairs["tss"], errors="raise").astype(int)
    ch_pairs["abs_distance_to_tss"] = ch_pairs["distance_to_tss"].abs().astype(int)
    ch_pairs["maf"] = pd.to_numeric(ch_pairs["maf"], errors="coerce")
    ch_pairs["pvalue"] = pd.to_numeric(ch_pairs["pvalue"], errors="coerce")
    ch_pairs["qvalue"] = pd.to_numeric(ch_pairs["qvalue"], errors="coerce")
    ch_pairs["z"] = pd.to_numeric(ch_pairs["beta"], errors="coerce") / pd.to_numeric(
        ch_pairs["std_error"], errors="coerce"
    )
    ch_pairs["variant_id"] = (
        "chr" + ch_pairs["chromosome"].astype(str) + ":" + ch_pairs["position"].astype(str)
        + "_" + ch_pairs["ref_allele"].astype(str) + "_" + ch_pairs["alt_allele"].astype(str)
    )

    required_finite = [
        "maf", "pvalue", "z", "qvalue", "borzoi_score", "alphagenome_score",
        "general_gene_inclusion_probability", "general_inverse_probability_weight",
    ]
    if not np.isfinite(ch_pairs[required_finite].astype(float)).all().all():
        failures = ch_pairs[~np.isfinite(ch_pairs[required_finite].astype(float)).all(axis=1)]
        raise RuntimeError(f"Nonfinite CH model-first fields in {len(failures)} pairs")

    manifest_columns = [
        "pair_key", "variant_id", "chromosome", "position", "ref_allele", "alt_allele",
        "gene_id", "tss", "abs_distance_to_tss", "general", "ad_tier1", "ad_tier2",
        "apoe_locus", "AA_present", "CH_present", "NHW_present", "maf", "pvalue", "z",
        "qvalue", "borzoi_score", "alphagenome_score", "general_sampling_stratum",
        "general_gene_inclusion_probability", "general_inverse_probability_weight",
        "three_way_shared",
    ]
    pair_dir = args.native_root / "manifests/pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    qc_rows = []
    for chromosome in range(1, 23):
        frame = ch_pairs[ch_pairs["chromosome"].eq(chromosome)].copy()
        existing_manifest_path = pair_dir / f"CH.chr{chromosome}.pairs.tsv.gz"
        existing_manifest = pd.read_csv(existing_manifest_path, sep="\t", low_memory=False)
        preserved_non_general = existing_manifest[~existing_manifest["general"].eq(1)].copy()
        coverage_path = args.native_root / f"qc/ld_chromosomes/CH.chr{chromosome}.variant_coverage.tsv.gz"
        coverage = pd.read_csv(coverage_path, sep="\t")
        coverage_ids = set(coverage["variant_id"].astype(str))
        missing_ids = sorted(set(frame["variant_id"].astype(str)) - coverage_ids)
        genotype_maf = coverage.set_index("variant_id")["genotype_maf"]
        frame["genotype_maf"] = frame["variant_id"].map(genotype_maf)
        diff = (frame["maf"] - frame["genotype_maf"]).abs()
        qc_rows.append(
            {
                "chromosome": chromosome,
                "ch_pairs": len(frame),
                "ch_genes": frame["gene_id"].nunique(),
                "ch_unique_variants": frame["variant_id"].nunique(),
                "preserved_non_general_pairs": len(preserved_non_general),
                "pairs_present_in_locked_design": int(frame["AA_present"].notna().sum()),
                "variants_missing_from_locked_ld_coverage": len(missing_ids),
                "maf_valid_variants": int(diff.notna().sum()),
                "maf_correlation": float(frame[["maf", "genotype_maf"]].corr().iloc[0, 1]),
                "maf_median_absolute_difference": float(diff.median()),
                "maf_q95_absolute_difference": float(diff.quantile(0.95)),
            }
        )
        if missing_ids:
            pd.Series(missing_ids).to_csv(
                args.native_root / f"qc/CH.chr{chromosome}.missing_locked_ld_variants.txt",
                index=False,
                header=False,
            )
        combined_manifest = pd.concat(
            [frame[manifest_columns], preserved_non_general[manifest_columns]],
            ignore_index=True,
        ).drop_duplicates("pair_key", keep="first")
        combined_manifest.sort_values(
            ["gene_id", "position", "ref_allele", "alt_allele"], kind="mergesort"
        ).to_csv(
            pair_dir / f"CH.chr{chromosome}.pairs.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

    qc = pd.DataFrame(qc_rows)
    qc_path = args.native_root / "qc/ch_manifest_and_locked_ld_reuse.tsv"
    qc.to_csv(qc_path, sep="\t", index=False)
    audit = {
        "ch_pairs": len(ch_pairs),
        "ch_genes": ch_pairs["gene_id"].nunique(),
        "pairs_found_in_locked_design": len(locked),
        "pairs_new_to_locked_design": len(ch_pairs) - len(locked),
        "variants_missing_from_locked_ld_coverage": int(
            qc["variants_missing_from_locked_ld_coverage"].sum()
        ),
        "minimum_maf_correlation": float(qc["maf_correlation"].min()),
        "maximum_maf_median_absolute_difference": float(
            qc["maf_median_absolute_difference"].max()
        ),
        "locked_gene_panel": True,
        "locked_model_scores": True,
        "locked_ld_reuse_permitted": bool(
            qc["variants_missing_from_locked_ld_coverage"].sum() == 0
            and qc["maf_correlation"].min() >= 0.9
            and qc["maf_median_absolute_difference"].max() <= 0.05
        ),
    }
    (args.native_root / "qc/ch_manifest_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, sort_keys=True))
    if not audit["locked_ld_reuse_permitted"] and not args.allow_ld_recompute:
        raise RuntimeError("CH manifest did not pass locked-LD reuse gates")


if __name__ == "__main__":
    main()
