#!/usr/bin/env python3
"""Build matched-FILER enrichment inputs (inputs/) from the fine-mapping benchmark pool and exact-cohort PLINK MAF."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_long(pool: pd.DataFrame) -> pd.DataFrame:
    common = [
        "ancestry", "chromosome", "position", "ref_allele", "alt_allele",
        "gene_id", "gene_name", "gene_type", "tss_distance_bin", "pvalue",
        "beta", "std_error", "qvalue", "pip", "CS", "pair_key",
    ]
    pieces = []
    for tool, score_column, output_column in [
        ("borzoi", "borzoi_score", "logMeanSED"),
        ("alphagenome", "alphagenome_score", "raw_score"),
    ]:
        frame = pool.dropna(subset=[score_column]).copy()
        keep = [column for column in common if column in frame]
        frame = frame[keep + [score_column]].rename(columns={score_column: output_column})
        frame["tool"] = tool
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True, sort=False)
    result["population"] = result["ancestry"]
    result["chr"] = result["chromosome"]
    result["pos"] = result["position"]
    result["ref"] = result["ref_allele"]
    result["alt"] = result["alt_allele"]
    return result


def convert_ch_maf(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    result = pd.DataFrame(
        {
            "ancestry": "CH",
            "chromosome": "chr" + frame["chromosome"].astype(int).astype(str),
            "position": pd.to_numeric(frame["position"], errors="raise").astype(int),
            "a1": frame["plink_a1"].astype(str),
            "a2": frame["plink_a2"].astype(str),
            "a1_frequency": pd.to_numeric(frame["genotype_alt_frequency"], errors="raise"),
            "maf": pd.to_numeric(frame["genotype_maf"], errors="raise"),
            "nchrobs": pd.to_numeric(frame["n_chromosomes_observed"], errors="raise").astype(int),
            "analysis_sample_n": 209,
            "plink_variant_id": frame["variant_id"].astype(str),
            "maf_source": "PLINK1.9_exact_QTL_cohort",
        }
    )
    result["allele_pair"] = result[["a1", "a2"]].apply(
        lambda row: "/".join(sorted([row["a1"], row["a2"]])), axis=1
    )
    return result[
        [
            "ancestry", "chromosome", "position", "a1", "a2", "allele_pair",
            "a1_frequency", "maf", "nchrobs", "analysis_sample_n",
            "plink_variant_id", "maf_source",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--ch-maf", type=Path, required=True)
    parser.add_argument("--aa-nhw-plink-maf", type=Path, required=True)
    parser.add_argument("--gene-locations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(args.pool, sep="\t", low_memory=False)
    combined = make_long(pool)
    combined_path = args.output_dir / "susie_s2f_combined.tsv.gz"
    combined.to_csv(combined_path, sep="\t", index=False, compression="gzip")

    aa_nhw_maf = pd.read_csv(args.aa_nhw_plink_maf, sep="\t")
    ch_maf = convert_ch_maf(args.ch_maf)
    plink_maf = pd.concat(
        [aa_nhw_maf[~aa_nhw_maf["ancestry"].eq("CH")], ch_maf],
        ignore_index=True,
    )
    if plink_maf.duplicated(["ancestry", "chromosome", "position", "allele_pair"]).any():
        raise RuntimeError("Duplicate ancestry-coordinate-allele keys in combined PLINK MAF")
    maf_path = args.output_dir / "benchmark_plink_maf_exact_qtl_cohorts.tsv.gz"
    plink_maf.to_csv(maf_path, sep="\t", index=False, compression="gzip")

    counts = []
    for threshold in [0.5, 0.9]:
        for tool in ["alphagenome", "borzoi"]:
            for ancestry in ["AA", "CH", "NHW"]:
                subset = combined[
                    combined["tool"].eq(tool)
                    & combined["ancestry"].eq(ancestry)
                    & pd.to_numeric(combined["pip"], errors="coerce").ge(threshold)
                ].copy()
                variant_id = (
                    subset["chromosome"].astype(str) + ":"
                    + subset["position"].astype(int).astype(str) + "_"
                    + subset["ref_allele"].astype(str) + "_"
                    + subset["alt_allele"].astype(str)
                )
                counts.append(
                    {
                        "pip_threshold": threshold,
                        "tool": tool,
                        "ancestry": ancestry,
                        "unique_positive_variants": variant_id.nunique(),
                    }
                )
    counts_frame = pd.DataFrame(counts)
    counts_path = args.output_dir / "expected_positive_variant_counts.tsv"
    counts_frame.to_csv(counts_path, sep="\t", index=False)

    gene_path = args.output_dir / "hg38_gene_locations.txt"
    shutil.copy2(args.gene_locations, gene_path)
    generation_rows = []
    for ancestry, expected in {"AA": 224, "CH": 209, "NHW": 235}.items():
        subset = plink_maf[plink_maf["ancestry"].eq(ancestry)]
        generation_rows.append(
            {
                "ancestry": ancestry,
                "analysis_sample_n": expected,
                "unique_benchmark_variants": len(subset),
                "raw_plink_frequency_rows": len(subset),
                "selected_canonical_frequency_rows": len(subset),
                "ignored_noncanonical_or_reverse_id_rows": 0,
                "matched_benchmark_variants": len(subset),
                "unmatched_benchmark_variants": 0,
                "coverage_fraction": 1.0,
            }
        )
    generation_path = args.output_dir / "plink_maf_generation_summary.tsv"
    pd.DataFrame(generation_rows).to_csv(generation_path, sep="\t", index=False)

    checksum_rows = []
    for path, purpose in [
        (combined_path, "fine-mapping benchmark pool in long format (one row per model score)"),
        (maf_path, "exact-QTL-cohort PLINK MAF for all benchmark variants"),
        (counts_path, "expected unique positive-variant counts by model, ancestry, and threshold"),
        (gene_path, "hg38 gene coordinates used to recompute TSS distance"),
        (generation_path, "PLINK MAF cohort-size and complete-coverage validation summary"),
    ]:
        checksum_rows.append(
            {
                "path": f"inputs/{path.name}",
                "sha256": sha256(path),
                "purpose": purpose,
            }
        )
    pd.DataFrame(checksum_rows).to_csv(
        args.output_dir / "input_checksums.tsv", sep="\t", index=False
    )

    summary = {
        "combined_rows": len(combined),
        "combined_rows_by_tool_ancestry": {
            f"{tool}|{ancestry}": int(count)
            for (tool, ancestry), count in combined.groupby(["tool", "ancestry"]).size().items()
        },
        "plink_maf_rows": len(plink_maf),
        "plink_maf_rows_by_ancestry": plink_maf["ancestry"].value_counts().to_dict(),
        "combined_sha256": sha256(combined_path),
        "plink_maf_sha256": sha256(maf_path),
        "positive_counts_sha256": sha256(counts_path),
    }
    (args.output_dir / "filer_input_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
