#!/usr/bin/env python3
"""Prepare CH LD jobs (CH QTL cohort, 22 chromosomes) for the model-first CH pair manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PLINK_TEMPLATE = (
    "/mnt/vstor/Data14/metabrain_lasso/rna_seq_norm/MAGENTA_Hispanic/geno/imp_fil/"
    "geno_final/plink_by_chr/"
    "MAGENTA_HISP_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var."
    "no_outlier_MAC10.plink_qc.{chromosome}"
)
DEFAULT_SAMPLE_FILE = Path(
    "/mnt/vstor/Data14/metabrain_lasso/filer_maf_current_manuscript_20260831/"
    "keep/CH.exact_qtl_cohort.keep"
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_table(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    temporary.replace(path)


def select_variant_metadata(pairs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = {
        "variant_id", "chromosome", "position", "ref_allele", "alt_allele", "maf", "general"
    }
    missing = required - set(pairs.columns)
    if missing:
        raise RuntimeError(f"Pair manifest lacks columns: {sorted(missing)}")
    candidates = pairs[
        ["variant_id", "chromosome", "position", "ref_allele", "alt_allele", "maf", "general"]
    ].copy()
    candidates["maf"] = pd.to_numeric(candidates["maf"], errors="coerce")
    candidates["general"] = pd.to_numeric(candidates["general"], errors="coerce").fillna(0).astype(int)
    candidates["source_priority"] = candidates["general"]
    candidates = candidates.sort_values(
        ["variant_id", "source_priority"], ascending=[True, False], kind="mergesort"
    )
    spread = candidates.groupby("variant_id", observed=True)["maf"].agg(
        lambda values: float(values.max() - values.min())
    )
    discordant = int(spread.fillna(0).gt(1e-10).sum())
    variants = candidates.drop_duplicates("variant_id", keep="first").rename(
        columns={"maf": "eqtl_maf"}
    )
    variants = variants[
        ["variant_id", "chromosome", "position", "ref_allele", "alt_allele", "eqtl_maf"]
    ].sort_values(["chromosome", "position", "ref_allele", "alt_allele"], kind="mergesort")
    if not np.isfinite(variants["eqtl_maf"]).all():
        raise RuntimeError("Nonfinite CH association MAF in LD input")
    return variants, discordant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--plink-template", default=DEFAULT_PLINK_TEMPLATE)
    args = parser.parse_args()

    sample_count = sum(1 for line in args.sample_file.read_text().splitlines() if line.strip())
    if sample_count != 209:
        raise RuntimeError(f"Expected CH keep file with 209 samples, found {sample_count}")

    jobs = []
    chromosome_summary = []
    for chromosome in range(1, 23):
        pair_path = args.native_root / f"manifests/pairs/CH.chr{chromosome}.pairs.tsv.gz"
        pairs = pd.read_csv(pair_path, sep="\t", low_memory=False)
        if pairs["pair_key"].duplicated().any():
            raise RuntimeError(f"Duplicate CH pair keys on chromosome {chromosome}")
        variants, discordant_maf = select_variant_metadata(pairs)
        variant_path = args.native_root / f"manifests/variants/CH.chr{chromosome}.variants.tsv.gz"
        extract_path = args.native_root / f"work/extract_ids/CH.chr{chromosome}.ids.txt"
        atomic_table(variants, variant_path, compression="gzip")
        atomic_text(extract_path, "\n".join(variants["variant_id"].astype(str)) + "\n")

        bfile_prefix = Path(args.plink_template.format(chromosome=chromosome))
        for extension in ("bed", "bim", "fam"):
            if not Path(str(bfile_prefix) + f".{extension}").is_file():
                raise FileNotFoundError(f"{bfile_prefix}.{extension}")
        jobs.append(
            {
                "job_index": chromosome,
                "ancestry": "CH",
                "chromosome": chromosome,
                "bfile_prefix": str(bfile_prefix),
                "sample_file": str(args.sample_file),
                "pair_manifest": str(pair_path),
                "variant_manifest": str(variant_path),
                "extract_ids_file": str(extract_path),
                "expected_variants": len(variants),
                "output_prefix": str(args.native_root / f"work/ld/CH.chr{chromosome}"),
                "qc_output": str(args.native_root / f"qc/ld_chromosomes/CH.chr{chromosome}.json"),
            }
        )
        chromosome_summary.append(
            {
                "chromosome": chromosome,
                "pair_rows": len(pairs),
                "unique_variants": len(variants),
                "variants_with_cross_panel_maf_disagreement": discordant_maf,
            }
        )

    jobs_path = args.native_root / "manifests/ch_ld_jobs.tsv"
    summary_path = args.native_root / "qc/ch_ld_input_summary.json"
    chromosome_path = args.native_root / "qc/ch_ld_input_by_chromosome.tsv"
    atomic_table(pd.DataFrame(jobs), jobs_path)
    atomic_table(pd.DataFrame(chromosome_summary), chromosome_path)
    summary = {
        "ancestry": "CH",
        "sample_count": sample_count,
        "chromosome_jobs": len(jobs),
        "pair_rows": int(sum(row["pair_rows"] for row in chromosome_summary)),
        "unique_variants_summed_across_chromosomes": int(
            sum(row["unique_variants"] for row in chromosome_summary)
        ),
        "variants_with_cross_panel_maf_disagreement": int(
            sum(row["variants_with_cross_panel_maf_disagreement"] for row in chromosome_summary)
        ),
        "job_manifest": str(jobs_path),
        "sample_file": str(args.sample_file),
        "ready": len(jobs) == 22 and sample_count == 209,
    }
    atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
