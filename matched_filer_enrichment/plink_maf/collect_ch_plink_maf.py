#!/usr/bin/env python3
"""Collect and validate CH exact-QTL-cohort PLINK allele frequencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(
        args.work_dir / "ch_selected_variant_metadata.tsv.gz", sep="\t"
    )
    path = args.work_dir / "plink/CH.exact_qtl_cohort.frq"
    frequency = pd.read_csv(path, sep=r"\s+")
    frequency = frequency.rename(
        columns={
            "CHR": "chromosome", "SNP": "variant_id", "A1": "plink_a1",
            "A2": "plink_a2", "MAF": "genotype_alt_frequency",
            "NCHROBS": "n_chromosomes_observed",
        }
    )
    frequency["chromosome"] = pd.to_numeric(frequency["chromosome"], errors="raise").astype(int)
    if frequency["variant_id"].duplicated().any():
        raise RuntimeError("Duplicate PLINK frequency variant IDs")
    result = metadata.merge(frequency, on=["variant_id", "chromosome"], how="left", validate="one_to_one")
    result["allele_order_valid"] = (
        result["plink_a1"].astype(str).eq(result["alt_allele"].astype(str))
        & result["plink_a2"].astype(str).eq(result["ref_allele"].astype(str))
    )
    result["genotype_maf"] = np.minimum(
        pd.to_numeric(result["genotype_alt_frequency"], errors="coerce"),
        1 - pd.to_numeric(result["genotype_alt_frequency"], errors="coerce"),
    )
    result["maf_absolute_difference"] = (
        result["association_maf"] - result["genotype_maf"]
    ).abs()
    result["analysis_sample_n"] = 209
    result["maf_source"] = "PLINK1.9_exact_QTL_cohort"
    valid = result.dropna(subset=["association_maf", "genotype_maf"])
    summary = {
        "requested_variants": len(metadata),
        "observed_variants": int(result["genotype_maf"].notna().sum()),
        "missing_variants": int(result["genotype_maf"].isna().sum()),
        "allele_order_failures": int((~result["allele_order_valid"]).sum()),
        "maf_correlation": float(valid[["association_maf", "genotype_maf"]].corr().iloc[0, 1]),
        "maf_median_absolute_difference": float(valid["maf_absolute_difference"].median()),
        "maf_q95_absolute_difference": float(valid["maf_absolute_difference"].quantile(0.95)),
    }
    summary["ready"] = bool(
        summary["missing_variants"] == 0
        and summary["allele_order_failures"] == 0
        and summary["maf_correlation"] >= 0.9
        and summary["maf_median_absolute_difference"] <= 0.05
    )
    result.to_csv(
        args.work_dir / "ch_plink_maf.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    (args.work_dir / "plink_maf_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    if not summary["ready"]:
        raise RuntimeError("CH PLINK MAF validation failed")


if __name__ == "__main__":
    main()
