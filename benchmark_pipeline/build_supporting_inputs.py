#!/usr/bin/env python3
"""Build compact supporting inputs from the benchmark pool: deduplicated FILER positive variants with
exact-cohort PLINK MAF, positive counts by group, and selected-pair summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_KEYS = ["ancestry", "chromosome", "position", "ref_allele", "alt_allele", "gene_id"]


def canonical_variant(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["chromosome"].astype(str)
        + ":"
        + frame["position"].astype(int).astype(str)
        + "_"
        + frame["ref_allele"].astype(str)
        + "_"
        + frame["alt_allele"].astype(str)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--maf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    maf = pd.read_csv(args.maf, sep="\t", low_memory=False)
    maf["position"] = pd.to_numeric(maf["position"], errors="raise").astype(int)
    maf = maf.sort_values(["ancestry", "chromosome", "position"]).drop_duplicates(
        ["ancestry", "chromosome", "position"]
    )
    maf_lookup = maf[
        ["ancestry", "chromosome", "position", "a1_frequency", "maf"]
    ].rename(columns={"a1_frequency": "plink_a1_frequency", "maf": "plink_maf"})

    combined = pd.read_csv(args.combined, sep="\t", low_memory=False)
    combined["position"] = pd.to_numeric(combined["position"], errors="coerce")
    combined = combined[combined["position"].notna()].copy()
    combined["position"] = combined["position"].astype(int)
    combined["pip"] = pd.to_numeric(combined["pip"], errors="coerce")
    positive = combined[combined["pip"].ge(0.5)].copy()
    positive = positive.drop_duplicates(["tool", *PAIR_KEYS])
    positive = positive.merge(
        maf_lookup, on=["ancestry", "chromosome", "position"], how="left", validate="many_to_one"
    )
    if positive["plink_maf"].isna().any():
        raise RuntimeError("Missing exact-cohort PLINK MAF for positive FILER inputs")
    positive["variant_id"] = canonical_variant(positive)
    positive["af"] = positive["plink_a1_frequency"]
    positive["maf"] = positive["plink_maf"]
    columns = [
        "ancestry", "tool", "variant_id", "chromosome", "position", "ref_allele",
        "alt_allele", "gene_id", "pip", "CS", "tss_distance_bin", "af", "maf",
        "pvalue", "beta", "std_error", "qvalue",
    ]
    positive[columns].sort_values(
        ["ancestry", "tool", "chromosome", "position", "gene_id"]
    ).to_csv(args.output_dir / "deduplicated_positive_variants_for_filer.tsv", sep="\t", index=False)

    rows = []
    for ancestry in ["AA", "CH", "NHW"]:
        ancestry_positive = positive[positive["ancestry"].eq(ancestry)]
        unique_pairs = combined[
            combined["ancestry"].eq(ancestry) & combined["pip"].ge(0.5)
        ].drop_duplicates(PAIR_KEYS)
        for tool in ["alphagenome", "borzoi"]:
            rows.append(
                {
                    "ancestry": ancestry,
                    "tool": tool,
                    "n_tool_eqtls": int(ancestry_positive["tool"].eq(tool).sum()),
                    "n_unique_variant_gene_pairs": len(unique_pairs),
                }
            )
    pd.DataFrame(rows).to_csv(
        args.output_dir / "positive_variant_counts_by_ancestry.tsv", sep="\t", index=False
    )

    selected = pd.read_csv(args.selected, sep="\t", low_memory=False)
    selected["position"] = pd.to_numeric(selected["position"], errors="raise").astype(int)
    selected["pip"] = pd.to_numeric(selected["pip"], errors="raise")
    missing_class = selected["selection_class"].isna()
    selected.loc[missing_class & selected["pip"].ge(0.5), "selection_class"] = "positive_pip_ge_0.5"
    selected.loc[missing_class & selected["pip"].lt(0.01), "selection_class"] = "sampled_negative_pip_lt_0.01"
    selected.loc[
        missing_class & selected["pip"].ge(0.01) & selected["pip"].lt(0.5), "selection_class"
    ] = "sampled_intermediate_pip_0.01_to_0.5"
    selected = selected.merge(
        maf_lookup, on=["ancestry", "chromosome", "position"], how="left", validate="many_to_one"
    )
    if selected["plink_maf"].isna().any():
        raise RuntimeError("Missing exact-cohort PLINK MAF for selected benchmark inputs")
    selected["maf_bin"] = pd.cut(
        selected["plink_maf"], bins=[-np.inf, 0.05, 0.2, np.inf],
        labels=["lt0.05", "0.05-0.2", "ge0.2"], right=False,
    ).astype(str)
    summary = selected.groupby(["ancestry", "selection_class"], observed=True).size().rename("n").reset_index()
    strata = selected.groupby(
        ["ancestry", "selection_class", "tss_distance_bin", "maf_bin"], observed=True
    ).size().rename("n").reset_index()
    summary.to_csv(args.output_dir / "selected_finemapped_eqtl_pair_summary.tsv", sep="\t", index=False)
    strata.to_csv(
        args.output_dir / "selected_finemapped_eqtl_pair_strata_summary.tsv", sep="\t", index=False
    )

    print(pd.DataFrame(rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
