#!/usr/bin/env python3
"""
Materialize LD-filtered negative-class prediction sets from the existing scored SuSiE universe.

This does not recompute Borzoi or AlphaGenome scores. Instead, it exports new negative-class
prediction tables after applying the same r^2-based filters used in the contamination analysis.

Outputs mirror the original outputs_susie layout:

  manuscript_negative_contamination/filtered_predictions/{negative_definition}/
    analysis/susie_s2f_combined.tsv.gz
    analysis/negative_prediction_export_summary.tsv
    {tool}/{ancestry}/{tool}.blood.requested_pairs.tsv.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_negative_contamination_analysis import (
    ANCESTRIES,
    NEGATIVE_DEFS,
    TOOLS,
    build_negative_subset,
    ensure_dir,
    load_scored_susie,
    merge_contamination_to_scored_rows,
)


def build_export_summary(df: pd.DataFrame, negative_definition: str) -> pd.DataFrame:
    rows = []
    for tool in TOOLS:
        tool_df = df[df["tool"] == tool].copy()
        for ancestry in ANCESTRIES:
            sub = tool_df[tool_df["ancestry"] == ancestry].copy()
            rows.append(
                {
                    "negative_definition": negative_definition,
                    "tool": tool,
                    "ancestry": ancestry,
                    "n_rows": int(len(sub)),
                    "n_unique_variant_locus": int(sub["variant_locus_key"].nunique()),
                    "n_unique_genes": int(sub["gene_id"].nunique()),
                    "n_rows_with_ld": int(sub["ld_available"].fillna(False).sum()),
                }
            )
    return pd.DataFrame(rows)


def write_negative_definition_exports(
    scored_ld_df: pd.DataFrame,
    negative_definition: str,
    output_root: Path,
) -> pd.DataFrame:
    subset = build_negative_subset(scored_ld_df, negative_definition).copy()
    subset["negative_definition"] = negative_definition
    subset["negative_definition_label"] = NEGATIVE_DEFS[negative_definition]["label"]
    subset = subset.sort_values(
        ["tool", "ancestry", "gene_id", "chrom_u", "pos_u", "ref_u", "alt_u"],
        kind="mergesort",
    ).reset_index(drop=True)

    def_root = output_root / negative_definition
    analysis_dir = def_root / "analysis"
    ensure_dir(analysis_dir)

    subset.to_csv(analysis_dir / "susie_s2f_combined.tsv.gz", sep="\t", index=False, compression="gzip")

    summary = build_export_summary(subset, negative_definition)
    summary.to_csv(analysis_dir / "negative_prediction_export_summary.tsv", sep="\t", index=False)

    for tool in TOOLS:
        tool_df = subset[subset["tool"] == tool].copy()
        for ancestry in ANCESTRIES:
            anc_df = tool_df[tool_df["ancestry"] == ancestry].copy()
            anc_dir = def_root / tool / ancestry
            ensure_dir(anc_dir)
            anc_df.to_csv(
                anc_dir / f"{tool}.blood.requested_pairs.tsv.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LD-filtered negative-class prediction files")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contamination-table",
        type=Path,
        default=Path.cwd() / "manuscript_negative_contamination" / "results" / "variant_level_negative_ld_contamination.tsv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "manuscript_negative_contamination" / "filtered_predictions",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    contamination_table = args.contamination_table.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    if not contamination_table.exists():
        raise FileNotFoundError(f"Contamination table not found: {contamination_table}")

    print("[1/3] Loading scored SuSiE predictions")
    scored_df = load_scored_susie(base_dir)
    print(f"  Scored rows: {len(scored_df):,}")

    print("[2/3] Loading variant-level LD contamination annotations")
    variant_level = pd.read_csv(contamination_table, sep="\t", low_memory=False)
    print(f"  Variant-level rows: {len(variant_level):,}")

    print("[3/3] Merging annotations and exporting filtered negative-class prediction sets")
    scored_ld_df = merge_contamination_to_scored_rows(scored_df, variant_level)

    all_summaries = []
    for negative_definition in NEGATIVE_DEFS:
        summary = write_negative_definition_exports(scored_ld_df, negative_definition, output_dir)
        all_summaries.append(summary)
        print(
            f"  {negative_definition}: "
            f"{int(summary['n_rows'].sum()):,} rows "
            f"across {int(summary['n_unique_variant_locus'].sum()):,} tool-ancestry variant-locus entries"
        )

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(output_dir / "negative_prediction_export_summary.tsv", sep="\t", index=False)

    readme_lines = [
        "LD-filtered negative-class prediction exports",
        "",
        "These files reuse the original Borzoi and AlphaGenome scores from outputs_susie/analysis/susie_s2f_combined.tsv.gz.",
        "The LD filter changes negative-class membership only; it does not change the underlying prediction value.",
        "",
        "Each negative definition has its own outputs_susie-like subtree:",
        "  {negative_definition}/analysis/susie_s2f_combined.tsv.gz",
        "  {negative_definition}/{tool}/{ancestry}/{tool}.blood.requested_pairs.tsv.gz",
        "",
        "Negative definitions:",
    ]
    for key, meta in NEGATIVE_DEFS.items():
        readme_lines.append(f"  {key}: {meta['label']}")
    (output_dir / "README.txt").write_text("\n".join(readme_lines) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
