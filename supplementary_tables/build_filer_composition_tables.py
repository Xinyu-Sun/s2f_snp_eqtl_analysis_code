#!/usr/bin/env python3
"""Compute FILER annotation composition of the high-PIP positive sets (sources of Tables S19 and S20)."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


MODELS = ["alphagenome", "borzoi"]
ANCESTRIES = ["AA", "CH", "NHW"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
KEY_CATEGORIES = ["Accessible_chromatin", "Hi-C", "IM-PET", "pcHi-C", "eQTL", "sQTL"]
DISPLAY = {
    "Accessible_chromatin": "Accessible chromatin",
    "Hi-C": "Hi-C",
    "IM-PET": "IM-PET",
    "pcHi-C": "pcHi-C",
    "eQTL": "eQTL",
    "sQTL": "sQTL",
}


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = (ordered.to_numpy() * len(ordered) / ranks)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def coalesce(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    left = frame[primary] if primary in frame else pd.Series(np.nan, index=frame.index)
    right = frame[fallback] if fallback in frame else pd.Series(np.nan, index=frame.index)
    return left.where(left.notna() & ~left.astype(str).isin(["", "nan"]), right)


def canonical_variant_id(frame: pd.DataFrame) -> pd.Series:
    chromosome = coalesce(frame, "chromosome", "chr")
    position = pd.to_numeric(coalesce(frame, "position", "pos"), errors="raise").astype(int)
    ref = coalesce(frame, "ref_allele", "ref")
    alt = coalesce(frame, "alt_allele", "alt")
    return (
        chromosome.astype(str) + ":" + position.astype(str) + "_"
        + ref.astype(str) + "_" + alt.astype(str)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.read_csv(args.combined, sep="\t", low_memory=False)
    coordinate_position = coalesce(combined, "position", "pos")
    numeric_position = pd.to_numeric(coordinate_position, errors="coerce")
    malformed = numeric_position.isna()
    if malformed.any():
        repeated_headers = coordinate_position.loc[malformed].astype(str).isin(["position", "pos"])
        if not repeated_headers.all():
            examples = coordinate_position.loc[malformed].astype(str).head().tolist()
            raise RuntimeError(f"Malformed positions in combined score table: {examples}")
        combined = combined.loc[~malformed].copy()
    combined["variant_id"] = canonical_variant_id(combined)
    combined["pip"] = pd.to_numeric(combined["pip"], errors="coerce")
    annotations = pd.read_csv(args.annotations, sep="\t", low_memory=False)
    binary_columns = [column for column in annotations if column.startswith("filer_binary_")]
    categories = [column.removeprefix("filer_binary_") for column in binary_columns]
    annotations = annotations[["variant_id", *binary_columns]].drop_duplicates("variant_id")

    sample_rows = []
    composition_rows = []
    pairwise_rows = []
    for model in MODELS:
        for threshold in THRESHOLDS:
            ancestry_tables = {}
            for ancestry in ANCESTRIES:
                positives = combined[
                    combined["tool"].eq(model)
                    & combined["ancestry"].eq(ancestry)
                    & combined["pip"].ge(threshold)
                ][["variant_id"]].drop_duplicates()
                merged = positives.merge(annotations, on="variant_id", how="left", validate="one_to_one")
                if merged[binary_columns].isna().any().any():
                    missing = int(merged[binary_columns].isna().all(axis=1).sum())
                    raise RuntimeError(f"Missing FILER annotation rows for {missing} {model}/{ancestry}/{threshold} positives")
                merged[binary_columns] = merged[binary_columns].astype(int)
                ancestry_tables[ancestry] = merged
                sample_rows.append({
                    "model": model,
                    "pip_threshold": threshold,
                    "ancestry": ancestry,
                    "n_unique_positive_variants": len(merged),
                })

            for category in categories:
                column = f"filer_binary_{category}"
                counts = [int(ancestry_tables[a][column].sum()) for a in ANCESTRIES]
                totals = [len(ancestry_tables[a]) for a in ANCESTRIES]
                table = np.asarray([[yes, total - yes] for yes, total in zip(counts, totals)])
                if table.sum() == 0 or np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
                    # Preserve the manuscript analysis convention: a category
                    # with no distributional information contributes p=1 to
                    # the complete per-model/per-threshold FDR family.
                    p_value = 1.0
                else:
                    _, p_value, _, _ = chi2_contingency(table, correction=False)
                row = {
                    "model": model,
                    "pip_threshold": threshold,
                    "category": category,
                    "chi_square_p": p_value,
                }
                for ancestry, yes, total in zip(ANCESTRIES, counts, totals):
                    row[f"{ancestry}_n"] = yes
                    row[f"{ancestry}_total"] = total
                    row[f"{ancestry}_percent"] = 100 * yes / total if total else np.nan
                composition_rows.append(row)

                for ancestry_a, ancestry_b in combinations(ANCESTRIES, 2):
                    total_a, total_b = len(ancestry_tables[ancestry_a]), len(ancestry_tables[ancestry_b])
                    yes_a = int(ancestry_tables[ancestry_a][column].sum())
                    yes_b = int(ancestry_tables[ancestry_b][column].sum())
                    _, fisher_p = fisher_exact(
                        [[yes_a, total_a - yes_a], [yes_b, total_b - yes_b]], alternative="two-sided"
                    )
                    frac_a = 100 * yes_a / total_a if total_a else np.nan
                    frac_b = 100 * yes_b / total_b if total_b else np.nan
                    pairwise_rows.append({
                        "model": model,
                        "pip_threshold": threshold,
                        "category": category,
                        "comparison": f"{ancestry_a} vs {ancestry_b}",
                        "ancestry_a": ancestry_a,
                        "ancestry_b": ancestry_b,
                        "frac1_percent": frac_a,
                        "frac2_percent": frac_b,
                        "diff_percent_points": frac_a - frac_b,
                        "fisher_p": fisher_p,
                    })

    samples_long = pd.DataFrame(sample_rows)
    samples = samples_long.pivot(
        index=["model", "pip_threshold"], columns="ancestry", values="n_unique_positive_variants"
    ).reset_index()
    composition = pd.DataFrame(composition_rows)
    composition["chi_square_q"] = composition.groupby(
        ["model", "pip_threshold"], group_keys=False
    )["chi_square_p"].apply(bh)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["fisher_q"] = pairwise.groupby(
        ["model", "pip_threshold"], group_keys=False
    )["fisher_p"].apply(bh)

    key_composition = composition[
        composition["category"].isin(KEY_CATEGORIES)
        & composition["pip_threshold"].isin([0.5, 0.9])
    ].copy()
    key_composition["category_display"] = key_composition["category"].map(DISPLAY)
    key_pairwise = pairwise[
        pairwise["category"].isin(KEY_CATEGORIES)
        & pairwise["pip_threshold"].isin([0.5, 0.9])
    ].copy()
    key_pairwise["category_display"] = key_pairwise["category"].map(DISPLAY)

    samples.to_csv(args.output_dir / "filer_positive_variant_sample_sizes.tsv", sep="\t", index=False)
    key_composition.to_csv(args.output_dir / "filer_annotation_overlap_key.tsv", sep="\t", index=False)
    key_pairwise.to_csv(args.output_dir / "filer_pairwise_comparisons_key.tsv", sep="\t", index=False)
    composition.to_csv(args.output_dir / "filer_positive_composition_all_categories.tsv", sep="\t", index=False)
    pairwise.to_csv(args.output_dir / "filer_positive_pairwise_all_categories.tsv", sep="\t", index=False)

    summary = samples_long.groupby(["model", "ancestry"])["n_unique_positive_variants"].agg(["min", "max"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
