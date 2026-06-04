#!/usr/bin/env python3
"""
Recompute negative-cleaning AUROC sensitivity using collected rerun outputs.

Positives always come from the original scored SuSiE universe.
Negatives are:
  - original: original scored universe with PIP < 0.01
  - LD-cleaned: collected rerun outputs for each filtered negative definition
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_negative_contamination_analysis import (
    ANCESTRIES,
    NEGATIVE_DEFS,
    PIP_THRESHOLDS,
    TOOLS,
    ANCESTRY_LABELS,
    add_auc_delta_columns,
    add_tss_distance,
    compute_bootstrap_auroc,
    ensure_dir,
    safe_numeric,
)


def normalize_scored_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["pip", "logMeanSED", "raw_score", "pos", "position"]:
        if col in out.columns:
            out[col] = safe_numeric(out[col])

    chrom_source = out["chromosome"] if "chromosome" in out.columns else pd.Series(pd.NA, index=out.index)
    if "chr" in out.columns:
        chrom_source = chrom_source.fillna(out["chr"])
    out["chrom_u"] = chrom_source.astype(str).str.replace("^chr", "", regex=True).radd("chr")

    pos_source = out["position"] if "position" in out.columns else pd.Series(pd.NA, index=out.index)
    if "pos" in out.columns:
        pos_source = pos_source.fillna(out["pos"])
    out["pos_u"] = pos_source
    out["pos_u"] = safe_numeric(out["pos_u"]).astype("Int64")

    ref_source = out["ref_allele"] if "ref_allele" in out.columns else pd.Series(pd.NA, index=out.index)
    if "ref" in out.columns:
        ref_source = ref_source.fillna(out["ref"])
    out["ref_u"] = ref_source

    alt_source = out["alt_allele"] if "alt_allele" in out.columns else pd.Series(pd.NA, index=out.index)
    if "alt" in out.columns:
        alt_source = alt_source.fillna(out["alt"])
    out["alt_u"] = alt_source
    out["variant_pos"] = safe_numeric(out["pos_u"])
    out["pred_score"] = out["logMeanSED"].where(out["tool"] == "borzoi", out["raw_score"])
    return out


def load_rerun_negative_table(base_dir: Path, negative_definition: str) -> pd.DataFrame:
    path = (
        base_dir
        / "manuscript_negative_contamination"
        / "filtered_predictions"
        / negative_definition
        / "rerun_analysis"
        / "susie_s2f_combined.tsv.gz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing rerun analysis table: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["tool"].isin(TOOLS) & df["ancestry"].isin(ANCESTRIES)].copy()
    return normalize_scored_df(df)


def compute_gap_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    focus = summary_df[summary_df["pip_threshold"] == 0.9].copy()
    aa = focus[focus["ancestry"] == "AA"][["tool", "negative_definition", "auc_mean"]].rename(
        columns={"auc_mean": "auc_mean_aa"}
    )
    gaps = focus.merge(aa, on=["tool", "negative_definition"], how="left")
    gaps["gap_vs_aa"] = gaps["auc_mean_aa"] - gaps["auc_mean"]
    return gaps.sort_values(["tool", "negative_definition", "ancestry"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun-based negative-cleaning AUROC sensitivity")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "manuscript_negative_contamination" / "results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    original = pd.read_csv(base_dir / "outputs_susie" / "analysis" / "susie_s2f_combined.tsv.gz", sep="\t", low_memory=False)
    original = original[original["tool"].isin(TOOLS) & original["ancestry"].isin(ANCESTRIES)].copy()
    original = normalize_scored_df(original)
    original = add_tss_distance(original, base_dir)

    negative_tables = {
        "pip_lt_0.01": original[original["pip"] < 0.01].copy(),
        "pip_lt_0.01_ld_lt_0.2": add_tss_distance(load_rerun_negative_table(base_dir, "pip_lt_0.01_ld_lt_0.2"), base_dir),
        "pip_lt_0.01_ld_lt_0.1": add_tss_distance(load_rerun_negative_table(base_dir, "pip_lt_0.01_ld_lt_0.1"), base_dir),
    }

    summary_rows = []
    for tool in TOOLS:
        tool_df = original[original["tool"] == tool].copy()
        for ancestry in ANCESTRIES:
            positives_base = tool_df[tool_df["ancestry"] == ancestry].copy()
            if positives_base.empty:
                continue

            for negative_definition, negatives_full in negative_tables.items():
                negatives = negatives_full[
                    (negatives_full["tool"] == tool) & (negatives_full["ancestry"] == ancestry)
                ].copy()

                for pip_threshold in PIP_THRESHOLDS:
                    result = compute_bootstrap_auroc(
                        positives=positives_base,
                        negatives=negatives,
                        pred_col="pred_score",
                        pip_threshold=pip_threshold,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed,
                        n_workers=args.n_workers,
                    )

                    row = {
                        "tool": tool,
                        "ancestry": ancestry,
                        "ancestry_label": ANCESTRY_LABELS[ancestry],
                        "pip_threshold": pip_threshold,
                        "negative_definition": negative_definition,
                        "negative_definition_label": NEGATIVE_DEFS[negative_definition]["label"],
                    }
                    if result is None:
                        row.update(
                            {
                                "auc_mean": pd.NA,
                                "auc_std": pd.NA,
                                "auc_ci_low": pd.NA,
                                "auc_ci_high": pd.NA,
                                "n_bootstrap": 0,
                                "n_pos_raw": int((positives_base["pip"] >= pip_threshold).sum()),
                                "n_neg_raw": int(len(negatives)),
                                "n_pos_balanced_example": pd.NA,
                                "n_neg_balanced_example": pd.NA,
                            }
                        )
                    else:
                        row.update(
                            {
                                "auc_mean": result["auc_mean"],
                                "auc_std": result["auc_std"],
                                "auc_ci_low": result["auc_ci_low"],
                                "auc_ci_high": result["auc_ci_high"],
                                "n_bootstrap": result["n_bootstrap"],
                                "n_pos_raw": result["n_pos_raw"],
                                "n_neg_raw": result["n_neg_raw"],
                                "n_pos_balanced_example": result["n_pos_balanced_example"],
                                "n_neg_balanced_example": result["n_neg_balanced_example"],
                            }
                        )
                    summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_auc_delta_columns(summary_df)
    gap_df = compute_gap_summary(summary_df)

    summary_df.to_csv(output_dir / "auroc_negative_cleaning_rerun_confirmed_summary.tsv", sep="\t", index=False)
    gap_df.to_csv(output_dir / "auroc_negative_cleaning_rerun_confirmed_gap_summary_pip0.9.tsv", sep="\t", index=False)
    print("Wrote rerun-confirmed AUROC sensitivity outputs.")


if __name__ == "__main__":
    main()
