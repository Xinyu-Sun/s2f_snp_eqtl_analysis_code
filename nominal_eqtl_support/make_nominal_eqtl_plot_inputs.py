#!/usr/bin/env python3
"""Build combined 10k-cap plot inputs from the completed 5k run plus top-up outputs."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_REDO_ROOT = PROJECT_ROOT / "nominal_eqtl_redo"
TOPUP_ROOT = PROJECT_ROOT / "nominal_eqtl_redo_10k_topup"
sys.path.insert(0, str(BASE_REDO_ROOT / "scripts"))

import make_nominal_redo_plot_inputs as helper  # noqa: E402


def load_long_from_root(root):
    return pd.concat(
        [helper.load_tool_outputs(root, tool) for tool in helper.TOOL_CONFIG],
        ignore_index=True,
    )


def normalize_numeric(long_df):
    numeric_cols = [
        "prediction",
        "num_ancestries_shared",
        "position",
        "TSS",
        "distance_to_tss",
        "alt_allele_freq",
        "pvalue",
        "beta",
        "std_error",
        "qvalue",
    ]
    for col in numeric_cols:
        if col in long_df.columns:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
    return long_df


def main():
    analysis_dir = TOPUP_ROOT / "outputs" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base_long = load_long_from_root(BASE_REDO_ROOT)
    topup_long = load_long_from_root(TOPUP_ROOT)
    long_df = pd.concat([base_long, topup_long], ignore_index=True)
    long_df = normalize_numeric(long_df)

    long_cols = helper.BASE_COLS + [
        c
        for c in ["logMeanSED", "raw_score"]
        if c in long_df.columns and c not in helper.BASE_COLS
    ]
    long_df = long_df[long_cols]

    dup_n = int(long_df.duplicated(["tool", "nominal_redo_record_id"]).sum())
    if dup_n:
        raise RuntimeError("Found {} duplicate tool/nominal_redo_record_id rows after merge".format(dup_n))

    wide_df = helper.make_wide(long_df)
    both_df = wide_df[wide_df["has_both_models"]].copy()

    count_cols = ["tool", "population", "tss_distance_bin", "sharing_class"]
    counts = (
        long_df.groupby(count_cols, dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(count_cols)
    )
    convergence = helper.compute_model_convergence_results(wide_df)
    tool_performance = helper.compute_tool_performance_results(long_df)

    long_path = analysis_dir / "nominal_redo_10k_predictions_long.tsv.gz"
    wide_path = analysis_dir / "nominal_redo_10k_predictions_wide.tsv.gz"
    both_path = analysis_dir / "nominal_redo_10k_predictions_both_models.tsv.gz"
    counts_path = analysis_dir / "nominal_redo_10k_plot_input_counts.tsv"
    convergence_path = analysis_dir / "nominal_redo_10k_model_convergence_results.tsv"
    tool_performance_path = analysis_dir / "nominal_redo_10k_tool_performance_results.tsv"

    long_df.to_csv(long_path, sep="\t", index=False, compression="gzip")
    wide_df.to_csv(wide_path, sep="\t", index=False, compression="gzip")
    both_df.to_csv(both_path, sep="\t", index=False, compression="gzip")
    counts.to_csv(counts_path, sep="\t", index=False)
    convergence.to_csv(convergence_path, sep="\t", index=False)
    tool_performance.to_csv(tool_performance_path, sep="\t", index=False)

    print("[OK] long table: {:,} rows -> {}".format(len(long_df), long_path))
    print("[OK] wide table: {:,} rows -> {}".format(len(wide_df), wide_path))
    print("[OK] both-model table: {:,} rows -> {}".format(len(both_df), both_path))
    print("[OK] counts: {:,} rows -> {}".format(len(counts), counts_path))
    print("[OK] model convergence table: {:,} rows -> {}".format(len(convergence), convergence_path))
    print("[OK] tool performance table: {:,} rows -> {}".format(len(tool_performance), tool_performance_path))


if __name__ == "__main__":
    main()
