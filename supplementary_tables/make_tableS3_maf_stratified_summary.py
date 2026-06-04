#!/usr/bin/env python3
"""Build Supplementary Table S3 from the Figure S1 AUROC-by-threshold table."""

import argparse
from pathlib import Path

import pandas as pd


TOOLS = ["borzoi", "alphagenome"]
ANCESTRIES = ["AA", "CH", "NHW"]
MAF_BINS = ["lt0.05", "0.05-0.2", "ge0.2"]


def value_range(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if values.empty:
        return ""
    low = int(values.min())
    high = int(values.max())
    return str(low) if low == high else f"{low}-{high}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Supplementary Table S3.")
    parser.add_argument("--input", type=Path, required=True, help="Figure S1 AUROC-by-threshold TSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    df["auc_mean"] = pd.to_numeric(df["auc_mean"], errors="coerce")

    rows = []
    for ancestry in ANCESTRIES:
        for maf_bin in MAF_BINS:
            sub = df[(df["ancestry"] == ancestry) & (df["maf_bin"] == maf_bin)].copy()
            if sub.empty:
                continue
            label = sub["maf_bin_label"].dropna().iloc[0]
            row = {"ancestry": ancestry, "maf_bin": label}
            for tool in TOOLS:
                tool_df = sub[sub["tool"] == tool]
                row[f"{tool}_mean_auroc"] = f"{tool_df['auc_mean'].mean():.3f}" if len(tool_df) else ""
                row[f"{tool}_n_range"] = value_range(tool_df["n_pos_raw"]) if len(tool_df) else ""
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
