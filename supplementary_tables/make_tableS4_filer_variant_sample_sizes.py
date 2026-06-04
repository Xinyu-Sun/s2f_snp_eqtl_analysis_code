#!/usr/bin/env python3
"""Build Supplementary Table S4 from the deduplicated positive-variant table."""

import argparse
from pathlib import Path

import pandas as pd


TOOLS = ["alphagenome", "borzoi"]
TOOL_LABELS = {"alphagenome": "AlphaGenome", "borzoi": "Borzoi"}
ANCESTRIES = ["AA", "CH", "NHW"]
PIP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Supplementary Table S4.")
    parser.add_argument("--input", type=Path, required=True, help="Deduplicated positive-variant TSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    df["pip"] = pd.to_numeric(df["pip"], errors="coerce")

    rows = []
    for tool in TOOLS:
        tool_df = df[df["tool"] == tool]
        for threshold in PIP_THRESHOLDS:
            row = {"model": TOOL_LABELS[tool], "pip_threshold": f">={threshold:g}"}
            for ancestry in ANCESTRIES:
                sub = tool_df[(tool_df["ancestry"] == ancestry) & (tool_df["pip"] >= threshold)]
                row[ancestry] = int(sub["variant_id"].nunique())
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
