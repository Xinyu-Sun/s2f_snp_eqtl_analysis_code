#!/usr/bin/env python3
"""Build Supplementary Table S2 from packaged positive-variant allele frequencies."""

import argparse
from pathlib import Path

import pandas as pd


TOOLS = ["borzoi", "alphagenome"]
ANCESTRIES = ["AA", "CH", "NHW"]
PIP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def summarize(sub: pd.DataFrame, tool: str, ancestry: str, threshold: float) -> dict:
    af = pd.to_numeric(sub["af"], errors="coerce")
    maf = pd.to_numeric(sub["maf"], errors="coerce")
    mask = af.notna() & maf.notna()
    af = af[mask]
    maf = maf[mask]
    return {
        "tool": tool,
        "ancestry": ancestry,
        "pip_threshold": threshold,
        "n_positives": int(len(af)),
        "af_mean": float(af.mean()),
        "af_median": float(af.median()),
        "af_q25": float(af.quantile(0.25)),
        "af_q75": float(af.quantile(0.75)),
        "maf_mean": float(maf.mean()),
        "maf_median": float(maf.median()),
        "maf_q25": float(maf.quantile(0.25)),
        "maf_q75": float(maf.quantile(0.75)),
        "pct_maf_lt_0_01": float((maf < 0.01).mean() * 100),
        "pct_maf_lt_0_05": float((maf < 0.05).mean() * 100),
        "pct_maf_ge_0_05": float((maf >= 0.05).mean() * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Supplementary Table S2.")
    parser.add_argument("--input", type=Path, required=True, help="Deduplicated positive-variant TSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path.")
    args = parser.parse_args()

    positives = pd.read_csv(args.input, sep="\t")
    positives["pip"] = pd.to_numeric(positives["pip"], errors="coerce")

    rows = []
    for tool in TOOLS:
        for ancestry in ANCESTRIES:
            base = positives[(positives["tool"] == tool) & (positives["ancestry"] == ancestry)]
            for threshold in PIP_THRESHOLDS:
                sub = base[base["pip"] >= threshold]
                if len(sub):
                    rows.append(summarize(sub, tool, ancestry, threshold))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
