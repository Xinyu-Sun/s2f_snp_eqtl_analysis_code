#!/usr/bin/env python3
"""Prepare variant extract lists for the CH benchmark PLINK MAF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.selected_pairs, sep="\t", low_memory=False)
    frame["chromosome"] = frame["chromosome"].astype(str).str.removeprefix("chr").astype(int)
    frame["position"] = pd.to_numeric(frame["position"], errors="raise").astype(int)
    frame["variant_id"] = (
        "chr" + frame["chromosome"].astype(str) + ":" + frame["position"].astype(str)
        + "_" + frame["ref_allele"].astype(str) + "_" + frame["alt_allele"].astype(str)
    )
    frame["association_maf"] = np.minimum(
        pd.to_numeric(frame["alt_allele_freq"], errors="coerce"),
        1 - pd.to_numeric(frame["alt_allele_freq"], errors="coerce"),
    )
    metadata = frame[
        [
            "variant_id", "chromosome", "position", "ref_allele", "alt_allele",
            "association_maf",
        ]
    ].drop_duplicates("variant_id")
    metadata.to_csv(
        args.output_dir / "ch_selected_variant_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    metadata["variant_id"].sort_values().to_csv(
        args.output_dir / "CH.all.extract_ids.txt", index=False, header=False
    )
    counts = []
    for chromosome in range(1, 23):
        identifiers = metadata.loc[metadata["chromosome"].eq(chromosome), "variant_id"].sort_values()
        identifiers.to_csv(
            args.output_dir / f"CH.chr{chromosome}.extract_ids.txt",
            index=False,
            header=False,
        )
        counts.append({"chromosome": chromosome, "unique_variants": len(identifiers)})
    pd.DataFrame(counts).to_csv(args.output_dir / "variant_counts_by_chromosome.tsv", sep="\t", index=False)
    summary = {"selected_pair_rows": len(frame), "selected_unique_variants": len(metadata)}
    (args.output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
