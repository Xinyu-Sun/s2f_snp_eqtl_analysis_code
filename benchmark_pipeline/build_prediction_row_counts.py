#!/usr/bin/env python3
"""Set model-availability flags from score non-missingness and write the prediction-table row counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LINEAGE = {
    "nominal_eqtl_10k_predictions_wide.tsv.gz": "nominal-eQTL stratified benchmark",
    "finemapped_pip_ge_0.9_predictions_wide.tsv.gz": "high-confidence fine-mapped PIP >= 0.9",
    "finemapped_auroc_selected_pairs_predictions_wide.tsv.gz": "fine-mapped AUROC selected-pair analysis",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=Path)
    args = parser.parse_args()

    census = []
    for filename, lineage in LINEAGE.items():
        path = args.prediction_dir / filename
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        required = {"borzoi_score", "alphagenome_score"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"{filename} lacks score columns: {missing}")

        frame["has_borzoi"] = frame["borzoi_score"].notna()
        frame["has_alphagenome"] = frame["alphagenome_score"].notna()
        frame["has_both_models"] = frame["has_borzoi"] & frame["has_alphagenome"]

        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
        temporary.replace(path)

        census.append(
            {
                "table": filename,
                "rows": len(frame),
                "bytes_gzip": path.stat().st_size,
                "has_borzoi_true": int(frame["has_borzoi"].sum()),
                "has_alphagenome_true": int(frame["has_alphagenome"].sum()),
                "has_both_models_true": int(frame["has_both_models"].sum()),
                "analysis_lineage": lineage,
            }
        )

    census_path = args.prediction_dir / "detailed_prediction_row_counts.tsv"
    pd.DataFrame(census).to_csv(census_path, sep="\t", index=False)
    print(pd.DataFrame(census).to_string(index=False))


if __name__ == "__main__":
    main()
