#!/usr/bin/env python3
"""Summarize sparse r2, per-variant neighbor burden, and gene coverage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R2_BINS = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.000001])


def quantile_fields(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_q05": float(np.quantile(values, .05)),
        f"{prefix}_q25": float(np.quantile(values, .25)),
        f"{prefix}_median": float(np.quantile(values, .50)),
        f"{prefix}_q75": float(np.quantile(values, .75)),
        f"{prefix}_q95": float(np.quantile(values, .95)),
        f"{prefix}_maximum": float(np.max(values)),
    }


def main() -> None:
    chromosome_rows = []
    histogram_rows = []
    gene_rows = []
    ancestry_scores: dict[str, list[np.ndarray]] = {a: [] for a in ("AA", "CH", "NHW")}
    ancestry_degrees: dict[str, list[np.ndarray]] = {a: [] for a in ("AA", "CH", "NHW")}
    for ancestry in ("AA", "CH", "NHW"):
        for chromosome in range(1, 23):
            qc_path = ROOT / f"qc/ld_chromosomes/{ancestry}.chr{chromosome}.json"
            qc = json.loads(qc_path.read_text())
            if not qc.get("ready"):
                raise RuntimeError(f"LD chromosome is not ready: {ancestry} chr{chromosome}")
            coverage = pd.read_csv(qc["variant_coverage_output"], sep="\t")
            valid = coverage.loc[coverage["ld_calculable"].eq(1), "variant_id"].astype(str).to_numpy()
            lookup = {variant: index for index, variant in enumerate(valid)}
            # This is a deliberately truncated local LD score: self-r2 plus
            # direct r2 values at the stored sparse edge floor (r2 >= 0.1).
            ld_score = np.ones(len(valid), dtype=float)
            neighbor_degree = np.zeros(len(valid), dtype=np.int32)
            bin_counts = np.zeros(len(R2_BINS) - 1, dtype=np.int64)
            edge_rows = 0
            for chunk in pd.read_csv(qc["ld_output"], sep=r"\s+", compression="gzip", chunksize=500000):
                r2 = pd.to_numeric(chunk["R2"], errors="raise").to_numpy(float)
                left = chunk["SNP_A"].astype(str).map(lookup).to_numpy()
                right = chunk["SNP_B"].astype(str).map(lookup).to_numpy()
                if pd.isna(left).any() or pd.isna(right).any():
                    raise RuntimeError(f"LD edge references noncalculable variant: {ancestry} chr{chromosome}")
                left = left.astype(np.int64)
                right = right.astype(np.int64)
                np.add.at(ld_score, left, r2)
                np.add.at(ld_score, right, r2)
                np.add.at(neighbor_degree, left, 1)
                np.add.at(neighbor_degree, right, 1)
                bin_counts += np.histogram(r2, bins=R2_BINS)[0]
                edge_rows += len(chunk)
            if edge_rows != int(qc["sparse_ld_edges"]):
                raise RuntimeError(f"LD edge row mismatch: {ancestry} chr{chromosome}")
            ancestry_scores[ancestry].append(ld_score)
            ancestry_degrees[ancestry].append(neighbor_degree.astype(float))
            chromosome_rows.append({
                "scope": "chromosome", "ancestry": ancestry, "chromosome": chromosome,
                "ld_calculable_variants": len(valid), "sparse_edges": edge_rows,
                **quantile_fields("sparse_local_ld_score_r2_ge_0_1_including_self", ld_score),
                **quantile_fields("neighbors_r2_ge_0_1", neighbor_degree.astype(float)),
            })
            for left_edge, right_edge, count in zip(R2_BINS[:-1], R2_BINS[1:], bin_counts, strict=True):
                histogram_rows.append({
                    "ancestry": ancestry, "chromosome": chromosome,
                    "r2_left_inclusive": left_edge,
                    "r2_right_exclusive": 1.0 if right_edge > 1 else right_edge,
                    "edges": int(count),
                })

            pair_path = ROOT / f"manifests/pairs/{ancestry}.chr{chromosome}.pairs.tsv.gz"
            pairs = pd.read_csv(pair_path, sep="\t", usecols=["gene_id", "variant_id"])
            pairs["ld_calculable"] = pairs["variant_id"].astype(str).isin(lookup)
            gene = pairs.groupby("gene_id", as_index=False).agg(
                native_pairs=("variant_id", "size"),
                ld_calculable_pairs=("ld_calculable", "sum"),
                unique_native_variants=("variant_id", "nunique"),
            )
            gene["ancestry"] = ancestry
            gene["chromosome"] = chromosome
            gene["ld_calculable_pair_fraction"] = gene["ld_calculable_pairs"] / gene["native_pairs"]
            gene_rows.append(gene)

    for ancestry in ("AA", "CH", "NHW"):
        scores = np.concatenate(ancestry_scores[ancestry])
        degrees = np.concatenate(ancestry_degrees[ancestry])
        chromosome_rows.append({
            "scope": "ancestry", "ancestry": ancestry, "chromosome": "all",
            "ld_calculable_variants": len(scores),
            "sparse_edges": int(sum(
                row["sparse_edges"] for row in chromosome_rows
                if row["scope"] == "chromosome" and row["ancestry"] == ancestry
            )),
            **quantile_fields("sparse_local_ld_score_r2_ge_0_1_including_self", scores),
            **quantile_fields("neighbors_r2_ge_0_1", degrees),
        })

    architecture = pd.DataFrame(chromosome_rows)
    histogram = pd.DataFrame(histogram_rows)
    genes = pd.concat(gene_rows, ignore_index=True)
    architecture_path = ROOT / "qc/native_ld_architecture_summary.tsv"
    histogram_path = ROOT / "qc/native_ld_r2_histogram.tsv"
    gene_path = ROOT / "qc/native_ld_gene_coverage.tsv.gz"
    architecture.to_csv(architecture_path, sep="\t", index=False)
    histogram.to_csv(histogram_path, sep="\t", index=False)
    genes.to_csv(gene_path, sep="\t", index=False, compression="gzip")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chromosome_units": 66,
        "r2_edge_floor": 0.1,
        "local_ld_score_definition": "self_r2_plus_sum_of_stored_direct_r2_ge_0.1",
        "architecture_rows": len(architecture),
        "histogram_rows": len(histogram),
        "gene_coverage_rows": len(genes),
        "architecture_output": str(architecture_path),
        "r2_histogram_output": str(histogram_path),
        "gene_coverage_output": str(gene_path),
        "ready": True,
    }
    output = ROOT / "qc/native_ld_architecture_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
