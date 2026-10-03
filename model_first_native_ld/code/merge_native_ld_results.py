#!/usr/bin/env python3
"""Validate and merge all ancestry/chromosome LD-clump outputs.

This script is deliberately outcome-agnostic with respect to ranking: it only
uses the precomputed model priorities written by the chromosome clump jobs.
eQTL outcomes are carried through for downstream evaluation after index
selection has already been determined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LD_ROOT.parent
ANCESTRIES = ("AA", "CH", "NHW")
PANELS = ("general", "ad_tier1_excluding_apoe", "ad_tier1_apoe_only", "ad_tier2")
METHODS = ("borzoi", "alphagenome", "consensus")
THRESHOLDS = (0.1, 0.2, 0.5)
DEPLOYMENTS = ("native", "three_way_shared")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_table(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(
        temporary,
        sep="\t",
        index=False,
        na_rep="NA",
        compression="gzip" if gzip else None,
    )
    temporary.replace(path)


def build_locus_clusters(analysis_pairs: Path) -> pd.DataFrame:
    genes = pd.read_csv(
        analysis_pairs,
        sep="\t",
        usecols=["gene_id", "chromosome", "tss", "ad_compound_locus_cluster"],
        dtype={"gene_id": str},
    ).drop_duplicates()
    if genes["gene_id"].duplicated().any():
        conflict = genes.loc[genes["gene_id"].duplicated(False), "gene_id"].nunique()
        raise RuntimeError(f"{conflict} genes have inconsistent chromosome/TSS annotations")
    genes["chromosome"] = pd.to_numeric(genes["chromosome"], errors="raise").astype(int)
    genes["tss"] = pd.to_numeric(genes["tss"], errors="raise").astype(int)
    result: list[dict[str, object]] = []
    for chromosome, chrom in genes.groupby("chromosome", sort=True):
        chrom = chrom.sort_values(["tss", "gene_id"], kind="mergesort")
        component = 0
        active_end: int | None = None
        for row in chrom.itertuples(index=False):
            start = max(1, int(row.tss) - 100000)
            end = int(row.tss) + 100000
            if active_end is None or start > active_end:
                component += 1
                active_end = end
            else:
                active_end = max(active_end, end)
            result.append({
                "gene_id": str(row.gene_id),
                "local_locus_cluster": f"chr{chromosome}:overlap{component:04d}",
                "ad_compound_locus_cluster": row.ad_compound_locus_cluster,
            })
    mapping = pd.DataFrame(result)
    if mapping["gene_id"].duplicated().any() or len(mapping) != len(genes):
        raise RuntimeError("Locus-cluster mapping is not one-to-one by gene")
    return mapping


def add_score_ranks(indices: pd.DataFrame) -> pd.DataFrame:
    result = indices.copy()
    grouping = ["ancestry", "panel", "deployment", "method", "clump_threshold_r2"]
    result["global_score_rank"] = 0
    result["global_score_count"] = 0
    result["global_ranking_weight_total"] = np.nan
    result["global_top_tail_fraction"] = np.nan
    result["within_gene_score_rank"] = 0
    result["within_gene_score_count"] = 0
    result["within_gene_top_tail_fraction"] = np.nan
    for _, group in result.groupby(grouping, sort=True):
        ordered = group.sort_values(
            ["priority_value_preclump", "index_pair_key"],
            ascending=[False, True],
            kind="mergesort",
        )
        ranks = np.arange(1, len(ordered) + 1, dtype=int)
        ranking_weights = (
            pd.to_numeric(ordered["general_inverse_probability_weight"], errors="raise").to_numpy(float)
            if str(ordered["panel"].iloc[0]) == "general"
            else np.ones(len(ordered), dtype=float)
        )
        if not np.isfinite(ranking_weights).all() or (ranking_weights <= 0).any():
            raise RuntimeError("Invalid global ranking weights")
        total_ranking_weight = float(ranking_weights.sum())
        result.loc[ordered.index, "global_score_rank"] = ranks
        result.loc[ordered.index, "global_score_count"] = len(ordered)
        result.loc[ordered.index, "global_ranking_weight_total"] = total_ranking_weight
        result.loc[ordered.index, "global_top_tail_fraction"] = (
            (np.cumsum(ranking_weights) - ranking_weights) / total_ranking_weight
        )
        for _, gene in group.groupby("gene_id", sort=True):
            ordered_gene = gene.sort_values(
                ["priority_value_preclump", "index_pair_key"],
                ascending=[False, True],
                kind="mergesort",
            )
            gene_ranks = np.arange(1, len(ordered_gene) + 1, dtype=int)
            result.loc[ordered_gene.index, "within_gene_score_rank"] = gene_ranks
            result.loc[ordered_gene.index, "within_gene_score_count"] = len(ordered_gene)
            result.loc[ordered_gene.index, "within_gene_top_tail_fraction"] = (
                (gene_ranks - 1) / len(ordered_gene)
            )
    integer_columns = [
        "global_score_rank", "global_score_count",
        "within_gene_score_rank", "within_gene_score_count",
    ]
    result[integer_columns] = result[integer_columns].astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-pairs",
        type=Path,
        default=PROJECT_ROOT / "results/analysis_pairs.tsv.gz",
    )
    parser.add_argument(
        "--indices-output",
        type=Path,
        default=LD_ROOT / "results/native_ld_indices.tsv.gz",
    )
    parser.add_argument(
        "--memberships-output",
        type=Path,
        default=LD_ROOT / "results/native_ld_primary_memberships.tsv.gz",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=LD_ROOT / "qc/native_ld_merge_summary.json",
    )
    args = parser.parse_args()

    index_frames: list[pd.DataFrame] = []
    membership_frames: list[pd.DataFrame] = []
    ld_qc_rows: list[dict[str, object]] = []
    clump_qc_rows: list[dict[str, object]] = []
    for ancestry in ANCESTRIES:
        for chromosome in range(1, 23):
            ld_qc_path = LD_ROOT / f"qc/ld_chromosomes/{ancestry}.chr{chromosome}.json"
            clump_qc_path = LD_ROOT / f"qc/clump_chromosomes/{ancestry}.chr{chromosome}.json"
            if not ld_qc_path.is_file() or not clump_qc_path.is_file():
                raise RuntimeError(f"Missing chromosome QC: {ancestry} chr{chromosome}")
            ld_qc = json.loads(ld_qc_path.read_text())
            clump_qc = json.loads(clump_qc_path.read_text())
            if not ld_qc.get("ready") or not clump_qc.get("ready"):
                raise RuntimeError(f"Chromosome not ready: {ancestry} chr{chromosome}")
            ld_qc_rows.append(ld_qc)
            clump_qc_rows.append({
                key: value for key, value in clump_qc.items()
                if key not in {"selection_counts"}
            })
            index_path = Path(clump_qc["index_output"])
            membership_path = Path(clump_qc["membership_output"])
            if sha256_file(index_path) != clump_qc["index_output_sha256"]:
                raise RuntimeError(f"Index checksum mismatch: {index_path}")
            if sha256_file(membership_path) != clump_qc["membership_output_sha256"]:
                raise RuntimeError(f"Membership checksum mismatch: {membership_path}")
            index_frame = pd.read_csv(index_path, sep="\t", low_memory=False)
            membership_frame = pd.read_csv(membership_path, sep="\t", low_memory=False)
            if len(index_frame) != int(clump_qc["index_rows"]):
                raise RuntimeError(f"Index row mismatch: {index_path}")
            if len(membership_frame) != int(clump_qc["primary_native_membership_rows"]):
                raise RuntimeError(f"Membership row mismatch: {membership_path}")
            index_frames.append(index_frame)
            membership_frames.append(membership_frame)

    indices = pd.concat(index_frames, ignore_index=True)
    memberships = pd.concat(membership_frames, ignore_index=True)
    if indices["clump_id"].duplicated().any():
        raise RuntimeError("Merged clump IDs are not unique")
    membership_key = ["clump_id", "member_pair_key"]
    if memberships.duplicated(membership_key).any():
        raise RuntimeError("Duplicate pair membership within a primary clump")
    partition_key = ["ancestry", "chromosome", "panel", "method", "member_pair_key"]
    if memberships.duplicated(partition_key).any():
        raise RuntimeError("A pair belongs to more than one primary direct clump")
    index_per_clump = memberships.groupby("clump_id", observed=True)["is_index"].sum()
    if not index_per_clump.eq(1).all():
        raise RuntimeError("Every primary clump must have exactly one index membership")
    nonindex = memberships["is_index"].eq(0)
    if not memberships.loc[nonindex, "member_r2_to_index"].astype(float).between(0.2, 1.0).all():
        raise RuntimeError("Primary non-index memberships violate the direct r2 >= 0.2 rule")
    primary_indices = indices.loc[
        indices["deployment"].eq("native")
        & np.isclose(indices["clump_threshold_r2"].astype(float), 0.2)
    ]
    group_key = ["ancestry", "chromosome", "panel", "method"]
    expected_members = primary_indices.groupby(group_key, observed=True)["clump_member_count"].sum()
    observed_members = memberships.groupby(group_key, observed=True)["member_pair_key"].size()
    accounting = expected_members.to_frame("expected").join(
        observed_members.rename("observed"), how="outer"
    ).fillna(-1)
    if not accounting["expected"].eq(accounting["observed"]).all():
        raise RuntimeError("Primary direct-clump membership accounting failed")
    expected_thresholds = set(np.round(indices["clump_threshold_r2"].astype(float), 1))
    if expected_thresholds != set(THRESHOLDS):
        raise RuntimeError(f"Unexpected clump thresholds: {expected_thresholds}")
    if set(indices["method"].astype(str)) != set(METHODS):
        raise RuntimeError("Not all methods are represented")
    if set(indices["deployment"].astype(str)) != set(DEPLOYMENTS):
        raise RuntimeError("Not all deployments are represented")

    locus_mapping = build_locus_clusters(args.analysis_pairs)
    indices = indices.merge(locus_mapping, on="gene_id", how="left", validate="many_to_one")
    if indices["local_locus_cluster"].isna().any():
        raise RuntimeError("Missing local locus clusters")
    indices = add_score_ranks(indices)

    rank_census = (
        indices.groupby(
            ["ancestry", "panel", "deployment", "method", "clump_threshold_r2"],
            as_index=False,
            observed=True,
        )
        .agg(
            retained_indices=("clump_id", "size"),
            genes=("gene_id", "nunique"),
            exact_stringent_indices=("index_stringent_association", "sum"),
            regional_any_stringent_indices=("regional_any_stringent_association", "sum"),
            median_clump_members=("clump_member_count", "median"),
            median_clump_span_bp=("clump_physical_span_bp", "median"),
            singleton_clump_fraction=("clump_member_count", lambda x: float(np.mean(np.asarray(x) == 1))),
        )
    )
    distribution_rows: list[dict[str, object]] = []
    for keys, group in indices.groupby(
        ["ancestry", "panel", "deployment", "method", "clump_threshold_r2"],
        sort=True, observed=True,
    ):
        ancestry, panel, deployment, method, threshold = keys
        row: dict[str, object] = {
            "ancestry": ancestry, "panel": panel, "deployment": deployment,
            "method": method, "clump_threshold_r2": threshold,
        }
        for column in ("clump_member_count", "clump_physical_span_bp", "close_proxy_count_including_index"):
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            for label, quantile in (("q05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95)):
                row[f"{column}_{label}"] = float(np.quantile(values, quantile))
            row[f"{column}_maximum"] = float(np.max(values))
        distribution_rows.append(row)
    clump_distributions = pd.DataFrame(distribution_rows)

    overlap_rows: list[dict[str, object]] = []
    base_columns = ["ancestry", "panel", "deployment", "clump_threshold_r2"]
    for keys, group in indices.groupby(base_columns, sort=True, observed=True):
        ancestry, panel, deployment, threshold = keys
        sets = {
            method: set(
                group.loc[group["method"].eq(method), "gene_id"].astype(str)
                + "|" + group.loc[group["method"].eq(method), "index_variant_id"].astype(str)
            )
            for method in METHODS
        }
        for method_a, method_b in (("borzoi", "alphagenome"), ("consensus", "borzoi"), ("consensus", "alphagenome")):
            intersection = len(sets[method_a] & sets[method_b])
            union = len(sets[method_a] | sets[method_b])
            overlap_rows.append({
                "ancestry": ancestry, "panel": panel, "deployment": deployment,
                "clump_threshold_r2": threshold,
                "method_a": method_a, "method_b": method_b,
                "indices_a": len(sets[method_a]), "indices_b": len(sets[method_b]),
                "shared_gene_variant_indices": intersection,
                "jaccard_index_overlap": intersection / union if union else float("nan"),
                "fraction_of_a_shared": intersection / len(sets[method_a]) if sets[method_a] else float("nan"),
                "fraction_of_b_shared": intersection / len(sets[method_b]) if sets[method_b] else float("nan"),
            })
    model_index_overlap = pd.DataFrame(overlap_rows)
    ld_qc_table = pd.DataFrame(ld_qc_rows)
    clump_qc_table = pd.DataFrame(clump_qc_rows)
    atomic_table(indices, args.indices_output, gzip=True)
    atomic_table(memberships, args.memberships_output, gzip=True)
    atomic_table(rank_census, LD_ROOT / "qc/native_ld_rank_census.tsv")
    atomic_table(clump_distributions, LD_ROOT / "qc/native_ld_clump_distributions.tsv")
    atomic_table(model_index_overlap, LD_ROOT / "qc/native_ld_model_index_overlap.tsv")
    atomic_table(ld_qc_table, LD_ROOT / "qc/native_ld_chromosome_qc.tsv")
    atomic_table(clump_qc_table, LD_ROOT / "qc/native_clump_chromosome_qc.tsv")
    atomic_table(locus_mapping, LD_ROOT / "manifests/local_locus_clusters.tsv")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chromosome_jobs_validated": int(len(ld_qc_table)),
        "ancestries": sorted(indices["ancestry"].astype(str).unique()),
        "panels": sorted(indices["panel"].astype(str).unique()),
        "deployments": sorted(indices["deployment"].astype(str).unique()),
        "methods": sorted(indices["method"].astype(str).unique()),
        "thresholds": sorted(indices["clump_threshold_r2"].astype(float).unique()),
        "index_rows": int(len(indices)),
        "primary_native_membership_rows": int(len(memberships)),
        "local_locus_clusters": int(locus_mapping["local_locus_cluster"].nunique()),
        "clump_distribution_rows": int(len(clump_distributions)),
        "model_index_overlap_rows": int(len(model_index_overlap)),
        "indices_output": str(args.indices_output),
        "indices_sha256": sha256_file(args.indices_output),
        "memberships_output": str(args.memberships_output),
        "memberships_sha256": sha256_file(args.memberships_output),
        "outcomes_used_to_construct_ranks_or_clumps": False,
        "ready": True,
    }
    atomic_text(args.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
