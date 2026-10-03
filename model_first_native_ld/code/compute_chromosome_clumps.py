#!/usr/bin/env python3
"""Apply direct model-first gene-specific LD clumping on one ancestry/chromosome."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LD_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.1, 0.2, 0.5)
METHODS = ("borzoi", "alphagenome", "consensus")
PANELS = (
    "general", "ad_tier1_excluding_apoe", "ad_tier1_apoe_only", "ad_tier2",
)
DEPLOYMENTS = ("native", "three_way_shared")
STRINGENT_P = 2.02e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_dataframe_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
    temporary.replace(path)


def panel_mask(frame: pd.DataFrame, panel: str) -> pd.Series:
    if panel == "general":
        return frame["general"].eq(1)
    if panel == "ad_tier1_excluding_apoe":
        return frame["ad_tier1"].eq(1) & frame["apoe_locus"].eq(0)
    if panel == "ad_tier1_apoe_only":
        return frame["ad_tier1"].eq(1) & frame["apoe_locus"].eq(1)
    if panel == "ad_tier2":
        return frame["ad_tier2"].eq(1)
    raise ValueError(panel)


def percentile(values: pd.Series, pair_keys: pd.Series) -> pd.Series:
    order = pd.DataFrame({"value": values.astype(float), "pair_key": pair_keys.astype(str)}, index=values.index)
    order = order.sort_values(["value", "pair_key"], kind="mergesort")
    result = pd.Series(index=values.index, dtype=float)
    if len(order) == 1:
        result.loc[order.index] = 1.0
    else:
        result.loc[order.index] = np.arange(len(order), dtype=float) / (len(order) - 1)
    return result


def add_outcome_blind_priorities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["borzoi_priority"] = result["borzoi_score"].abs()
    result["alphagenome_priority"] = result["alphagenome_score"].abs()
    b_pct = pd.Series(index=result.index, dtype=float)
    a_pct = pd.Series(index=result.index, dtype=float)
    for _, gene in result.groupby("gene_id", sort=False):
        b_pct.loc[gene.index] = percentile(gene["borzoi_score"].abs(), gene["pair_key"])
        a_pct.loc[gene.index] = percentile(gene["alphagenome_score"].abs(), gene["pair_key"])
    result["consensus_borzoi_within_gene_percentile"] = b_pct
    result["consensus_alphagenome_within_gene_percentile"] = a_pct
    result["consensus_priority"] = (b_pct + a_pct) / 2.0
    signed_rank = np.sign(result["borzoi_score"]) * b_pct + np.sign(result["alphagenome_score"]) * a_pct
    result["consensus_direction"] = np.sign(signed_rank)
    return result


def load_adjacency(path: Path, valid_ids: set[str]) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for chunk in pd.read_csv(path, sep=r"\s+", compression="gzip", chunksize=500000):
        required = {"SNP_A", "SNP_B", "R2"}
        if not required.issubset(chunk.columns):
            raise RuntimeError(f"Malformed LD columns: {chunk.columns.tolist()}")
        chunk = chunk.loc[chunk["SNP_A"].isin(valid_ids) & chunk["SNP_B"].isin(valid_ids)]
        for left, right, r2 in chunk[["SNP_A", "SNP_B", "R2"]].itertuples(index=False, name=None):
            value = float(r2)
            adjacency[str(left)].append((str(right), value))
            adjacency[str(right)].append((str(left), value))
    return adjacency


def simes(pvalues: list[float]) -> float:
    values = np.sort(np.asarray([p for p in pvalues if np.isfinite(p)], dtype=float))
    if not len(values):
        return float("nan")
    ranks = np.arange(1, len(values) + 1, dtype=float)
    return float(min(1.0, np.min(values * len(values) / ranks)))


def acat(pvalues: list[float]) -> float:
    values = np.asarray([p for p in pvalues if np.isfinite(p)], dtype=float)
    if not len(values):
        return float("nan")
    values = np.clip(values, 1e-15, 1 - 1e-15)
    statistic = float(np.mean(np.tan((0.5 - values) * math.pi)))
    return float(np.clip(0.5 - math.atan(statistic) / math.pi, 0.0, 1.0))


def clump_gene(
    gene: pd.DataFrame,
    adjacency: dict[str, list[tuple[str, float]]],
    method: str,
    threshold: float,
) -> list[tuple[int, list[tuple[str, float]]]]:
    priority_column = f"{method}_priority"
    ordered = gene.sort_values([priority_column, "pair_key"], ascending=[False, True], kind="mergesort")
    row_for_variant = dict(zip(gene["variant_id"].astype(str), gene.index))
    active = set(row_for_variant)
    selected: list[tuple[int, list[tuple[str, float]]]] = []
    for row in ordered.itertuples():
        variant = str(row.variant_id)
        if variant not in active:
            continue
        members = [(variant, 1.0)]
        for neighbor, r2 in adjacency.get(variant, []):
            if neighbor in active and r2 >= threshold:
                members.append((neighbor, r2))
        for member, _ in members:
            active.discard(member)
        selected.append((row_for_variant[variant], members))
    if sum(len(members) for _, members in selected) != len(gene):
        raise RuntimeError("Clump membership does not partition the gene variants")
    return selected


def outcome_fields(
    index: pd.Series,
    gene_by_variant: pd.DataFrame,
    adjacency: dict[str, list[tuple[str, float]]],
    score_direction: float,
) -> dict[str, object]:
    index_variant = str(index["variant_id"])
    proxy_variants = {index_variant}
    for neighbor, r2 in adjacency.get(index_variant, []):
        if r2 >= 0.8 and neighbor in gene_by_variant.index:
            proxy_variants.add(neighbor)
    regional = gene_by_variant.loc[sorted(proxy_variants)]
    pvalues = regional["pvalue"].astype(float).to_list()
    index_p = float(index["pvalue"])
    index_z = float(index["z"])
    exact_assoc = np.isfinite(index_p) and index_p < STRINGENT_P
    exact_directional = exact_assoc and score_direction != 0 and np.sign(index_z) == np.sign(score_direction)
    exact_opposite = exact_assoc and score_direction != 0 and np.sign(index_z) != np.sign(score_direction)
    regional_any = any(np.isfinite(p) and p < STRINGENT_P for p in pvalues)
    if exact_assoc:
        category = "exact_index_supported"
    elif regional_any:
        category = "regional_only_supported"
    else:
        category = "no_detected_regional_support"
    return {
        "index_stringent_association": int(exact_assoc),
        "index_directionally_supported": int(exact_directional),
        "index_stringent_opposite": int(exact_opposite),
        "index_unsupported_or_indeterminate": int(not exact_assoc),
        "index_paper_exact_association": int(exact_assoc and float(index["qvalue"]) < 0.05),
        "aligned_z": float(score_direction * index_z) if score_direction != 0 else float("nan"),
        "close_proxy_count_including_index": int(len(regional)),
        "close_proxy_nonindex_count": int(len(regional) - 1),
        "regional_support_category": category,
        "regional_any_stringent_association": int(regional_any),
        "regional_min_p": float(np.nanmin(pvalues)) if np.isfinite(pvalues).any() else float("nan"),
        "regional_simes_p": simes(pvalues),
        "regional_acat_p": acat(pvalues),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancestry", choices=("AA", "CH", "NHW"), required=True)
    parser.add_argument("--chromosome", type=int, choices=range(1, 23), required=True)
    args = parser.parse_args()

    ancestry = args.ancestry
    chromosome = args.chromosome
    started = datetime.now(timezone.utc)
    pair_path = LD_ROOT / f"manifests/pairs/{ancestry}.chr{chromosome}.pairs.tsv.gz"
    coverage_path = LD_ROOT / f"qc/ld_chromosomes/{ancestry}.chr{chromosome}.variant_coverage.tsv.gz"
    ld_path = LD_ROOT / f"work/ld/{ancestry}.chr{chromosome}.ld.gz"
    ld_qc_path = LD_ROOT / f"qc/ld_chromosomes/{ancestry}.chr{chromosome}.json"
    ld_qc = json.loads(ld_qc_path.read_text())
    if not ld_qc.get("ready"):
        raise RuntimeError(f"LD QC not ready: {ld_qc_path}")
    pairs = pd.read_csv(pair_path, sep="\t", low_memory=False)
    coverage = pd.read_csv(coverage_path, sep="\t")
    valid_ids = set(coverage.loc[coverage["ld_calculable"].eq(1), "variant_id"].astype(str))
    pairs = pairs.loc[pairs["variant_id"].isin(valid_ids)].copy()
    if pairs["pair_key"].duplicated().any():
        raise RuntimeError("Duplicate pair keys in chromosome manifest")
    adjacency = load_adjacency(ld_path, valid_ids)

    index_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    selection_counts: dict[str, int] = {}
    for panel in PANELS:
        panel_pairs = pairs.loc[panel_mask(pairs, panel)].copy()
        if panel_pairs.empty:
            continue
        for deployment in DEPLOYMENTS:
            deployed = panel_pairs if deployment == "native" else panel_pairs.loc[panel_pairs["three_way_shared"].eq(1)]
            if deployed.empty:
                continue
            deployed = add_outcome_blind_priorities(deployed)
            for gene_id, gene in deployed.groupby("gene_id", sort=True):
                gene = gene.copy()
                gene_by_variant = gene.set_index("variant_id", drop=False)
                if gene_by_variant.index.duplicated().any():
                    raise RuntimeError(f"Duplicate variant within gene {gene_id}")
                for method in METHODS:
                    for threshold in THRESHOLDS:
                        clumps = clump_gene(gene, adjacency, method, threshold)
                        key = f"{panel}|{deployment}|{method}|{threshold:.1f}"
                        selection_counts[key] = selection_counts.get(key, 0) + len(clumps)
                        for ordinal, (index_row_id, members) in enumerate(clumps, start=1):
                            index = gene.loc[index_row_id]
                            if method == "borzoi":
                                score = float(index["borzoi_score"])
                                direction = float(np.sign(score))
                            elif method == "alphagenome":
                                score = float(index["alphagenome_score"])
                                direction = float(np.sign(score))
                            else:
                                direction = float(index["consensus_direction"])
                                score = direction * float(index["consensus_priority"])
                            member_positions = [int(gene_by_variant.loc[variant, "position"]) for variant, _ in members]
                            record = {
                                "ancestry": ancestry,
                                "chromosome": chromosome,
                                "panel": panel,
                                "deployment": deployment,
                                "method": method,
                                "clump_threshold_r2": threshold,
                                "gene_id": gene_id,
                                "clump_ordinal": ordinal,
                                "clump_id": f"{ancestry}|{panel}|{deployment}|{method}|r2_{threshold:.1f}|{gene_id}|{ordinal}",
                                "index_pair_key": index["pair_key"],
                                "index_variant_id": index["variant_id"],
                                "index_position": int(index["position"]),
                                "tss": int(index["tss"]),
                                "abs_distance_to_tss": int(index["abs_distance_to_tss"]),
                                "score": score,
                                "score_direction": direction,
                                "priority_value_preclump": float(index[f"{method}_priority"]),
                                "borzoi_score": float(index["borzoi_score"]),
                                "alphagenome_score": float(index["alphagenome_score"]),
                                "consensus_borzoi_within_gene_percentile": float(index["consensus_borzoi_within_gene_percentile"]),
                                "consensus_alphagenome_within_gene_percentile": float(index["consensus_alphagenome_within_gene_percentile"]),
                                "pvalue": float(index["pvalue"]),
                                "z": float(index["z"]),
                                "qvalue": float(index["qvalue"]),
                                "maf": float(index["maf"]),
                                "general_sampling_stratum": index["general_sampling_stratum"],
                                "general_inverse_probability_weight": float(index["general_inverse_probability_weight"]),
                                "clump_member_count": len(members),
                                "clump_physical_span_bp": max(member_positions) - min(member_positions),
                            }
                            record.update(outcome_fields(index, gene_by_variant, adjacency, direction))
                            index_rows.append(record)
                            if deployment == "native" and threshold == 0.2:
                                for member_variant, member_r2 in members:
                                    member = gene_by_variant.loc[member_variant]
                                    membership_rows.append({
                                        "ancestry": ancestry,
                                        "chromosome": chromosome,
                                        "panel": panel,
                                        "method": method,
                                        "gene_id": gene_id,
                                        "clump_id": record["clump_id"],
                                        "index_pair_key": record["index_pair_key"],
                                        "member_pair_key": member["pair_key"],
                                        "member_variant_id": member_variant,
                                        "member_r2_to_index": member_r2,
                                        "is_index": int(member_variant == index["variant_id"]),
                                    })

    indices = pd.DataFrame(index_rows)
    memberships = pd.DataFrame(membership_rows)
    output_dir = LD_ROOT / "work/clumps"
    index_path = output_dir / f"{ancestry}.chr{chromosome}.indices.tsv.gz"
    membership_path = output_dir / f"{ancestry}.chr{chromosome}.primary_memberships.tsv.gz"
    qc_path = LD_ROOT / f"qc/clump_chromosomes/{ancestry}.chr{chromosome}.json"
    atomic_dataframe_gzip(indices, index_path)
    atomic_dataframe_gzip(memberships, membership_path)
    completed = datetime.now(timezone.utc)
    qc = {
        "created_at": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "ancestry": ancestry,
        "chromosome": chromosome,
        "input_native_pairs_with_ld": int(len(pairs)),
        "input_native_genes": int(pairs["gene_id"].nunique()),
        "sparse_variants_with_neighbors": int(len(adjacency)),
        "index_rows": int(len(indices)),
        "primary_native_membership_rows": int(len(memberships)),
        "selection_counts": selection_counts,
        "index_output": str(index_path),
        "index_output_sha256": sha256_file(index_path),
        "membership_output": str(membership_path),
        "membership_output_sha256": sha256_file(membership_path),
        "outcome_fields_used_for_selection": False,
        "ready": bool(len(indices) > 0 and len(memberships) > 0),
    }
    atomic_text(qc_path, json.dumps(qc, indent=2, sort_keys=True) + "\n")
    if not qc["ready"]:
        raise RuntimeError(f"Clump QC failed: {qc_path}")
    print(json.dumps(qc, sort_keys=True))


if __name__ == "__main__":
    main()
