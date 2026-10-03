#!/usr/bin/env python3
"""Gene-fixed-effect adjusted top-versus-rest models after primary LD clumping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


LD_ROOT = Path(__file__).resolve().parents[1]


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


def group_weighted_center(values: np.ndarray, codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_groups = int(codes.max()) + 1
    denominator = np.bincount(codes, weights=weights, minlength=n_groups)
    numerator = np.bincount(codes, weights=weights * values, minlength=n_groups)
    means = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return values - means[codes]


def fit_cluster_robust(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xtwx = x.T @ (weights[:, None] * x)
    rank = np.linalg.matrix_rank(xtwx)
    if rank != x.shape[1]:
        raise RuntimeError(f"Rank-deficient adjusted design: {rank}/{x.shape[1]}")
    bread = np.linalg.inv(xtwx)
    beta = bread @ (x.T @ (weights * y))
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ (weights[mask] * residual[mask])
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    n, p, g = len(y), x.shape[1], len(unique_clusters)
    if g > 1 and n > p:
        covariance *= (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    return beta, (covariance + covariance.T) / 2


def holm(pvalues: np.ndarray) -> np.ndarray:
    result = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if not len(valid):
        return result
    ordered = valid[np.argsort(pvalues[valid], kind="mergesort")]
    adjusted = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * pvalues[ordered])
    result[ordered] = np.minimum(1.0, adjusted)
    return result


def fit_one(data: pd.DataFrame, rank_column: str, evidence: np.ndarray) -> dict[str, object]:
    weights = pd.to_numeric(data["general_inverse_probability_weight"], errors="coerce").to_numpy(float)
    maf = pd.to_numeric(data["maf"], errors="coerce").to_numpy(float)
    distance = np.log1p(pd.to_numeric(data["abs_distance_to_tss"], errors="coerce").to_numpy(float))
    top = pd.to_numeric(data[rank_column], errors="coerce").to_numpy(float) < 0.01
    valid = (
        np.isfinite(evidence) & np.isfinite(weights) & (weights > 0)
        & np.isfinite(maf) & np.isfinite(distance)
    )
    y = evidence[valid]
    w = weights[valid]
    top_valid = top[valid].astype(float)
    gene = data.loc[valid, "gene_id"].astype(str).to_numpy()
    codes, unique_genes = pd.factorize(gene, sort=True)
    maf_centered = maf[valid] - np.average(maf[valid], weights=w)
    distance_centered = distance[valid] - np.average(distance[valid], weights=w)
    raw_x = np.column_stack([
        top_valid,
        distance_centered,
        distance_centered ** 2,
        maf_centered,
        maf_centered ** 2,
    ])
    x = np.column_stack([
        group_weighted_center(raw_x[:, column], codes, w)
        for column in range(raw_x.shape[1])
    ])
    y = group_weighted_center(y, codes, w)
    beta, covariance = fit_cluster_robust(x, y, w, codes)
    se = float(np.sqrt(max(0.0, covariance[0, 0])))
    statistic = float(beta[0] / se) if se > 0 else float("nan")
    degrees = len(unique_genes) - 1
    pvalue = float(2 * student_t.sf(abs(statistic), df=degrees)) if degrees > 0 else float("nan")
    critical = float(student_t.ppf(0.975, df=degrees)) if degrees > 0 else float("nan")
    return {
        "observations": int(valid.sum()),
        "genes": int(len(unique_genes)),
        "top_indices": int(top_valid.sum()),
        "rest_indices": int(len(top_valid) - top_valid.sum()),
        "top_indicator_estimate": float(beta[0]),
        "gene_clustered_se": se,
        "t_statistic": statistic,
        "degrees_of_freedom": degrees,
        "pvalue": pvalue,
        "ci_low": float(beta[0] - critical * se),
        "ci_high": float(beta[0] + critical * se),
        "design_rank": int(np.linalg.matrix_rank(x)),
        "design_condition_number": float(np.linalg.cond(x.T @ (w[:, None] * x))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, default=LD_ROOT / "results/native_ld_indices.tsv.gz")
    parser.add_argument("--output", type=Path, default=LD_ROOT / "results/native_ld_adjusted_primary.tsv")
    parser.add_argument("--summary", type=Path, default=LD_ROOT / "qc/native_ld_adjusted_summary.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.indices, sep="\t", low_memory=False)
    frame = frame.loc[
        frame["panel"].eq("general")
        & np.isclose(frame["clump_threshold_r2"].astype(float), 0.2)
    ].copy()
    rows: list[dict[str, object]] = []
    for (deployment, ancestry, method), data in frame.groupby(
        ["deployment", "ancestry", "method"], sort=True, observed=True
    ):
        z = pd.to_numeric(data["z"], errors="coerce").to_numpy(float)
        aligned = pd.to_numeric(data["aligned_z"], errors="coerce").to_numpy(float)
        for rank_view, rank_column in (
            ("global", "global_top_tail_fraction"),
            ("within_gene", "within_gene_top_tail_fraction"),
        ):
            for evidence_type, evidence in (("association", np.abs(z)), ("directional", aligned)):
                rows.append({
                    "panel": "general", "deployment": deployment,
                    "ancestry": ancestry, "method": method,
                    "clump_threshold_r2": 0.2, "rank_view": rank_view,
                    "cutoff_percent": 1.0, "evidence_type": evidence_type,
                    "outcome": "absolute_eqtl_z" if evidence_type == "association" else "aligned_eqtl_z",
                    "formula": "outcome ~ top_1pct + gene fixed effects + centered log1p distance + squared term + centered ancestry-specific MAF + squared term",
                    **fit_one(data, rank_column, evidence),
                })
    result = pd.DataFrame(rows)
    result["holm_pvalue_within_9_test_family"] = np.nan
    for (deployment, rank_view, evidence_type), group in result.groupby(
        ["deployment", "rank_view", "evidence_type"], sort=True
    ):
        if len(group) != 9:
            raise RuntimeError(
                f"Expected 9 ancestry×model tests, found {len(group)} for "
                f"{deployment}/{rank_view}/{evidence_type}"
            )
        result.loc[group.index, "holm_pvalue_within_9_test_family"] = holm(
            group["pvalue"].to_numpy(float)
        )
    result.sort_values(
        ["deployment", "rank_view", "evidence_type", "ancestry", "method"],
        inplace=True, kind="mergesort",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    result.to_csv(temporary, sep="\t", index=False, na_rep="NA")
    temporary.replace(args.output)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(result)),
        "expected_rows": 72,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "gene_fixed_effect": True,
        "covariates": ["log1p_abs_tss_distance", "quadratic_distance", "ancestry_specific_maf", "quadratic_maf"],
        "gene_clustered_covariance": True,
        "holm_family": "9 ancestry-by-method tests within deployment, rank view, and evidence type",
        "ready": bool(len(result) == 72),
    }
    atomic_text(args.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["ready"]:
        raise RuntimeError(f"Adjusted result row count mismatch: {len(result)}")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
