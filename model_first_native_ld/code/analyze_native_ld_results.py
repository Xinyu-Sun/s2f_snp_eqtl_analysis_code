#!/usr/bin/env python3
"""Analyze top-versus-rest and regional support after native LD clumping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm


LD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LD_ROOT.parent
CUTOFFS = (0.005, 0.01, 0.02, 0.05, 0.10)
PRIMARY_THRESHOLD = 0.2
PRIMARY_CUTOFF = 0.01
METHOD_COMPARISONS = (
    ("borzoi", "alphagenome"),
    ("consensus", "borzoi"),
    ("consensus", "alphagenome"),
)
ENDPOINTS = {
    "index_stringent_association": "index_stringent_association",
    "index_directionally_supported": "index_directionally_supported",
    "index_stringent_opposite": "index_stringent_opposite",
    "index_stringent_direction_unresolved": "index_stringent_direction_unresolved",
    "index_paper_exact_association": "index_paper_exact_association",
    "regional_any_stringent_association": "regional_any_stringent_association",
    "regional_only_supported": "regional_only_supported",
    "no_detected_regional_support": "no_detected_regional_support",
}


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


def derived_seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{label}".encode()).digest()[:8], "big") % (2 ** 32)


def bootstrap_inference(point: float, replicates: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(replicates, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return {
            "ci_low": float("nan"), "bootstrap_median": float("nan"),
            "ci_high": float("nan"), "bootstrap_se": float("nan"),
            "bootstrap_normal_pvalue": float("nan"),
            "bootstrap_sign_pvalue": float("nan"),
            "finite_bootstrap_replicates": int(len(finite)),
        }
    low, median, high = np.quantile(finite, [0.025, 0.5, 0.975])
    se = float(np.std(finite, ddof=1))
    normal_p = float(2 * norm.sf(abs(point / se))) if se > 0 else float(point == 0)
    below = int(np.sum(finite <= 0))
    above = int(np.sum(finite >= 0))
    sign_p = min(1.0, 2.0 * min((below + 1) / (len(finite) + 1), (above + 1) / (len(finite) + 1)))
    return {
        "ci_low": float(low), "bootstrap_median": float(median),
        "ci_high": float(high), "bootstrap_se": se,
        "bootstrap_normal_pvalue": normal_p,
        "bootstrap_sign_pvalue": float(sign_p),
        "finite_bootstrap_replicates": int(len(finite)),
    }


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    result = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if not len(finite):
        return result
    ordered = finite[np.argsort(pvalues[finite], kind="mergesort")]
    adjusted = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * pvalues[ordered])
    result[ordered] = np.minimum(1.0, adjusted)
    return result


def cluster_sums(codes: np.ndarray, values: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.bincount(codes, weights=values.astype(float), minlength=n_clusters)


def bootstrap_draws(
    universe: pd.DataFrame,
    cluster_column: str,
    stratum_column: str | None,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    columns = [cluster_column] + ([stratum_column] if stratum_column else [])
    meta = universe[columns].drop_duplicates().copy()
    if meta[cluster_column].duplicated().any():
        raise RuntimeError(f"Bootstrap cluster crosses strata: {cluster_column}")
    meta.sort_values(cluster_column, inplace=True, kind="mergesort")
    labels = meta[cluster_column].astype(str).to_numpy()
    lookup = {label: index for index, label in enumerate(labels)}
    rng = np.random.default_rng(seed)
    multiplicity = np.zeros((replicates, len(labels)), dtype=np.int16)
    if len(labels) <= 1:
        return multiplicity, lookup
    if stratum_column is None:
        multiplicity[:] = rng.multinomial(len(labels), np.repeat(1 / len(labels), len(labels)), size=replicates)
    else:
        strata = meta[stratum_column].astype(str).to_numpy()
        for stratum in sorted(set(strata)):
            positions = np.flatnonzero(strata == stratum)
            multiplicity[:, positions] = rng.multinomial(
                len(positions), np.repeat(1 / len(positions), len(positions)), size=replicates
            )
    return multiplicity, lookup


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    use = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not use.any():
        return float("nan")
    order = np.argsort(values[use], kind="mergesort")
    sorted_values = values[use][order]
    cumulative = np.cumsum(weights[use][order])
    return float(sorted_values[min(np.searchsorted(cumulative, probability * cumulative[-1]), len(sorted_values) - 1)])


def weighted_lower_auc(pvalues: np.ndarray, selected: np.ndarray, weights: np.ndarray) -> float:
    use = np.isfinite(pvalues) & np.isfinite(weights) & (weights > 0)
    values = pvalues[use]
    top = selected[use]
    weight = weights[use]
    top_total = float(weight[top].sum())
    rest_total = float(weight[~top].sum())
    if top_total <= 0 or rest_total <= 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    top_weight = weight[order] * top[order]
    rest_weight = weight[order] * (~top[order])
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    top_at = np.add.reduceat(top_weight, starts)
    rest_at = np.add.reduceat(rest_weight, starts)
    rest_before = np.cumsum(rest_at) - rest_at
    rest_greater = rest_total - rest_before - rest_at
    return float(np.sum(top_at * (rest_greater + 0.5 * rest_at)) / (top_total * rest_total))


def make_gene_universes(general_path: Path, ad_path: Path, locus_path: Path) -> dict[str, pd.DataFrame]:
    locus = pd.read_csv(
        locus_path, sep="\t", usecols=["gene_id", "local_locus_cluster"],
        dtype={"gene_id": str},
    )
    general = pd.read_csv(
        general_path, sep="\t",
        usecols=["gene_id", "chromosome", "union_pair_count_quintile"],
        dtype={"gene_id": str, "chromosome": str, "union_pair_count_quintile": str},
    )
    general["general_sampling_stratum"] = (
        "chr" + general["chromosome"] + "_q" + general["union_pair_count_quintile"]
    )
    ad = pd.read_csv(
        ad_path, sep="\t",
        usecols=["gene_id", "tier", "locus_numbers", "apoe_locus"],
        dtype={"gene_id": str, "tier": str, "locus_numbers": str},
    )
    ad["apoe_locus"] = pd.to_numeric(ad["apoe_locus"], errors="raise").astype(int)
    ad["ad_compound_locus_cluster"] = [
        f"{tier}:" + "+".join(sorted(set(numbers.split(",")), key=int))
        for tier, numbers in zip(ad["tier"], ad["locus_numbers"], strict=True)
    ]
    universes = {
        "general": general.merge(locus, on="gene_id", validate="one_to_one"),
        "ad_tier1_excluding_apoe": ad.loc[ad["tier"].eq("Tier1") & ad["apoe_locus"].eq(0)].merge(locus, on="gene_id", validate="one_to_one"),
        "ad_tier1_apoe_only": ad.loc[ad["tier"].eq("Tier1") & ad["apoe_locus"].eq(1)].merge(locus, on="gene_id", validate="one_to_one"),
        "ad_tier2": ad.loc[ad["tier"].eq("Tier2")].merge(locus, on="gene_id", validate="one_to_one"),
    }
    if any(frame.empty for frame in universes.values()):
        raise RuntimeError("Empty analysis-unit universe")
    return universes


def point_mean_difference(values: np.ndarray, selected: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    top = valid & selected
    rest = valid & ~selected
    if not top.any() or not rest.any():
        return float("nan"), float("nan"), float("nan")
    top_mean = float(np.average(values[top], weights=weights[top]))
    rest_mean = float(np.average(values[rest], weights=weights[rest]))
    return top_mean, rest_mean, top_mean - rest_mean


def bootstrap_ratio_metrics(
    data: pd.DataFrame,
    values: np.ndarray,
    selected: np.ndarray,
    weights: np.ndarray,
    cluster_column: str,
    multiplicity: np.ndarray,
    lookup: dict[str, int],
    binary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = data[cluster_column].astype(str).map(lookup)
    if codes.isna().any():
        raise RuntimeError(f"Observed cluster absent from universe: {cluster_column}")
    codes_array = codes.to_numpy(np.int32)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    top = valid & selected
    rest = valid & ~selected
    full = valid
    n_clusters = len(lookup)
    matrices = []
    for mask in (top, rest, full):
        matrices.append(cluster_sums(codes_array, weights * mask, n_clusters))
        matrices.append(cluster_sums(codes_array, weights * np.where(mask, values, 0.0), n_clusters))
    totals = multiplicity.astype(float) @ np.column_stack(matrices)
    with np.errstate(divide="ignore", invalid="ignore"):
        top_est = totals[:, 1] / totals[:, 0]
        rest_est = totals[:, 3] / totals[:, 2]
        full_est = totals[:, 5] / totals[:, 4]
        difference = top_est - rest_est
        lift = top_est / full_est if binary else np.full(len(top_est), np.nan)
    return top_est, difference, lift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, default=LD_ROOT / "results/native_ld_indices.tsv.gz")
    parser.add_argument("--general-genes", type=Path, default=PROJECT_ROOT / "manifests/general_benchmark_analysis_weights.tsv")
    parser.add_argument("--ad-genes", type=Path, default=PROJECT_ROOT / "manifests/ad_application_genes.tsv")
    parser.add_argument("--locus-map", type=Path, default=LD_ROOT / "manifests/local_locus_clusters.tsv")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--summary", type=Path, default=LD_ROOT / "qc/native_ld_analysis_summary.json")
    args = parser.parse_args()
    if args.replicates < 1000:
        raise ValueError("The locked analysis requires at least 1,000 bootstrap replicates")

    frame = pd.read_csv(args.indices, sep="\t", low_memory=False)
    if frame["clump_id"].duplicated().any():
        raise RuntimeError("Duplicate clump IDs")
    frame["regional_only_supported"] = (
        frame["regional_support_category"].eq("regional_only_supported")
    ).astype(int)
    frame["index_stringent_direction_unresolved"] = (
        frame["index_stringent_association"].eq(1)
        & np.isclose(pd.to_numeric(frame["score_direction"], errors="coerce"), 0.0)
    ).astype(int)
    frame["no_detected_regional_support"] = (
        frame["regional_support_category"].eq("no_detected_regional_support")
    ).astype(int)
    universes = make_gene_universes(args.general_genes, args.ad_genes, args.locus_map)

    yield_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    regional_pvalue_rows: list[dict[str, object]] = []
    regional_stratum_rows: list[dict[str, object]] = []
    ecdf_rows: list[dict[str, object]] = []
    replicate_cache: dict[tuple, np.ndarray] = {}
    point_cache: dict[tuple, float] = {}
    bootstrap_group_rows: list[dict[str, object]] = []

    group_columns = ["ancestry", "panel", "deployment", "method", "clump_threshold_r2"]
    for group_key, data in frame.groupby(group_columns, sort=True, observed=True):
        ancestry, panel, deployment, method, threshold = group_key
        data = data.copy()
        weights = (
            pd.to_numeric(data["general_inverse_probability_weight"], errors="raise").to_numpy(float)
            if panel == "general" else np.ones(len(data), dtype=float)
        )
        if panel == "general" and (not np.isfinite(weights).all() or (weights <= 0).any()):
            raise RuntimeError("Invalid general-benchmark weights")
        bootstrap_specs = []
        if math.isclose(float(threshold), PRIMARY_THRESHOLD):
            cluster_definitions = (
                (
                    ("gene", "gene_id", "general_sampling_stratum"),
                    ("local_locus", "local_locus_cluster", None),
                )
                if panel == "general" else
                (
                    ("gene", "gene_id", None),
                    ("published_ad_locus", "ad_compound_locus_cluster", None),
                )
            )
            for cluster_type, cluster_column, stratum_column in cluster_definitions:
                universe = universes[str(panel)]
                draw_key = (str(panel), str(deployment), cluster_type)
                if draw_key not in replicate_cache:
                    multiplicity, lookup = bootstrap_draws(
                        universe, cluster_column, stratum_column, args.replicates,
                        derived_seed(args.seed, "|".join(draw_key)),
                    )
                    replicate_cache[draw_key] = multiplicity
                    replicate_cache[draw_key + ("lookup",)] = lookup  # type: ignore[assignment]
                    bootstrap_group_rows.append({
                        "panel": panel, "deployment": deployment,
                        "cluster_type": cluster_type, "clusters": len(lookup),
                        "replicates": args.replicates if len(lookup) > 1 else 0,
                    })
                bootstrap_specs.append((
                    cluster_type, cluster_column,
                    replicate_cache[draw_key], replicate_cache[draw_key + ("lookup",)],
                ))

        evidence = {
            "association": (
                pd.to_numeric(data["pvalue"], errors="coerce").to_numpy(float),
                np.abs(pd.to_numeric(data["z"], errors="coerce").to_numpy(float)),
            ),
            "directional": (
                ndtr(-pd.to_numeric(data["aligned_z"], errors="coerce").to_numpy(float)),
                pd.to_numeric(data["aligned_z"], errors="coerce").to_numpy(float),
            ),
        }
        for rank_view, tail_column in (
            ("global", "global_top_tail_fraction"),
            ("within_gene", "within_gene_top_tail_fraction"),
        ):
            top_tail = pd.to_numeric(data[tail_column], errors="raise").to_numpy(float)
            for cutoff in CUTOFFS:
                selected = top_tail < cutoff
                if not selected.any() or selected.all():
                    raise RuntimeError(f"Degenerate top/rest selection: {group_key} {rank_view} {cutoff}")
                base = {
                    "ancestry": ancestry, "panel": panel, "deployment": deployment,
                    "method": method, "clump_threshold_r2": float(threshold),
                    "rank_view": rank_view, "cutoff_fraction": cutoff,
                    "cutoff_percent": 100 * cutoff, "retained_indices": len(data),
                    "selected_indices": int(selected.sum()),
                    "selected_weighted_indices": float(weights[selected].sum()),
                    "background_weighted_indices": float(weights.sum()),
                }
                for evidence_type, (pvalues, evidence_values) in evidence.items():
                    valid = np.isfinite(pvalues) & np.isfinite(evidence_values)
                    top_mean, rest_mean, difference = point_mean_difference(
                        evidence_values, selected & valid, weights
                    )
                    distribution = {
                        **base, "evidence_type": evidence_type,
                        "valid_indices": int(valid.sum()),
                        "top_valid_indices": int((selected & valid).sum()),
                        "rest_valid_indices": int((~selected & valid).sum()),
                        "top_mean_evidence": top_mean, "rest_mean_evidence": rest_mean,
                        "top_minus_rest_mean_evidence": difference,
                        "auc_probability_top_has_smaller_p": weighted_lower_auc(pvalues, selected, weights),
                        "top_p_q01": weighted_quantile(pvalues[selected], weights[selected], 0.01),
                        "top_p_q05": weighted_quantile(pvalues[selected], weights[selected], 0.05),
                        "top_p_median": weighted_quantile(pvalues[selected], weights[selected], 0.50),
                        "rest_p_q01": weighted_quantile(pvalues[~selected], weights[~selected], 0.01),
                        "rest_p_q05": weighted_quantile(pvalues[~selected], weights[~selected], 0.05),
                        "rest_p_median": weighted_quantile(pvalues[~selected], weights[~selected], 0.50),
                    }
                    for cluster_type, cluster_column, multiplicity, lookup in bootstrap_specs:
                        _, difference_rep, _ = bootstrap_ratio_metrics(
                            data, evidence_values, selected & valid, weights,
                            cluster_column, multiplicity, lookup, binary=False,
                        )
                        inference = bootstrap_inference(difference, difference_rep)
                        for key, value in inference.items():
                            distribution[f"{cluster_type}_{key}"] = value
                        cache_key = (
                            "distribution", ancestry, panel, deployment, method,
                            rank_view, cutoff, evidence_type, cluster_type,
                        )
                        replicate_cache[cache_key] = difference_rep
                        point_cache[cache_key] = difference
                    distribution_rows.append(distribution)

                    if (
                        math.isclose(float(threshold), PRIMARY_THRESHOLD)
                        and math.isclose(cutoff, PRIMARY_CUTOFF)
                        and panel == "general"
                    ):
                        grid = np.linspace(0, 8, 161)
                        thresholds = np.power(10.0, -grid)
                        for label, mask in (("top_1_percent", selected & valid), ("rest_99_percent", ~selected & valid)):
                            denominator = float(weights[mask].sum())
                            for logp, pthreshold in zip(grid, thresholds, strict=True):
                                ecdf_rows.append({
                                    "ancestry": ancestry, "deployment": deployment,
                                    "method": method, "rank_view": rank_view,
                                    "evidence_type": evidence_type, "group": label,
                                    "minus_log10_p_threshold": logp,
                                    "pvalue_threshold": pthreshold,
                                    "weighted_cumulative_fraction": float(
                                        weights[mask & (pvalues <= pthreshold)].sum() / denominator
                                    ),
                                })

                regional_pvalues = {
                    "minimum_p": pd.to_numeric(data["regional_min_p"], errors="coerce").to_numpy(float),
                    "simes_p": pd.to_numeric(data["regional_simes_p"], errors="coerce").to_numpy(float),
                    "acat_p": pd.to_numeric(data["regional_acat_p"], errors="coerce").to_numpy(float),
                }
                for statistic, pvalues in regional_pvalues.items():
                    regional_pvalue_rows.append({
                        **base,
                        "regional_statistic": statistic,
                        "auc_probability_top_has_smaller_p": weighted_lower_auc(pvalues, selected, weights),
                        "top_p_q01": weighted_quantile(pvalues[selected], weights[selected], 0.01),
                        "top_p_q05": weighted_quantile(pvalues[selected], weights[selected], 0.05),
                        "top_p_median": weighted_quantile(pvalues[selected], weights[selected], 0.50),
                        "rest_p_q01": weighted_quantile(pvalues[~selected], weights[~selected], 0.01),
                        "rest_p_q05": weighted_quantile(pvalues[~selected], weights[~selected], 0.05),
                        "rest_p_median": weighted_quantile(pvalues[~selected], weights[~selected], 0.50),
                    })
                proxy_count = pd.to_numeric(
                    data["close_proxy_count_including_index"], errors="raise"
                ).to_numpy(int)
                regional_event = pd.to_numeric(
                    data["regional_any_stringent_association"], errors="raise"
                ).to_numpy(float)
                proxy_strata = (
                    ("1_singleton", proxy_count == 1),
                    ("2", proxy_count == 2),
                    ("3_to_5", (proxy_count >= 3) & (proxy_count <= 5)),
                    ("6_plus", proxy_count >= 6),
                )
                for proxy_stratum, stratum_mask in proxy_strata:
                    for comparison_group, mask in (
                        ("top", stratum_mask & selected),
                        ("rest", stratum_mask & ~selected),
                        ("background", stratum_mask),
                    ):
                        denominator = float(weights[mask].sum())
                        regional_stratum_rows.append({
                            **base,
                            "proxy_count_stratum": proxy_stratum,
                            "comparison_group": comparison_group,
                            "raw_indices": int(mask.sum()),
                            "weighted_indices": denominator,
                            "regional_stringent_events": int(regional_event[mask].sum()),
                            "weighted_regional_stringent_events": float(
                                (weights[mask] * regional_event[mask]).sum()
                            ),
                            "regional_stringent_yield": float(
                                (weights[mask] * regional_event[mask]).sum() / denominator
                            ) if denominator > 0 else float("nan"),
                        })

                for endpoint, value_column in ENDPOINTS.items():
                    values = pd.to_numeric(data[value_column], errors="raise").to_numpy(float)
                    top_yield = float(np.average(values[selected], weights=weights[selected]))
                    background_yield = float(np.average(values, weights=weights))
                    rest_yield = float(np.average(values[~selected], weights=weights[~selected]))
                    row = {
                        **base, "endpoint": endpoint,
                        "selected_event_count": int(values[selected].sum()),
                        "background_event_count": int(values.sum()),
                        "selected_weighted_event_count": float((weights[selected] * values[selected]).sum()),
                        "background_weighted_event_count": float((weights * values).sum()),
                        "selected_yield": top_yield,
                        "rest_yield": rest_yield,
                        "background_yield": background_yield,
                        "top_minus_rest_yield": top_yield - rest_yield,
                        "lift_over_background": top_yield / background_yield if background_yield > 0 else float("nan"),
                    }
                    for cluster_type, cluster_column, multiplicity, lookup in bootstrap_specs:
                        top_rep, difference_rep, lift_rep = bootstrap_ratio_metrics(
                            data, values, selected, weights, cluster_column,
                            multiplicity, lookup, binary=True,
                        )
                        for metric, point, reps in (
                            ("selected_yield", top_yield, top_rep),
                            ("top_minus_rest_yield", top_yield - rest_yield, difference_rep),
                            ("lift_over_background", row["lift_over_background"], lift_rep),
                        ):
                            inference = bootstrap_inference(float(point), reps)
                            for key, value in inference.items():
                                row[f"{cluster_type}_{metric}_{key}"] = value
                            cache_key = (
                                metric, ancestry, panel, deployment, method,
                                rank_view, cutoff, endpoint, cluster_type,
                            )
                            replicate_cache[cache_key] = reps
                            point_cache[cache_key] = float(point)
                    yield_rows.append(row)

    yields = pd.DataFrame(yield_rows)
    distributions = pd.DataFrame(distribution_rows)
    regional_pvalues = pd.DataFrame(regional_pvalue_rows)
    regional_strata = pd.DataFrame(regional_stratum_rows)
    ecdf = pd.DataFrame(ecdf_rows)
    bootstrap_groups = pd.DataFrame(bootstrap_group_rows).drop_duplicates()

    comparison_rows: list[dict[str, object]] = []
    cluster_types = sorted({key[-1] for key in point_cache})
    for cluster_type in cluster_types:
        for metric, endpoint_type in (
            ("selected_yield", "yield"),
            ("top_minus_rest_yield", "yield"),
            ("lift_over_background", "yield"),
            ("distribution", "distribution"),
        ):
            keys = [key for key in point_cache if key[0] == metric and key[-1] == cluster_type]
            for key in keys:
                if endpoint_type == "yield":
                    _, ancestry, panel, deployment, method, rank_view, cutoff, endpoint, _ = key
                else:
                    _, ancestry, panel, deployment, method, rank_view, cutoff, endpoint, _ = key
                for method_a, method_b in METHOD_COMPARISONS:
                    if method != method_a:
                        continue
                    key_b = (metric, ancestry, panel, deployment, method_b, rank_view, cutoff, endpoint, cluster_type)
                    if key_b not in point_cache:
                        continue
                    point = point_cache[key] - point_cache[key_b]
                    reps = replicate_cache[key] - replicate_cache[key_b]
                    comparison_rows.append({
                        "metric": metric, "evidence_or_endpoint": endpoint,
                        "ancestry": ancestry, "panel": panel, "deployment": deployment,
                        "rank_view": rank_view, "cutoff_percent": 100 * cutoff,
                        "method_a": method_a, "method_b": method_b,
                        "cluster_type": cluster_type,
                        "difference_a_minus_b": point,
                        **bootstrap_inference(point, reps),
                    })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["holm_pvalue_within_9_test_family"] = np.nan
    comparison_family = [
        "panel", "deployment", "rank_view", "cutoff_percent", "metric",
        "evidence_or_endpoint", "cluster_type",
    ]
    for _, group in comparisons.groupby(comparison_family, sort=True, observed=True):
        comparisons.loc[group.index, "holm_pvalue_within_9_test_family"] = holm_adjust(
            group["bootstrap_normal_pvalue"].to_numpy(float)
        )

    ancestry_comparison_rows: list[dict[str, object]] = []
    for cluster_type in cluster_types:
        keys = [
            key for key in point_cache
            if key[-1] == cluster_type and key[3] == "three_way_shared"
        ]
        for key in keys:
            metric, ancestry, panel, deployment, method, rank_view, cutoff, endpoint, _ = key
            for ancestry_a, ancestry_b in (("AA", "CH"), ("AA", "NHW"), ("CH", "NHW")):
                if ancestry != ancestry_a:
                    continue
                key_b = (
                    metric, ancestry_b, panel, deployment, method,
                    rank_view, cutoff, endpoint, cluster_type,
                )
                if key_b not in point_cache:
                    continue
                point = point_cache[key] - point_cache[key_b]
                reps = replicate_cache[key] - replicate_cache[key_b]
                ancestry_comparison_rows.append({
                    "metric": metric, "evidence_or_endpoint": endpoint,
                    "panel": panel, "deployment": deployment,
                    "rank_view": rank_view, "cutoff_percent": 100 * cutoff,
                    "method": method, "ancestry_a": ancestry_a,
                    "ancestry_b": ancestry_b, "cluster_type": cluster_type,
                    "difference_a_minus_b": point,
                    **bootstrap_inference(point, reps),
                })
    ancestry_comparisons = pd.DataFrame(ancestry_comparison_rows)
    ancestry_comparisons["holm_pvalue_within_9_test_family"] = np.nan
    ancestry_family = [
        "panel", "deployment", "rank_view", "cutoff_percent", "metric",
        "evidence_or_endpoint", "cluster_type",
    ]
    for _, group in ancestry_comparisons.groupby(ancestry_family, sort=True, observed=True):
        ancestry_comparisons.loc[group.index, "holm_pvalue_within_9_test_family"] = holm_adjust(
            group["bootstrap_normal_pvalue"].to_numpy(float)
        )

    sort = ["panel", "deployment", "ancestry", "clump_threshold_r2", "rank_view", "method", "cutoff_percent"]
    yields.sort_values(sort + ["endpoint"], inplace=True, kind="mergesort")
    distributions.sort_values(sort + ["evidence_type"], inplace=True, kind="mergesort")
    regional_pvalues.sort_values(sort + ["regional_statistic"], inplace=True, kind="mergesort")
    regional_strata.sort_values(
        sort + ["proxy_count_stratum", "comparison_group"],
        inplace=True, kind="mergesort",
    )
    if not ecdf.empty:
        ecdf.sort_values(
            ["deployment", "ancestry", "rank_view", "method", "evidence_type", "group", "minus_log10_p_threshold"],
            inplace=True, kind="mergesort",
        )
    if not comparisons.empty:
        comparisons.sort_values(
            ["panel", "deployment", "ancestry", "rank_view", "cutoff_percent", "metric", "evidence_or_endpoint", "cluster_type", "method_a"],
            inplace=True, kind="mergesort",
        )
    if not ancestry_comparisons.empty:
        ancestry_comparisons.sort_values(
            ["panel", "rank_view", "cutoff_percent", "metric", "evidence_or_endpoint", "cluster_type", "method", "ancestry_a", "ancestry_b"],
            inplace=True, kind="mergesort",
        )

    yield_path = LD_ROOT / "results/native_ld_yields.tsv"
    distribution_path = LD_ROOT / "results/native_ld_top_vs_rest.tsv"
    regional_pvalue_path = LD_ROOT / "results/native_ld_regional_pvalues.tsv"
    regional_strata_path = LD_ROOT / "results/native_ld_regional_proxy_strata.tsv"
    comparison_path = LD_ROOT / "results/native_ld_model_comparisons.tsv"
    ancestry_comparison_path = LD_ROOT / "results/native_ld_ancestry_comparisons.tsv"
    ecdf_path = LD_ROOT / "results/native_ld_primary_ecdf.tsv.gz"
    group_path = LD_ROOT / "qc/native_ld_bootstrap_groups.tsv"
    atomic_table(yields, yield_path)
    atomic_table(distributions, distribution_path)
    atomic_table(regional_pvalues, regional_pvalue_path)
    atomic_table(regional_strata, regional_strata_path)
    atomic_table(comparisons, comparison_path)
    atomic_table(ancestry_comparisons, ancestry_comparison_path)
    atomic_table(ecdf, ecdf_path, gzip=True)
    atomic_table(bootstrap_groups, group_path)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_indices": str(args.indices),
        "source_indices_sha256": sha256_file(args.indices),
        "yield_rows": int(len(yields)),
        "distribution_rows": int(len(distributions)),
        "regional_pvalue_rows": int(len(regional_pvalues)),
        "regional_proxy_stratum_rows": int(len(regional_strata)),
        "model_comparison_rows": int(len(comparisons)),
        "ancestry_comparison_rows": int(len(ancestry_comparisons)),
        "ecdf_rows": int(len(ecdf)),
        "bootstrap_replicates": args.replicates,
        "primary_clump_threshold_r2": PRIMARY_THRESHOLD,
        "primary_cutoff_percent": 100 * PRIMARY_CUTOFF,
        "yield_output": str(yield_path), "yield_sha256": sha256_file(yield_path),
        "distribution_output": str(distribution_path), "distribution_sha256": sha256_file(distribution_path),
        "regional_pvalue_output": str(regional_pvalue_path), "regional_pvalue_sha256": sha256_file(regional_pvalue_path),
        "regional_proxy_strata_output": str(regional_strata_path), "regional_proxy_strata_sha256": sha256_file(regional_strata_path),
        "comparison_output": str(comparison_path), "comparison_sha256": sha256_file(comparison_path),
        "ancestry_comparison_output": str(ancestry_comparison_path), "ancestry_comparison_sha256": sha256_file(ancestry_comparison_path),
        "outcomes_used_to_define_indices_or_top_sets": False,
        "ready": True,
    }
    atomic_text(args.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
