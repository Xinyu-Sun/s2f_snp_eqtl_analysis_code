#!/usr/bin/env python3

from __future__ import print_function

import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from filer_enrichment_lib import (
    ANCESTRIES,
    BASE_DIR,
    CATEGORIES,
    COMPARISONS,
    MODELS,
    SEED,
    UNIT_TYPES,
    bh_qvalues,
    empirical_p_from_boot,
    safe_exp,
)


BOOTSTRAP_REPS = 1000
PERMUTATION_REPS = 200
RNG = np.random.RandomState(SEED + 3000)


def percentile(values, q):
    vals = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if len(vals) == 0:
        return np.nan
    return float(np.percentile(vals, q))


def safe_percentile_exp(values, q):
    value = percentile(values, q)
    return safe_exp(value)


def mh_arrays(pos_mat, ctrl_mat, n_controls, weights=None, return_p=False):
    if pos_mat.shape[0] == 0:
        n_cat = pos_mat.shape[1]
        empty = np.full(n_cat, np.nan)
        if return_p:
            return empty, empty.copy(), np.zeros(n_cat, dtype=int)
        return empty

    pos = pos_mat.astype(float)
    ctrl = ctrl_mat.astype(float)
    n_ctrl = n_controls.astype(float)
    n = n_ctrl + 1.0
    weight = np.ones(pos.shape[0], dtype=float) if weights is None else weights.astype(float)
    weight = weight.reshape((-1, 1))

    a = pos
    b = 1.0 - pos
    c = ctrl
    d = n_ctrl.reshape((-1, 1)) - c
    n_col = n.reshape((-1, 1))

    numerator = np.sum(weight * a * d / n_col, axis=0)
    denominator = np.sum(weight * b * c / n_col, axis=0)
    log_or = np.log((numerator + 0.5) / (denominator + 0.5))

    if not return_p:
        return log_or

    col1 = a + c
    col0 = b + d
    row1 = 1.0
    row0 = n_ctrl.reshape((-1, 1))
    var_a = row1 * row0 * col1 * col0 / (n_col * n_col * (n_col - 1.0))
    obs_minus_exp = np.sum(a - (row1 * col1 / n_col), axis=0)
    var_sum = np.sum(var_a, axis=0)
    stat = np.full(len(CATEGORIES), np.nan)
    ok = var_sum > 0
    stat[ok] = (obs_minus_exp[ok] * obs_minus_exp[ok]) / var_sum[ok]
    p_value = np.full(len(CATEGORIES), np.nan)
    p_value[ok] = chi2.sf(stat[ok], 1)
    informative = np.sum((col1 > 0) & (col0 > 0), axis=0).astype(int)
    return log_or, p_value, informative


def read_annotations():
    path = BASE_DIR / "variant_filer_category_counts.tsv.gz"
    if not path.exists():
        raise SystemExit("Missing %s; run FILER annotation before inference" % path)
    usecols = ["variant_id"] + ["filer_binary_" + c for c in CATEGORIES]
    ann = pd.read_csv(path, sep="\t", compression="gzip", usecols=usecols)
    if ann["variant_id"].duplicated().any():
        raise SystemExit("variant_filer_category_counts.tsv.gz has duplicate variant_id rows")
    ann = ann.rename(columns={"filer_binary_" + c: c for c in CATEGORIES})
    for cat in CATEGORIES:
        ann[cat] = pd.to_numeric(ann[cat], errors="raise").astype(np.int8)
    return ann


def read_matched_with_annotations():
    matched_path = BASE_DIR / "matched_pairs.tsv.gz"
    if not matched_path.exists():
        raise SystemExit("Missing %s; run matching before inference" % matched_path)
    ann = read_annotations()
    usecols = [
        "unit_type",
        "model",
        "ancestry",
        "pip_threshold",
        "background",
        "iteration",
        "match_group_id",
        "role",
        "variant_id",
        "gene_id",
    ]
    dtypes = {
        "unit_type": "category",
        "model": "category",
        "ancestry": "category",
        "background": "category",
        "iteration": np.int16,
        "role": "category",
        "variant_id": str,
        "gene_id": str,
        "match_group_id": str,
    }
    print("[load] reading matched pairs", flush=True)
    matched = pd.read_csv(matched_path, sep="\t", compression="gzip", usecols=usecols, dtype=dtypes)
    matched["pip_threshold"] = pd.to_numeric(matched["pip_threshold"], errors="raise")
    print("[load] merging FILER binary annotations", flush=True)
    out = matched.merge(ann, on="variant_id", how="left", validate="many_to_one")
    missing = int(out[CATEGORIES].isna().any(axis=1).sum())
    if missing:
        raise SystemExit("Matched rows missing FILER annotation: %d" % missing)
    for cat in CATEGORIES:
        out[cat] = out[cat].astype(np.int8)
    return out


def combo_key(unit_type, model, ancestry, threshold, background):
    return (unit_type, model, ancestry, float(threshold), background)


def build_iteration_groups(combo_df):
    keys = ["iteration", "match_group_id"]
    pos = combo_df[combo_df["role"] == "positive"][keys + ["gene_id"] + CATEGORIES].copy()
    ctrl = combo_df[combo_df["role"] == "control"][keys + CATEGORIES].copy()
    if len(pos) == 0 or len(ctrl) == 0:
        return OrderedDict()

    pos = pos.drop_duplicates(keys)
    pos = pos.rename(columns={cat: cat + "_positive" for cat in CATEGORIES})
    ctrl_sum = ctrl.groupby(keys, sort=False, observed=True)[CATEGORIES].sum()
    ctrl_sum = ctrl_sum.rename(columns={cat: cat + "_control" for cat in CATEGORIES})
    n_ctrl = ctrl.groupby(keys, sort=False, observed=True).size().rename("n_controls")
    summary = pos.set_index(keys).join(ctrl_sum, how="inner").join(n_ctrl, how="inner").reset_index()

    by_iteration = OrderedDict()
    for iteration, g in summary.groupby("iteration", sort=True):
        g = g.reset_index(drop=True)
        genes = g["gene_id"].astype(str)
        gene_codes, gene_values = pd.factorize(genes, sort=False)
        pos_mat = g[[cat + "_positive" for cat in CATEGORIES]].to_numpy(dtype=np.int8)
        ctrl_mat = g[[cat + "_control" for cat in CATEGORIES]].to_numpy(dtype=np.int16)
        n_controls = g["n_controls"].to_numpy(dtype=np.int16)
        by_iteration[int(iteration)] = {
            "match_group_id": g["match_group_id"].astype(str).to_numpy(),
            "genes": np.asarray(gene_values, dtype=object),
            "gene_codes": gene_codes.astype(np.int32),
            "pos_mat": pos_mat,
            "ctrl_mat": ctrl_mat,
            "n_controls": n_controls,
            "total_mat": (pos_mat.astype(np.int16) + ctrl_mat).astype(np.int16),
        }
    return by_iteration


def iteration_point_rows(iter_groups, base):
    rows = []
    for iteration, group in iter_groups.items():
        log_or, p_value, informative = mh_arrays(
            group["pos_mat"],
            group["ctrl_mat"],
            group["n_controls"],
            return_p=True,
        )
        positive_n = int(group["pos_mat"].shape[0])
        control_n = int(group["n_controls"].sum())
        positive_overlap = group["pos_mat"].sum(axis=0)
        control_overlap = group["ctrl_mat"].sum(axis=0)
        for j, cat in enumerate(CATEGORIES):
            row = dict(base)
            row.update(
                {
                    "iteration": iteration,
                    "category": cat,
                    "log_or": float(log_or[j]),
                    "or": safe_exp(log_or[j]),
                    "cmh_p_value": float(p_value[j]) if not pd.isna(p_value[j]) else np.nan,
                    "informative_strata": int(informative[j]),
                    "matched_positive_units": positive_n,
                    "matched_control_rows": control_n,
                    "positive_overlap_count": int(positive_overlap[j]),
                    "control_overlap_count": int(control_overlap[j]),
                    "positive_overlap_fraction": float(positive_overlap[j] / positive_n) if positive_n else np.nan,
                    "control_overlap_fraction": float(control_overlap[j] / control_n) if control_n else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_combo(iter_groups):
    iteration_ids = list(iter_groups.keys())
    out = np.full((BOOTSTRAP_REPS, len(CATEGORIES)), np.nan, dtype=float)
    for rep in range(BOOTSTRAP_REPS):
        group = iter_groups[int(RNG.choice(iteration_ids))]
        n_genes = len(group["genes"])
        if n_genes == 0:
            continue
        gene_weights = RNG.multinomial(n_genes, np.repeat(1.0 / n_genes, n_genes))
        weights = gene_weights[group["gene_codes"]].astype(float)
        out[rep, :] = mh_arrays(group["pos_mat"], group["ctrl_mat"], group["n_controls"], weights=weights)
    return out


def permute_combo(iter_groups):
    iteration_ids = list(iter_groups.keys())
    out = np.full((PERMUTATION_REPS, len(CATEGORIES)), np.nan, dtype=float)
    for rep in range(PERMUTATION_REPS):
        group = iter_groups[int(RNG.choice(iteration_ids))]
        n_total = (group["n_controls"] + 1).astype(int)
        probs = group["total_mat"].astype(float) / n_total.reshape((-1, 1)).astype(float)
        perm_pos = RNG.binomial(1, probs).astype(np.int8)
        perm_ctrl = group["total_mat"].astype(np.int16) - perm_pos.astype(np.int16)
        out[rep, :] = mh_arrays(perm_pos, perm_ctrl, group["n_controls"])
    return out


def summarize_enrichment(base, point_df, boot_mat):
    rows = []
    for j, cat in enumerate(CATEGORIES):
        cat_points = point_df[point_df["category"] == cat]
        boots = boot_mat[:, j]
        row = dict(base)
        row.update(
            {
                "category": cat,
                "n_match_iterations": int(cat_points["iteration"].nunique()),
                "bootstrap_reps": BOOTSTRAP_REPS,
                "matched_positive_units_median": float(cat_points["matched_positive_units"].median()),
                "matched_control_rows_median": float(cat_points["matched_control_rows"].median()),
                "positive_overlap_count_median": float(cat_points["positive_overlap_count"].median()),
                "control_overlap_count_median": float(cat_points["control_overlap_count"].median()),
                "positive_overlap_fraction_median": float(cat_points["positive_overlap_fraction"].median()),
                "control_overlap_fraction_median": float(cat_points["control_overlap_fraction"].median()),
                "informative_strata_median": float(cat_points["informative_strata"].median()),
                "iteration_log_or_median": float(cat_points["log_or"].median()),
                "iteration_or_median": safe_exp(float(cat_points["log_or"].median())),
                "iteration_log_or_min": float(cat_points["log_or"].min()),
                "iteration_log_or_max": float(cat_points["log_or"].max()),
                "cmh_p_value_median": float(cat_points["cmh_p_value"].median())
                if cat_points["cmh_p_value"].notna().any()
                else np.nan,
                "bootstrap_log_or_median": percentile(boots, 50),
                "odds_ratio": safe_percentile_exp(boots, 50),
                "ci95_low": safe_percentile_exp(boots, 2.5),
                "ci95_high": safe_percentile_exp(boots, 97.5),
                "bootstrap_p_value": empirical_p_from_boot(boots),
            }
        )
        rows.append(row)
    return rows


def summarize_permutation(base, perm_mat):
    rows = []
    for j, cat in enumerate(CATEGORIES):
        vals = perm_mat[:, j]
        median_log = percentile(vals, 50)
        low_log = percentile(vals, 2.5)
        high_log = percentile(vals, 97.5)
        null_p = empirical_p_from_boot(vals)
        center_pass = int(null_p >= 0.05) if not pd.isna(null_p) else 0
        ci_includes_one = int(low_log <= 0.0 <= high_log) if not pd.isna(low_log) and not pd.isna(high_log) else 0
        row = dict(base)
        row.update(
            {
                "calibration_type": "within_stratum_label_permutation",
                "contrast": "",
                "category": cat,
                "permutation_reps": PERMUTATION_REPS,
                "null_log_or_median": median_log,
                "null_or_median": safe_exp(median_log),
                "null_or_ci95_low": safe_exp(low_log),
                "null_or_ci95_high": safe_exp(high_log),
                "null_p_value": null_p,
                "center_pass": center_pass,
                "ci_includes_one": ci_includes_one,
                "status": "PASS" if center_pass and ci_includes_one else "FAIL",
            }
        )
        rows.append(row)
    return rows


def add_q_values(df, group_cols, p_col, q_col):
    df[q_col] = np.nan
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        df.loc[idx, q_col] = bh_qvalues(df.loc[idx, p_col].values)
    return df


def main():
    start = time.time()
    matched = read_matched_with_annotations()
    enrichment_rows = []
    point_rows = []
    permutation_rows = []
    boot_by_key = {}
    perm_by_key = {}

    total = len(UNIT_TYPES) * len(MODELS) * len(ANCESTRIES) * len(COMPARISONS)
    done = 0
    for unit_type in UNIT_TYPES:
        for model in MODELS:
            for ancestry in ANCESTRIES:
                for threshold, background in COMPARISONS:
                    done += 1
                    base = {
                        "unit_type": unit_type,
                        "model": model,
                        "ancestry": ancestry,
                        "pip_threshold": float(threshold),
                        "background": background,
                    }
                    mask = (
                        (matched["unit_type"].astype(str) == unit_type)
                        & (matched["model"].astype(str) == model)
                        & (matched["ancestry"].astype(str) == ancestry)
                        & (matched["pip_threshold"] == float(threshold))
                        & (matched["background"].astype(str) == background)
                    )
                    combo_df = matched.loc[mask, ["iteration", "match_group_id", "role", "variant_id", "gene_id"] + CATEGORIES]
                    iter_groups = build_iteration_groups(combo_df)
                    if not iter_groups:
                        raise SystemExit("No matched strata for %s" % (base,))
                    point_df = iteration_point_rows(iter_groups, base)
                    boot_mat = bootstrap_combo(iter_groups)
                    perm_mat = permute_combo(iter_groups)
                    point_rows.append(point_df)
                    enrichment_rows.extend(summarize_enrichment(base, point_df, boot_mat))
                    permutation_rows.extend(summarize_permutation(base, perm_mat))
                    for j, cat in enumerate(CATEGORIES):
                        key = combo_key(unit_type, model, ancestry, threshold, background) + (cat,)
                        boot_by_key[key] = boot_mat[:, j].copy()
                        perm_by_key[key] = perm_mat[:, j].copy()
                    print("[inference] %d / %d %s %s %s pip>=%s %s" % (done, total, unit_type, model, ancestry, threshold, background), flush=True)

    enrichment = pd.DataFrame(enrichment_rows)
    enrichment = add_q_values(
        enrichment,
        ["unit_type", "model", "ancestry", "pip_threshold", "background"],
        "bootstrap_p_value",
        "q_value_bh",
    )
    enrichment.to_csv(BASE_DIR / "annotation_enrichment_results.tsv", sep="\t", index=False)

    point_all = pd.concat(point_rows, ignore_index=True)
    point_all.to_csv(BASE_DIR / "logs" / "annotation_enrichment_iteration_estimates.tsv.gz", sep="\t", index=False, compression="gzip")

    interaction_rows = []
    for unit_type in UNIT_TYPES:
        for model in MODELS:
            for threshold, background in COMPARISONS:
                for cat in CATEGORIES:
                    for anc_a, anc_b in [("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")]:
                        key_a = combo_key(unit_type, model, anc_a, threshold, background) + (cat,)
                        key_b = combo_key(unit_type, model, anc_b, threshold, background) + (cat,)
                        boot_diff = boot_by_key[key_a] - boot_by_key[key_b]
                        point_a = enrichment[
                            (enrichment["unit_type"] == unit_type)
                            & (enrichment["model"] == model)
                            & (enrichment["ancestry"] == anc_a)
                            & (enrichment["pip_threshold"] == float(threshold))
                            & (enrichment["background"] == background)
                            & (enrichment["category"] == cat)
                        ]["bootstrap_log_or_median"].iloc[0]
                        point_b = enrichment[
                            (enrichment["unit_type"] == unit_type)
                            & (enrichment["model"] == model)
                            & (enrichment["ancestry"] == anc_b)
                            & (enrichment["pip_threshold"] == float(threshold))
                            & (enrichment["background"] == background)
                            & (enrichment["category"] == cat)
                        ]["bootstrap_log_or_median"].iloc[0]
                        interaction_rows.append(
                            {
                                "unit_type": unit_type,
                                "model": model,
                                "pip_threshold": float(threshold),
                                "background": background,
                                "category": cat,
                                "contrast": "%s_vs_%s" % (anc_a, anc_b),
                                "ancestry_a": anc_a,
                                "ancestry_b": anc_b,
                                "log_or_difference": float(point_a - point_b),
                                "interaction_odds_ratio": safe_exp(point_a - point_b),
                                "ci95_low": safe_percentile_exp(boot_diff, 2.5),
                                "ci95_high": safe_percentile_exp(boot_diff, 97.5),
                                "bootstrap_p_value": empirical_p_from_boot(boot_diff),
                                "bootstrap_reps": BOOTSTRAP_REPS,
                            }
                        )

    interactions = pd.DataFrame(interaction_rows)
    interactions = add_q_values(
        interactions,
        ["unit_type", "model", "pip_threshold", "background"],
        "bootstrap_p_value",
        "q_value_bh",
    )
    interactions.to_csv(BASE_DIR / "ancestry_enrichment_interactions.tsv", sep="\t", index=False)

    for unit_type in UNIT_TYPES:
        for model in MODELS:
            for threshold, background in COMPARISONS:
                for cat in CATEGORIES:
                    for anc_a, anc_b in [("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")]:
                        key_a = combo_key(unit_type, model, anc_a, threshold, background) + (cat,)
                        key_b = combo_key(unit_type, model, anc_b, threshold, background) + (cat,)
                        vals = perm_by_key[key_a] - perm_by_key[key_b]
                        median_log = percentile(vals, 50)
                        low_log = percentile(vals, 2.5)
                        high_log = percentile(vals, 97.5)
                        null_p = empirical_p_from_boot(vals)
                        center_pass = int(null_p >= 0.05) if not pd.isna(null_p) else 0
                        ci_includes_one = int(low_log <= 0.0 <= high_log) if not pd.isna(low_log) and not pd.isna(high_log) else 0
                        permutation_rows.append(
                            {
                                "unit_type": unit_type,
                                "model": model,
                                "ancestry": "",
                                "pip_threshold": float(threshold),
                                "background": background,
                                "calibration_type": "ancestry_interaction_label_permutation",
                                "contrast": "%s_vs_%s" % (anc_a, anc_b),
                                "category": cat,
                                "permutation_reps": PERMUTATION_REPS,
                                "null_log_or_median": median_log,
                                "null_or_median": safe_exp(median_log),
                                "null_or_ci95_low": safe_exp(low_log),
                                "null_or_ci95_high": safe_exp(high_log),
                                "null_p_value": empirical_p_from_boot(vals),
                                "center_pass": center_pass,
                                "ci_includes_one": ci_includes_one,
                                "status": "PASS" if center_pass and ci_includes_one else "FAIL",
                            }
                        )

    permutation = pd.DataFrame(permutation_rows)
    permutation.to_csv(BASE_DIR / "permutation_calibration.tsv", sep="\t", index=False)
    failures = permutation[permutation["status"] != "PASS"]
    if len(failures):
        failures.to_csv(BASE_DIR / "logs" / "permutation_calibration_failures.tsv", sep="\t", index=False)
        raise SystemExit("Permutation calibration failed for %d rows" % len(failures))

    elapsed = time.time() - start
    pd.DataFrame(
        [
            {
                "status": "PASS",
                "bootstrap_reps": BOOTSTRAP_REPS,
                "permutation_reps": PERMUTATION_REPS,
                "enrichment_rows": int(len(enrichment)),
                "interaction_rows": int(len(interactions)),
                "permutation_rows": int(len(permutation)),
                "elapsed_seconds": round(elapsed, 3),
            }
        ]
    ).to_csv(BASE_DIR / "logs" / "inference_run_summary.tsv", sep="\t", index=False)
    print("[PASS] enrichment inference and permutation calibration", flush=True)


if __name__ == "__main__":
    main()
