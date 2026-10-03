#!/usr/bin/env python3

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import json
import platform

import numpy as np
import pandas as pd

from filer_enrichment_lib import (
    ANCESTRIES,
    BASE_DIR,
    COMPARISONS,
    EXPECTED_POSITIVE_COUNTS,
    MODELS,
    N_MATCH_ITERATIONS,
    UNIT_TYPES,
    load_prepared_table,
    make_units,
    match_iterations,
    matched_pairs_columns,
    verify_checksums,
    write_input_manifest,
    write_union_bed,
)


def write_versions(log_dir):
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with open(log_dir / "package_versions.json", "w") as handle:
        json.dump(versions, handle, indent=2, sort_keys=True)


def positive_count_reproduction(df):
    rows = []
    failures = []
    for threshold in [0.5, 0.9]:
        for model in ["alphagenome", "borzoi"]:
            for ancestry in ANCESTRIES:
                sub = df[(df["tool"] == model) & (df["ancestry"] == ancestry) & (df["pip"] >= threshold)]
                observed = int(sub["variant_id_canonical"].nunique())
                expected = EXPECTED_POSITIVE_COUNTS[(threshold, model, ancestry)]
                status = "PASS" if observed == expected else "FAIL"
                rows.append(
                    {
                        "pip_threshold": threshold,
                        "model": model,
                        "ancestry": ancestry,
                        "observed_unique_variants": observed,
                        "expected_unique_variants": expected,
                        "status": status,
                    }
                )
                if status != "PASS":
                    failures.append((threshold, model, ancestry, observed, expected))
    return pd.DataFrame(rows), failures


def pip_group_value(pip):
    if pd.isna(pip):
        return "pip_missing"
    x = float(pip)
    if x < 0.01:
        return "pip_lt_0.01"
    if x < 0.5:
        return "pip_0.01_to_lt_0.5"
    if x < 0.9:
        return "pip_0.5_to_lt_0.9"
    return "pip_ge_0.9"


def maf_join_report(df):
    data = df.copy()
    data["pip_group"] = data["pip"].map(pip_group_value)
    rows = []
    groups = ["pip_lt_0.01", "pip_0.01_to_lt_0.5", "pip_0.5_to_lt_0.9", "pip_ge_0.9"]
    for model in MODELS:
        for ancestry in ANCESTRIES:
            sub_ma = data[(data["tool"] == model) & (data["ancestry"] == ancestry)]
            for pip_group in groups:
                sub = sub_ma[sub_ma["pip_group"] == pip_group]
                missing = sub[sub["maf"].isna()]
                unique_variants = int(sub["coordinate_id"].nunique())
                missing_unique = int(missing["coordinate_id"].nunique())
                rows.append(
                    {
                        "model": model,
                        "ancestry": ancestry,
                        "pip_group": pip_group,
                        "rows": int(len(sub)),
                        "unique_variants": unique_variants,
                        "missing_maf_rows": int(len(missing)),
                        "missing_maf_unique_variants": missing_unique,
                        "coverage_fraction_rows": float(1.0 - len(missing) / len(sub)) if len(sub) else 1.0,
                        "coverage_fraction_unique_variants": float(1.0 - missing_unique / unique_variants) if unique_variants else 1.0,
                    }
                )
    return pd.DataFrame(rows)


def unmatched_maf_keys(df):
    missing = df[df["maf"].isna()].copy()
    cols = ["ancestry", "chromosome_canonical", "position_canonical", "allele_pair", "coordinate_id", "variant_id_canonical"]
    return missing[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


def tss_recompute_report(df):
    rows = []
    if "tss_distance_bin" not in df.columns:
        return pd.DataFrame(rows)
    for model in MODELS:
        for ancestry in ANCESTRIES:
            sub = df[(df["tool"] == model) & (df["ancestry"] == ancestry)]
            comparable = sub["tss_distance_bin"].notna() & sub["tss_distance_bin_recomputed"].notna()
            mismatches = comparable & (sub["tss_distance_bin"].astype(str) != sub["tss_distance_bin_recomputed"].astype(str))
            rows.append(
                {
                    "model": model,
                    "ancestry": ancestry,
                    "rows": int(len(sub)),
                    "missing_recomputed_tss_rows": int(sub["tss_distance_recomputed"].isna().sum()),
                    "comparable_tss_bin_rows": int(comparable.sum()),
                    "tss_bin_mismatch_rows": int(mismatches.sum()),
                }
            )
    return pd.DataFrame(rows)


def add_variant_record(variant_records, row):
    vid = str(row["variant_id_canonical"])
    if vid not in variant_records:
        variant_records[vid] = {
            "variant_id": vid,
            "chromosome": str(row["chromosome_canonical"]),
            "position": int(row["position_canonical"]),
            "ref_allele": str(row["ref_canonical"]),
            "alt_allele": str(row["alt_canonical"]),
        }


def add_variant_record_from_match(variant_records, rec):
    vid = str(rec["variant_id"])
    if vid not in variant_records:
        variant_records[vid] = {
            "variant_id": vid,
            "chromosome": str(rec["chromosome"]),
            "position": int(rec["position"]),
            "ref_allele": str(rec["ref_allele"]),
            "alt_allele": str(rec["alt_allele"]),
        }


def control_signature_and_set(records):
    units = sorted(str(r["unit_id"]) for r in records if r["role"] == "control")
    digest = hashlib.sha256("\n".join(units).encode("utf-8")).hexdigest()
    return digest, set(units)


def mean_pairwise_jaccard(sets):
    if len(sets) < 2:
        return np.nan, np.nan, np.nan
    vals = []
    for i in range(len(sets)):
        a = sets[i]
        for j in range(i + 1, len(sets)):
            b = sets[j]
            denom = len(a | b)
            vals.append(float(len(a & b) / denom) if denom else np.nan)
    vals = pd.Series(vals).dropna()
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(vals.mean()), float(vals.median()), float(vals.max())


def main():
    parser = argparse.ArgumentParser(description="Prepare matched FILER enrichment inputs")
    parser.add_argument("--n-match-iterations", type=int, default=N_MATCH_ITERATIONS)
    args = parser.parse_args()

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    (BASE_DIR / "scripts").mkdir(exist_ok=True)
    write_versions(log_dir)

    checksum_df, checksum_failures = verify_checksums()
    checksum_df.to_csv(log_dir / "checksum_validation.tsv", sep="\t", index=False)
    write_input_manifest()
    if checksum_failures:
        print("Checksum failures:", checksum_failures)
        raise SystemExit(2)
    print("[PASS] all input checksums match")

    df, maf_validation = load_prepared_table()
    maf_validation.to_csv(log_dir / "plink_maf_validation.tsv", sep="\t", index=False)
    if (maf_validation["status"] != "PASS").any():
        print(maf_validation.to_string(index=False))
        raise SystemExit("PLINK MAF validation failed")
    print("[PASS] PLINK MAF keys, bounds, source label, and exact cohort sizes validate")

    count_df, count_failures = positive_count_reproduction(df)
    count_df.to_csv(log_dir / "positive_count_reproduction.tsv", sep="\t", index=False)
    if count_failures:
        print(count_df.to_string(index=False))
        raise SystemExit("Positive unique-variant counts did not reproduce; stopping before FILER")
    print("[PASS] positive-only unique-variant sample sizes reproduce")

    maf_report = maf_join_report(df)
    maf_report.to_csv(log_dir / "maf_join_report.tsv", sep="\t", index=False)
    missing_keys = unmatched_maf_keys(df)
    missing_keys.to_csv(log_dir / "unmatched_maf_keys.tsv", sep="\t", index=False)
    if int(maf_report["missing_maf_rows"].sum()) > 0:
        print(maf_report.to_string(index=False))
        print(missing_keys.head(50).to_string(index=False))
        raise SystemExit("PLINK MAF coverage is not 100%; stopping before matching")
    print("[PASS] PLINK MAF coverage is 100% for every model/ancestry/PIP group")

    tss_report = tss_recompute_report(df)
    tss_report.to_csv(log_dir / "tss_recompute_report.tsv", sep="\t", index=False)

    matched_path = BASE_DIR / "matched_pairs.tsv.gz"
    balance_rows = []
    variant_records = {}
    unmatched_detail_rows = []
    unit_count_rows = []
    diversity_rows = []
    positive_control_overlap_rows = []

    with gzip.open(matched_path, "wt", compresslevel=1, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matched_pairs_columns(), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for unit_type in UNIT_TYPES:
            for threshold, background in COMPARISONS:
                for model in MODELS:
                    for ancestry in ANCESTRIES:
                        positives, controls, positive_all = make_units(df, model, ancestry, threshold, background, unit_type)
                        positive_coords = set(positive_all["coordinate_id"].dropna().astype(str))
                        for _, row in positives.iterrows():
                            add_variant_record(variant_records, row)
                        unit_count_rows.append(
                            {
                                "unit_type": unit_type,
                                "model": model,
                                "ancestry": ancestry,
                                "pip_threshold": threshold,
                                "background": background,
                                "positive_units": int(len(positives)),
                                "eligible_control_units": int(len(controls)),
                                "positive_rows_before_tss_or_maf_filter": int(len(positive_all)),
                            }
                        )
                        print(
                            "[match] %s %s %s pip>=%s vs %s: %d positives, %d eligible controls"
                            % (unit_type, model, ancestry, threshold, background, len(positives), len(controls)),
                            flush=True,
                        )
                        family_signatures = []
                        family_sets = []
                        family_overlap_rows = []
                        for records, balance in match_iterations(
                            positives,
                            controls,
                            model,
                            ancestry,
                            threshold,
                            background,
                            unit_type,
                            args.n_match_iterations,
                        ):
                            signature, control_set = control_signature_and_set(records)
                            balance["control_set_signature"] = signature
                            family_signatures.append(signature)
                            family_sets.append(control_set)

                            control_coords = [str(rec["coordinate_id"]) for rec in records if rec["role"] == "control"]
                            overlap_coords = sorted(set(control_coords) & positive_coords)
                            overlap_row = {
                                "unit_type": unit_type,
                                "model": model,
                                "ancestry": ancestry,
                                "pip_threshold": threshold,
                                "background": background,
                                "iteration": balance["iteration"],
                                "overlap_control_rows": int(sum(1 for x in control_coords if x in positive_coords)),
                                "overlap_unique_coordinates": int(len(overlap_coords)),
                                "example_overlap_coordinates": ",".join(overlap_coords[:20]),
                                "status": "PASS" if len(overlap_coords) == 0 else "FAIL",
                            }
                            family_overlap_rows.append(overlap_row)
                            positive_control_overlap_rows.append(overlap_row)

                            for rec in records:
                                writer.writerow(rec)
                                add_variant_record_from_match(variant_records, rec)
                            balance_rows.append(balance)
                            if balance["unmatched_positive_units"] > 0:
                                unmatched_detail_rows.append(balance)

                        mean_j, median_j, max_j = mean_pairwise_jaccard(family_sets)
                        unique_signatures = len(set(family_signatures))
                        selected_control_size = max([len(s) for s in family_sets] or [0])
                        eligible_multiple = int(len(controls) > selected_control_size and args.n_match_iterations >= 2)
                        diversity_status = "PASS"
                        if eligible_multiple and unique_signatures < 2:
                            diversity_status = "FAIL"
                        if args.n_match_iterations < 2:
                            diversity_status = "NOT_EVALUATED"
                        diversity_rows.append(
                            {
                                "unit_type": unit_type,
                                "model": model,
                                "ancestry": ancestry,
                                "pip_threshold": threshold,
                                "background": background,
                                "n_iterations": int(args.n_match_iterations),
                                "eligible_control_units": int(len(controls)),
                                "max_selected_control_units_per_iteration": int(selected_control_size),
                                "more_than_one_eligible_matching_configuration": eligible_multiple,
                                "unique_control_set_signatures": int(unique_signatures),
                                "mean_pairwise_jaccard": mean_j,
                                "median_pairwise_jaccard": median_j,
                                "max_pairwise_jaccard": max_j,
                                "status": diversity_status,
                            }
                        )

    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(BASE_DIR / "matching_balance.tsv", sep="\t", index=False)
    pd.DataFrame(unit_count_rows).to_csv(log_dir / "matching_unit_counts.tsv", sep="\t", index=False)
    pd.DataFrame(diversity_rows).to_csv(log_dir / "matching_iteration_diversity.tsv", sep="\t", index=False)
    pd.DataFrame(positive_control_overlap_rows).to_csv(log_dir / "positive_coordinate_control_overlap.tsv", sep="\t", index=False)
    if unmatched_detail_rows:
        pd.DataFrame(unmatched_detail_rows).to_csv(log_dir / "unmatched_positive_iterations.tsv", sep="\t", index=False)
    else:
        pd.DataFrame(columns=balance_df.columns).to_csv(log_dir / "unmatched_positive_iterations.tsv", sep="\t", index=False)

    bed_n = write_union_bed(variant_records, BASE_DIR / "union_variants.bed.gz")
    print("[OK] wrote %s" % matched_path)
    print("[OK] wrote union_variants.bed.gz with %d unique variants" % bed_n)

    fail_balance = balance_df[balance_df["smd_after_gt_0.1_flag"] > 0]
    fail_balance.to_csv(log_dir / "matching_balance_failures.tsv", sep="\t", index=False)
    diversity_df = pd.DataFrame(diversity_rows)
    fail_diversity = diversity_df[diversity_df["status"] == "FAIL"]
    fail_diversity.to_csv(log_dir / "matching_iteration_diversity_failures.tsv", sep="\t", index=False)
    overlap_df = pd.DataFrame(positive_control_overlap_rows)
    fail_overlap = overlap_df[overlap_df["status"] == "FAIL"]
    fail_overlap.to_csv(log_dir / "positive_coordinate_control_overlap_failures.tsv", sep="\t", index=False)

    unmatched_total = int((balance_df["unmatched_positive_units"] > 0).sum())
    if len(fail_balance):
        print(fail_balance.head(20).to_string(index=False))
        raise SystemExit("Matching balance failed for %d rows" % len(fail_balance))
    if len(fail_diversity):
        print(fail_diversity.to_string(index=False))
        raise SystemExit("Control-set diversity failed for %d analysis families" % len(fail_diversity))
    if len(fail_overlap):
        print(fail_overlap.head(20).to_string(index=False))
        raise SystemExit("Positive coordinates appeared among controls")
    print("[PASS] all after-match SMD <= 0.1; unmatched-positive iterations reported separately (%d rows)" % unmatched_total)
    if args.n_match_iterations >= 2:
        print("[PASS] stochastic matching produced at least two distinct control sets wherever required")
    print("[PASS] no positive coordinate appears among controls")


if __name__ == "__main__":
    main()
