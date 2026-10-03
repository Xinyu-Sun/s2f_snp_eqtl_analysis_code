#!/usr/bin/env python3

from __future__ import print_function

from pathlib import Path

import pandas as pd

from filer_enrichment_lib import BASE_DIR, EXPECTED_COHORT_SIZES


def require_file(path, rows):
    exists = Path(path).exists()
    rows.append({"check": "file_exists", "target": str(path), "status": "PASS" if exists else "FAIL", "details": ""})
    if not exists:
        raise SystemExit("Missing required file: %s" % path)


def append_check(rows, name, target, ok, details=""):
    rows.append({"check": name, "target": str(target), "status": "PASS" if ok else "FAIL", "details": str(details)})
    if not ok:
        raise SystemExit("%s failed for %s: %s" % (name, target, details))


def main():
    rows = []
    required = [
        BASE_DIR / "README.md",
        BASE_DIR / "input_manifest.tsv",
        BASE_DIR / "analysis_plan_and_decisions.md",
        BASE_DIR / "matched_pairs.tsv.gz",
        BASE_DIR / "union_variants.bed.gz",
        BASE_DIR / "variant_filer_category_counts.tsv.gz",
        BASE_DIR / "matching_balance.tsv",
        BASE_DIR / "annotation_enrichment_results.tsv",
        BASE_DIR / "ancestry_enrichment_interactions.tsv",
        BASE_DIR / "permutation_calibration.tsv",
        BASE_DIR / "manuscript_ready_summary.md",
        BASE_DIR / "logs" / "checksum_validation.tsv",
        BASE_DIR / "logs" / "positive_count_reproduction.tsv",
        BASE_DIR / "logs" / "plink_maf_validation.tsv",
        BASE_DIR / "logs" / "maf_join_report.tsv",
        BASE_DIR / "logs" / "positive_coordinate_control_overlap.tsv",
        BASE_DIR / "logs" / "matching_iteration_diversity.tsv",
        BASE_DIR / "filer_union_variants_tabix_FILER" / "logs" / "annotation_agreement_summary.tsv",
        BASE_DIR / "filer_union_variants_tabix_FILER" / "logs" / "filer_annotation_reuse_summary.tsv",
        BASE_DIR / "logs" / "inference_run_summary.tsv",
    ]
    for path in required:
        require_file(path, rows)

    checksums = pd.read_csv(BASE_DIR / "logs" / "checksum_validation.tsv", sep="\t")
    append_check(
        rows,
        "checksum_validation",
        "logs/checksum_validation.tsv",
        bool((checksums["status"] == "PASS").all() and len(checksums) >= 5),
        "rows=%d" % len(checksums),
    )

    positives = pd.read_csv(BASE_DIR / "logs" / "positive_count_reproduction.tsv", sep="\t")
    append_check(
        rows,
        "positive_count_reproduction",
        "logs/positive_count_reproduction.tsv",
        bool((positives["status"] == "PASS").all() and len(positives) == 12),
        "rows=%d" % len(positives),
    )

    plink = pd.read_csv(BASE_DIR / "logs" / "plink_maf_validation.tsv", sep="\t")
    cohort_ok = True
    for ancestry, expected in EXPECTED_COHORT_SIZES.items():
        row = plink[(plink["check"] == "analysis_sample_n") & (plink["ancestry"] == ancestry)]
        cohort_ok = cohort_ok and len(row) == 1 and str(expected) == str(row["observed"].iloc[0]) and row["status"].iloc[0] == "PASS"
    append_check(
        rows,
        "plink_maf_validation",
        "logs/plink_maf_validation.tsv",
        bool((plink["status"] == "PASS").all() and cohort_ok),
        "rows=%d expected_cohorts=%s" % (len(plink), EXPECTED_COHORT_SIZES),
    )

    maf = pd.read_csv(BASE_DIR / "logs" / "maf_join_report.tsv", sep="\t")
    append_check(
        rows,
        "plink_maf_coverage",
        "logs/maf_join_report.tsv",
        bool((maf["missing_maf_rows"] == 0).all() and (maf["missing_maf_unique_variants"] == 0).all() and len(maf) == 24),
        "rows=%d min_row_coverage=%s min_unique_coverage=%s" % (len(maf), maf["coverage_fraction_rows"].min(), maf["coverage_fraction_unique_variants"].min()),
    )

    overlap = pd.read_csv(BASE_DIR / "logs" / "positive_coordinate_control_overlap.tsv", sep="\t")
    append_check(
        rows,
        "positive_coordinate_controls",
        "logs/positive_coordinate_control_overlap.tsv",
        bool((overlap["status"] == "PASS").all() and (overlap["overlap_unique_coordinates"] == 0).all()),
        "rows=%d failures=%d" % (len(overlap), int((overlap["status"] != "PASS").sum())),
    )

    diversity = pd.read_csv(BASE_DIR / "logs" / "matching_iteration_diversity.tsv", sep="\t")
    append_check(
        rows,
        "matching_iteration_diversity",
        "logs/matching_iteration_diversity.tsv",
        bool((diversity["status"] == "PASS").all() and len(diversity) == 48),
        "rows=%d min_unique_signatures=%s" % (len(diversity), diversity["unique_control_set_signatures"].min()),
    )

    ann = pd.read_csv(BASE_DIR / "filer_union_variants_tabix_FILER" / "logs" / "annotation_agreement_summary.tsv", sep="\t")
    append_check(
        rows,
        "annotation_agreement",
        "filer_union_variants_tabix_FILER/logs/annotation_agreement_summary.tsv",
        bool((ann["status"] == "PASS").all() and int(ann["mismatches"].sum()) == 0),
        ann.to_dict(orient="records"),
    )

    balance = pd.read_csv(BASE_DIR / "matching_balance.tsv", sep="\t")
    append_check(
        rows,
        "matching_balance",
        "matching_balance.tsv",
        bool((balance["smd_after_gt_0.1_flag"] == 0).all() and len(balance) == 4800),
        "rows=%d max_abs_smd_after=%s" % (len(balance), balance["max_abs_smd_after"].max()),
    )

    enrichment = pd.read_csv(BASE_DIR / "annotation_enrichment_results.tsv", sep="\t")
    append_check(
        rows,
        "enrichment_results_shape",
        "annotation_enrichment_results.tsv",
        bool(len(enrichment) == 720 and enrichment["q_value_bh"].notna().all()),
        "rows=%d" % len(enrichment),
    )

    interactions = pd.read_csv(BASE_DIR / "ancestry_enrichment_interactions.tsv", sep="\t")
    append_check(
        rows,
        "ancestry_interactions_shape",
        "ancestry_enrichment_interactions.tsv",
        bool(len(interactions) == 720 and interactions["q_value_bh"].notna().all()),
        "rows=%d" % len(interactions),
    )

    permutation = pd.read_csv(BASE_DIR / "permutation_calibration.tsv", sep="\t")
    append_check(
        rows,
        "permutation_calibration",
        "permutation_calibration.tsv",
        bool((permutation["status"] == "PASS").all() and len(permutation) == 1440),
        "rows=%d failures=%d min_null_p=%s" % (len(permutation), int((permutation["status"] != "PASS").sum()), permutation["null_p_value"].min()),
    )

    out = pd.DataFrame(rows)
    out.to_csv(BASE_DIR / "logs" / "final_validation_summary.tsv", sep="\t", index=False)
    print(out.to_string(index=False), flush=True)
    print("[PASS] final validation", flush=True)


if __name__ == "__main__":
    main()
