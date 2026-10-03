#!/usr/bin/env python3

from __future__ import print_function

import gzip
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_ANALYSIS_DIR = Path("/home/xxs410/xxs410/filer_work/matched_enrichment_current_manuscript_20260830")
REFERENCE_INPUT_DIR = REFERENCE_ANALYSIS_DIR / "inputs"
MAF_DIR = BASE_DIR / "inputs"
PLINK_MAF_PATH = MAF_DIR / "benchmark_plink_maf_exact_qtl_cohorts.tsv.gz"
# Optional cache of exact-variant FILER calls from an earlier query of the same tracks; variants absent from
# the cache (or all variants, if the cache is absent) are queried with tabix.
CACHED_VARIANT_COUNTS = Path("/path/to/filer_annotation_cache/variant_filer_category_counts.tsv.gz")

ANCESTRIES = ["AA", "CH", "NHW"]
MODELS = ["borzoi", "alphagenome"]
UNIT_TYPES = ["pair", "unique_variant"]
N_MATCH_ITERATIONS = 100
N_CONTROLS = 10
SEED = 42
MAF_CALIPER = 0.10
LOG_DISTANCE_CALIPER = 0.75
NEAREST_POOL_MIN = 12
NEAREST_POOL_MULTIPLIER = 1

EXPECTED_COHORT_SIZES = {"AA": 224, "CH": 209, "NHW": 235}
MAF_SOURCE_LABEL = "PLINK1.9_exact_QTL_cohort"

CATEGORIES = [
    "ABC_Activity-by-Contact_model",
    "Accessible_chromatin",
    "ChIA-PET",
    "Chromatin_states",
    "Enhancer",
    "eQTL",
    "Hi-C",
    "Histone_ChIP-seq_peaks",
    "IM-PET",
    "pcHi-C",
    "Small_RNA",
    "sQTL",
    "TF_ChIP-seq_peaks",
    "TFBS",
    "TSS",
]

COMPARISONS = [
    (0.9, "low_pip_lt_0.01"),
    (0.9, "intermediate_pip_0.01_to_lt_0.9"),
    (0.5, "low_pip_lt_0.01"),
    (0.5, "intermediate_pip_0.01_to_lt_0.5"),
]

EXPECTED_SHA256 = {
    "inputs/susie_s2f_combined.tsv.gz": "d5e39231a79e007a28a23d8425a86aa845b5d64d5eee8fffb5a90fee7b34b802",
    "inputs/benchmark_plink_maf_exact_qtl_cohorts.tsv.gz": "7bc6f5abc24c65dc3c2da22cdc29294f6dca94ee52123660647e155728035c9a",
    "inputs/expected_positive_variant_counts.tsv": "606304483ffd808648cb1aadeb2647d4a447299d426147befd80e541d3b01a23",
    "inputs/hg38_gene_locations.txt": "1cbb3a0622e73d659863bd29a641d07f27101916a6abc419532038a916f2f009",
    "inputs/plink_maf_generation_summary.tsv": "4d2c81cfc8e6e34a0daaca3fb13d3f12a5385861123b83c5ccaa5881ff6cb2f6",
}

INPUT_PURPOSES = {
    "inputs/susie_s2f_combined.tsv.gz": "fine-mapping benchmark pool in long format (one row per model score)",
    "inputs/benchmark_plink_maf_exact_qtl_cohorts.tsv.gz": "exact-QTL-cohort PLINK MAF for all benchmark variants",
    "inputs/expected_positive_variant_counts.tsv": "expected unique positive-variant counts by model, ancestry, and threshold",
    "inputs/hg38_gene_locations.txt": "hg38 gene coordinates used to recompute TSS distance",
    "inputs/plink_maf_generation_summary.tsv": "PLINK MAF cohort-size and complete-coverage validation summary",
}

EXPECTED_POSITIVE_COUNTS = {
    (0.5, "alphagenome", "AA"): 142,
    (0.5, "alphagenome", "CH"): 635,
    (0.5, "alphagenome", "NHW"): 325,
    (0.5, "borzoi", "AA"): 136,
    (0.5, "borzoi", "CH"): 625,
    (0.5, "borzoi", "NHW"): 309,
    (0.9, "alphagenome", "AA"): 76,
    (0.9, "alphagenome", "CH"): 316,
    (0.9, "alphagenome", "NHW"): 123,
    (0.9, "borzoi", "AA"): 70,
    (0.9, "borzoi", "CH"): 314,
    (0.9, "borzoi", "NHW"): 116,
}

# build_filer_inputs.py writes an input checksum manifest and the expected positive counts under inputs/.
# When present they override the constants above, so the validation gates stay explicit.
CHECKSUM_MANIFEST = BASE_DIR / "inputs/input_checksums.tsv"
if CHECKSUM_MANIFEST.is_file():
    _checksum_frame = pd.read_csv(CHECKSUM_MANIFEST, sep="\t")
    EXPECTED_SHA256 = dict(zip(_checksum_frame["path"], _checksum_frame["sha256"]))
    INPUT_PURPOSES = dict(zip(_checksum_frame["path"], _checksum_frame["purpose"]))

POSITIVE_COUNT_PATH = BASE_DIR / "inputs/expected_positive_variant_counts.tsv"
if POSITIVE_COUNT_PATH.is_file():
    _positive_frame = pd.read_csv(POSITIVE_COUNT_PATH, sep="\t")
    EXPECTED_POSITIVE_COUNTS = {
        (float(row.pip_threshold), str(row.tool), str(row.ancestry)): int(row.unique_positive_variants)
        for row in _positive_frame.itertuples(index=False)
    }

POSITIVE_ONLY_INPUT = Path("/home/xxs410/xxs410/filer_work/positive_variants_pip_gt_0.5_by_ancestry_tool_deduplicated.tsv")
POSITIVE_ONLY_COUNTS = Path(
    "/home/xxs410/xxs410/filer_work/filer_probe/"
    "positive_variants_pip_gt_0.5_by_ancestry_tool_deduplicated_FILER/"
    "positive_variants_pip_gt_0.5_by_ancestry_tool_deduplicated.with_filer_category_counts.tsv"
)
FILER_METADATA = Path("/home/xxs410/xxs410/filer_work/filer_probe/filtered.hg38-and-lifted.blood.no_ng00102.local_paths.tsv")
FILER_RUN_SCRIPT = Path("/home/xxs410/xxs410/filer_work/filer_probe/run_filer_variant.sh")


def project_path(path_string):
    path = Path(path_string)
    if path.is_absolute():
        return path
    if str(path_string).startswith("inputs/"):
        return BASE_DIR / path
    return BASE_DIR / path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_checksums():
    rows = []
    failures = []
    for path_string, expected in EXPECTED_SHA256.items():
        path = project_path(path_string)
        observed = sha256_file(path)
        ok = observed == expected
        rows.append(
            {
                "path": path_string,
                "resolved_path": str(path),
                "size_bytes": path.stat().st_size,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if ok else "FAIL",
                "purpose": INPUT_PURPOSES.get(path_string, ""),
            }
        )
        if not ok:
            failures.append(path_string)
    return pd.DataFrame(rows), failures


def write_input_manifest():
    rows = []
    for path_string, expected in EXPECTED_SHA256.items():
        path = project_path(path_string)
        rows.append(
            {
                "path": path_string,
                "resolved_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": expected,
                "checksum_status": "PASS",
                "purpose": INPUT_PURPOSES.get(path_string, ""),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(BASE_DIR / "input_manifest.tsv", sep="\t", index=False)
    return out


def normalize_chrom_value(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s.lower().startswith("chr"):
        s = s[3:]
    return "chr" + s.upper()


def normalize_allele_series(series):
    return series.astype(str).str.strip().str.upper()


def first_present(df, primary, fallback):
    if primary in df.columns:
        left = df[primary]
    else:
        left = pd.Series(np.nan, index=df.index)
    if fallback in df.columns:
        right = df[fallback]
    else:
        right = pd.Series(np.nan, index=df.index)
    left_str = left.astype(str)
    mask = left.notna() & (left_str != "") & (left_str != "nan")
    return left.where(mask, right)


def allele_pair_series(a, b):
    a_norm = normalize_allele_series(a)
    b_norm = normalize_allele_series(b)
    vals = np.sort(pd.DataFrame({"a": a_norm, "b": b_norm}).values, axis=1)
    return pd.Series(vals[:, 0] + "/" + vals[:, 1], index=a.index)


def tss_bin_from_distance(distance):
    if pd.isna(distance):
        return np.nan
    d = float(distance)
    if d <= 3000:
        return "0-3kb"
    if d <= 12000:
        return "3-12kb"
    if d <= 35000:
        return "12-35kb"
    return ">35kb"


def maf_bin_from_maf(maf):
    if pd.isna(maf):
        return np.nan
    x = float(maf)
    if x < 0.05:
        return "<0.05"
    if x < 0.2:
        return "0.05-0.2"
    return ">=0.2"


def natural_chrom_key(chrom):
    s = str(chrom).replace("chr", "")
    if s.isdigit():
        return (0, int(s))
    if s == "X":
        return (1, 23)
    if s == "Y":
        return (1, 24)
    if s in ("M", "MT"):
        return (1, 25)
    return (2, s)


def load_gene_tss():
    path = project_path("inputs/hg38_gene_locations.txt")
    df = pd.read_csv(path, sep="\t", dtype={"chromosome_name": str})
    df = df.rename(columns={"chromosome_name": "gene_chromosome", "ensgid": "gene_id", "TSS": "tss"})
    df["gene_chromosome"] = df["gene_chromosome"].map(normalize_chrom_value)
    df["tss"] = pd.to_numeric(df["tss"], errors="coerce")
    dup = int(df["gene_id"].duplicated(keep=False).sum())
    if dup:
        same = df.groupby("gene_id")["tss"].nunique(dropna=False).max()
        if same != 1:
            raise RuntimeError("Gene TSS table has duplicate gene IDs with inconsistent TSS values")
        df = df.drop_duplicates("gene_id")
    return df[["gene_id", "gene_chromosome", "tss"]]


def load_plink_maf():
    df = pd.read_csv(PLINK_MAF_PATH, sep="\t", compression="gzip", dtype={"chromosome": str, "a1": str, "a2": str})
    required = {"ancestry", "chromosome", "position", "a1", "a2", "a1_frequency", "maf", "nchrobs", "analysis_sample_n", "maf_source"}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise RuntimeError("PLINK MAF table is missing columns: %s" % sorted(missing_cols))

    df = df[df["ancestry"].isin(ANCESTRIES)].copy()
    df["chromosome_canonical"] = df["chromosome"].map(normalize_chrom_value)
    df["position_canonical"] = pd.to_numeric(df["position"], errors="coerce").round().astype("Int64")
    df["a1"] = normalize_allele_series(df["a1"])
    df["a2"] = normalize_allele_series(df["a2"])
    df["allele_pair"] = allele_pair_series(df["a1"], df["a2"])
    df["a1_frequency"] = pd.to_numeric(df["a1_frequency"], errors="coerce")
    df["maf"] = pd.to_numeric(df["maf"], errors="coerce")
    df["analysis_sample_n"] = pd.to_numeric(df["analysis_sample_n"], errors="coerce").astype("Int64")
    df["nchrobs"] = pd.to_numeric(df["nchrobs"], errors="coerce").astype("Int64")

    validation_rows = []
    failures = []
    for ancestry in ANCESTRIES:
        sub = df[df["ancestry"] == ancestry]
        observed_sizes = sorted(int(x) for x in sub["analysis_sample_n"].dropna().unique())
        expected = EXPECTED_COHORT_SIZES[ancestry]
        ok = observed_sizes == [expected]
        validation_rows.append(
            {
                "check": "analysis_sample_n",
                "ancestry": ancestry,
                "expected": expected,
                "observed": ",".join(map(str, observed_sizes)),
                "status": "PASS" if ok else "FAIL",
            }
        )
        if not ok:
            failures.append("analysis_sample_n:%s" % ancestry)

    invalid_maf = df[df["maf"].isna() | (df["maf"] < 0.0) | (df["maf"] > 0.5)]
    validation_rows.append(
        {
            "check": "maf_bounds",
            "ancestry": "ALL",
            "expected": "0<=maf<=0.5",
            "observed": int(len(invalid_maf)),
            "status": "PASS" if len(invalid_maf) == 0 else "FAIL",
        }
    )
    if len(invalid_maf):
        failures.append("maf_bounds")

    bad_source = df[df["maf_source"] != MAF_SOURCE_LABEL]
    validation_rows.append(
        {
            "check": "maf_source",
            "ancestry": "ALL",
            "expected": MAF_SOURCE_LABEL,
            "observed": int(len(bad_source)),
            "status": "PASS" if len(bad_source) == 0 else "FAIL",
        }
    )
    if len(bad_source):
        failures.append("maf_source")

    key = ["ancestry", "chromosome_canonical", "position_canonical", "allele_pair"]
    dup = df[df.duplicated(key, keep=False)].copy()
    inconsistent = pd.DataFrame()
    if len(dup):
        inconsistent = (
            dup.groupby(key)
            .agg(maf_nunique=("maf", "nunique"), rows=("maf", "size"), sample_n_nunique=("analysis_sample_n", "nunique"))
            .reset_index()
        )
        inconsistent = inconsistent[(inconsistent["maf_nunique"] > 1) | (inconsistent["sample_n_nunique"] > 1) | (inconsistent["rows"] > 1)]
    validation_rows.append(
        {
            "check": "unique_maf_key",
            "ancestry": "ALL",
            "expected": "one row per ancestry/chromosome/position/unordered_allele_pair",
            "observed": int(len(inconsistent)),
            "status": "PASS" if len(inconsistent) == 0 else "FAIL",
        }
    )
    if len(inconsistent):
        failures.append("unique_maf_key")

    summary_path = MAF_DIR / "plink_maf_generation_summary.tsv"
    summary = pd.read_csv(summary_path, sep="\t")
    for ancestry, expected in EXPECTED_COHORT_SIZES.items():
        row = summary[summary["ancestry"] == ancestry]
        ok = len(row) == 1 and int(row["analysis_sample_n"].iloc[0]) == expected and int(row["unmatched_benchmark_variants"].iloc[0]) == 0
        validation_rows.append(
            {
                "check": "generation_summary",
                "ancestry": ancestry,
                "expected": "sample_n=%d; unmatched=0" % expected,
                "observed": row.to_dict(orient="records"),
                "status": "PASS" if ok else "FAIL",
            }
        )
        if not ok:
            failures.append("generation_summary:%s" % ancestry)

    if failures:
        bad = pd.DataFrame(validation_rows)
        bad.to_csv(BASE_DIR / "logs" / "plink_maf_validation.tsv", sep="\t", index=False)
        raise RuntimeError("PLINK MAF validation failed: %s" % failures)

    out = df[[
        "ancestry",
        "chromosome_canonical",
        "position_canonical",
        "allele_pair",
        "a1_frequency",
        "maf",
        "nchrobs",
        "analysis_sample_n",
        "maf_source",
    ]].copy()
    return out, pd.DataFrame(validation_rows)


def load_combined_base():
    df = pd.read_csv(project_path("inputs/susie_s2f_combined.tsv.gz"), sep="\t", low_memory=False)
    df = df[df["ancestry"].isin(ANCESTRIES) & df["tool"].isin(MODELS)].copy()
    for col in ["pip", "pos", "position", "pvalue", "beta", "std_error", "qvalue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["chromosome_canonical"] = first_present(df, "chromosome", "chr").map(normalize_chrom_value)
    df["position_canonical"] = first_present(df, "position", "pos")
    df["position_canonical"] = pd.to_numeric(df["position_canonical"], errors="coerce").round().astype("Int64")
    df["ref_canonical"] = normalize_allele_series(first_present(df, "ref_allele", "ref"))
    df["alt_canonical"] = normalize_allele_series(first_present(df, "alt_allele", "alt"))
    df["variant_id_canonical"] = (
        df["chromosome_canonical"].astype(str)
        + ":"
        + df["position_canonical"].astype(str)
        + "_"
        + df["ref_canonical"].astype(str)
        + "_"
        + df["alt_canonical"].astype(str)
    )
    df["allele_pair"] = allele_pair_series(df["ref_canonical"], df["alt_canonical"])
    df["coordinate_id"] = (
        df["chromosome_canonical"].astype(str)
        + ":"
        + df["position_canonical"].astype(str)
        + ":"
        + df["allele_pair"].astype(str)
    )
    df["CS_string"] = first_present(df, "CS", "CS").astype(str)
    df.loc[df["CS_string"].isin(["nan", ""]), "CS_string"] = "NA"
    return df


def load_prepared_table():
    df = load_combined_base()
    gene_tss = load_gene_tss()
    df = df.merge(gene_tss, on="gene_id", how="left")
    df["tss_distance_recomputed"] = (df["position_canonical"].astype(float) - df["tss"].astype(float)).abs()
    df["log_distance"] = np.log10(df["tss_distance_recomputed"].where(df["tss_distance_recomputed"] > 0, 1.0))
    df["tss_distance_bin_recomputed"] = df["tss_distance_recomputed"].map(tss_bin_from_distance)

    maf, maf_validation = load_plink_maf()
    df = df.merge(
        maf,
        on=["ancestry", "chromosome_canonical", "position_canonical", "allele_pair"],
        how="left",
        validate="many_to_one",
    )
    df["maf_bin"] = df["maf"].map(maf_bin_from_maf)
    df["analysis_row_id"] = np.arange(len(df), dtype=np.int64)
    df["pair_unit_id"] = (
        df["tool"].astype(str)
        + "|"
        + df["ancestry"].astype(str)
        + "|"
        + df["gene_id"].astype(str)
        + "|"
        + df["coordinate_id"].astype(str)
        + "|CS="
        + df["CS_string"].astype(str)
    )
    df["unique_variant_unit_id"] = df["tool"].astype(str) + "|" + df["ancestry"].astype(str) + "|" + df["coordinate_id"].astype(str)
    return df, maf_validation


def background_mask(df, threshold, background):
    pip = df["pip"]
    if background == "low_pip_lt_0.01":
        return (pip >= 0.0) & (pip < 0.01)
    if background.startswith("intermediate_pip_0.01_to_lt_"):
        return (pip >= 0.01) & (pip < threshold)
    raise ValueError("Unknown background: %s" % background)


def collapse_unique(df):
    sort_cols = ["coordinate_id", "pip", "tss_distance_recomputed", "gene_id", "CS_string", "variant_id_canonical"]
    ascending = [True, False, True, True, True, True]
    out = df.sort_values(sort_cols, ascending=ascending).drop_duplicates("coordinate_id", keep="first").copy()
    out["unit_id"] = out["unique_variant_unit_id"]
    return out


def make_units(df, model, ancestry, threshold, background, unit_type):
    sub = df[
        (df["tool"] == model)
        & (df["ancestry"] == ancestry)
        & df["pip"].notna()
        & df["tss_distance_bin_recomputed"].notna()
        & df["maf_bin"].notna()
    ].copy()
    all_model_ancestry = df[(df["tool"] == model) & (df["ancestry"] == ancestry) & df["pip"].notna()].copy()
    positive_all = all_model_ancestry[all_model_ancestry["pip"] >= threshold].copy()
    positive_coords = set(positive_all["coordinate_id"].dropna().astype(str))
    eligible_control = sub[background_mask(sub, threshold, background) & ~sub["coordinate_id"].isin(positive_coords)].copy()
    positive = sub[sub["pip"] >= threshold].copy()

    if unit_type == "pair":
        positive["unit_id"] = positive["pair_unit_id"]
        eligible_control["unit_id"] = eligible_control["pair_unit_id"]
        positive = positive.drop_duplicates("unit_id")
        eligible_control = eligible_control.drop_duplicates("unit_id")
    elif unit_type == "unique_variant":
        positive = collapse_unique(positive)
        eligible_control = collapse_unique(eligible_control)
    else:
        raise ValueError("Unknown unit_type: %s" % unit_type)

    return positive.reset_index(drop=True), eligible_control.reset_index(drop=True), positive_all


def standardized_mean_difference(case_values, control_values):
    x = pd.to_numeric(pd.Series(case_values), errors="coerce").dropna().astype(float)
    y = pd.to_numeric(pd.Series(control_values), errors="coerce").dropna().astype(float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    vx = x.var(ddof=1) if len(x) > 1 else 0.0
    vy = y.var(ddof=1) if len(y) > 1 else 0.0
    pooled = math.sqrt((vx + vy) / 2.0)
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def summarize_values(prefix, values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if len(s) == 0:
        return {
            prefix + "_mean": np.nan,
            prefix + "_sd": np.nan,
            prefix + "_median": np.nan,
            prefix + "_min": np.nan,
            prefix + "_max": np.nan,
        }
    return {
        prefix + "_mean": float(s.mean()),
        prefix + "_sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        prefix + "_median": float(s.median()),
        prefix + "_min": float(s.min()),
        prefix + "_max": float(s.max()),
    }


def _build_match_cache(controls):
    by_stratum = defaultdict(list)
    by_gene_stratum = defaultdict(list)
    log_distance = pd.to_numeric(controls["log_distance"], errors="coerce").astype(float).values
    maf = pd.to_numeric(controls["maf"], errors="coerce").astype(float).values
    for idx, row in controls.iterrows():
        stratum = (row["tss_distance_bin_recomputed"], row["maf_bin"])
        by_stratum[stratum].append(idx)
        by_gene_stratum[(row["gene_id"], row["tss_distance_bin_recomputed"], row["maf_bin"])].append(idx)
    by_stratum = {k: np.asarray(v, dtype=np.int64) for k, v in by_stratum.items()}
    by_gene_stratum = {k: np.asarray(v, dtype=np.int64) for k, v in by_gene_stratum.items()}
    return {
        "by_stratum": by_stratum,
        "by_gene_stratum": by_gene_stratum,
        "log_distance": log_distance,
        "maf": maf,
    }


def _sorted_by_score(indices, cache, positive_row):
    if indices is None or len(indices) == 0:
        return np.asarray([], dtype=np.int64)
    log_diff = np.abs(cache["log_distance"][indices] - float(positive_row["log_distance"]))
    maf_diff = np.abs(cache["maf"][indices] - float(positive_row["maf"]))
    keep = (log_diff <= LOG_DISTANCE_CALIPER) & (maf_diff <= MAF_CALIPER)
    if not np.any(keep):
        return np.asarray([], dtype=np.int64)
    kept_indices = indices[keep]
    score = (log_diff[keep] / 0.20) + (maf_diff[keep] / 0.02)
    score = np.where(np.isfinite(score), score, 99.0)
    return kept_indices[np.argsort(score)]


def _precompute_candidate_lists(positive, cache):
    by_stratum = cache["by_stratum"]
    by_gene_stratum = cache["by_gene_stratum"]
    infos = []
    for _, row in positive.iterrows():
        stratum = (row["tss_distance_bin_recomputed"], row["maf_bin"])
        same_key = (row["gene_id"], row["tss_distance_bin_recomputed"], row["maf_bin"])
        fallback = _sorted_by_score(by_stratum.get(stratum), cache, row)
        same = _sorted_by_score(by_gene_stratum.get(same_key), cache, row)
        infos.append({"same": same, "fallback": fallback, "candidate_count": len(fallback)})
    return infos


def _pick_from_sorted(sorted_indices, used_flags, already_selected, n_needed, rng, prefer_unused=True):
    if n_needed <= 0 or sorted_indices is None or len(sorted_indices) == 0:
        return []
    candidates = sorted_indices
    if prefer_unused:
        candidates = candidates[~used_flags[candidates]]
    if already_selected:
        selected_arr = np.asarray(list(already_selected), dtype=np.int64)
        candidates = candidates[~np.isin(candidates, selected_arr)]
    if len(candidates) == 0:
        return []
    n_take = min(n_needed, len(candidates))
    pool_size = min(len(candidates), max(NEAREST_POOL_MIN, n_take * NEAREST_POOL_MULTIPLIER))
    pool = candidates[:pool_size]
    if len(pool) <= n_take:
        return [int(x) for x in pool]
    ranks = np.arange(len(pool), dtype=float)
    weights = 1.0 / np.power(ranks + 1.0, 2.0)
    weights = weights / weights.sum()
    picked = rng.choice(pool, size=n_take, replace=False, p=weights)
    return [int(x) for x in picked]


def unit_record_fields(row, role, control_rank, same_gene_control, control_reused):
    return {
        "role": role,
        "control_rank": control_rank,
        "unit_id": row["unit_id"],
        "variant_id": row["variant_id_canonical"],
        "coordinate_id": row["coordinate_id"],
        "chromosome": row["chromosome_canonical"],
        "position": int(row["position_canonical"]),
        "ref_allele": row["ref_canonical"],
        "alt_allele": row["alt_canonical"],
        "gene_id": row["gene_id"],
        "CS": row["CS_string"],
        "pip": row["pip"],
        "a1_frequency": row.get("a1_frequency", np.nan),
        "maf": row["maf"],
        "maf_bin": row["maf_bin"],
        "maf_source": row.get("maf_source", ""),
        "tss_distance": row["tss_distance_recomputed"],
        "log_distance": row["log_distance"],
        "tss_distance_bin": row["tss_distance_bin_recomputed"],
        "same_gene_control": same_gene_control,
        "control_reused": control_reused,
    }


def _match_once_precomputed(positive, controls, candidate_infos, model, ancestry, threshold, background, unit_type, iteration):
    rng = np.random.RandomState(SEED + iteration)
    control_counts = np.asarray([x["candidate_count"] for x in candidate_infos], dtype=np.int64)
    pos_order = list(np.lexsort((rng.uniform(size=len(positive)), control_counts)))

    used_flags = np.zeros(len(controls), dtype=bool)
    records = []
    selected_control_indices = []
    selected_control_units = []
    selected_control_coords = []
    matched_positive_indices = []
    same_gene_control_count = 0
    positives_with_same_gene = 0

    for match_number, pos_idx in enumerate(pos_order):
        p = positive.iloc[pos_idx]
        info = candidate_infos[pos_idx]
        selected = []
        selected_indices_local = set()

        pick = _pick_from_sorted(info["same"], used_flags, selected_indices_local, N_CONTROLS, rng, True)
        selected.extend(pick)
        selected_indices_local.update([int(x) for x in pick])

        if len(selected) < N_CONTROLS:
            pick = _pick_from_sorted(info["fallback"], used_flags, selected_indices_local, N_CONTROLS - len(selected), rng, True)
            selected.extend(pick)
            selected_indices_local.update([int(x) for x in pick])

        if len(selected) < N_CONTROLS:
            pick = _pick_from_sorted(info["same"], used_flags, selected_indices_local, N_CONTROLS - len(selected), rng, False)
            selected.extend(pick)
            selected_indices_local.update([int(x) for x in pick])
        if len(selected) < N_CONTROLS:
            pick = _pick_from_sorted(info["fallback"], used_flags, selected_indices_local, N_CONTROLS - len(selected), rng, False)
            selected.extend(pick)
            selected_indices_local.update([int(x) for x in pick])

        if not selected:
            continue

        matched_positive_indices.append(pos_idx)
        match_group_id = "%s|%s|%s|pip%s|%s|iter%03d|m%05d" % (
            unit_type,
            model,
            ancestry,
            str(threshold),
            background,
            iteration,
            match_number,
        )
        base = {
            "unit_type": unit_type,
            "model": model,
            "ancestry": ancestry,
            "pip_threshold": threshold,
            "background": background,
            "iteration": iteration,
            "match_group_id": match_group_id,
            "positive_unit_id": p["unit_id"],
        }
        pos_record = dict(base)
        pos_record.update(unit_record_fields(p, role="positive", control_rank=0, same_gene_control="", control_reused=0))
        records.append(pos_record)

        any_same_gene = False
        for rank, ctrl_idx in enumerate(selected, start=1):
            c = controls.iloc[int(ctrl_idx)]
            ctrl_unit = str(c["unit_id"])
            was_reused = bool(used_flags[int(ctrl_idx)])
            used_flags[int(ctrl_idx)] = True
            same_gene = int(c["gene_id"] == p["gene_id"])
            same_gene_control_count += same_gene
            any_same_gene = any_same_gene or bool(same_gene)
            selected_control_indices.append(int(ctrl_idx))
            selected_control_units.append(ctrl_unit)
            selected_control_coords.append(str(c["coordinate_id"]))
            ctrl_record = dict(base)
            ctrl_record.update(
                unit_record_fields(
                    c,
                    role="control",
                    control_rank=rank,
                    same_gene_control=same_gene,
                    control_reused=int(was_reused),
                )
            )
            records.append(ctrl_record)
        if any_same_gene:
            positives_with_same_gene += 1

    matched_pos = positive.iloc[matched_positive_indices].copy() if matched_positive_indices else positive.iloc[[]].copy()
    selected_ctrl = controls.iloc[selected_control_indices].copy() if selected_control_indices else controls.iloc[[]].copy()
    total_controls = len(selected_control_indices)
    balance = {
        "unit_type": unit_type,
        "model": model,
        "ancestry": ancestry,
        "pip_threshold": threshold,
        "background": background,
        "iteration": iteration,
        "positive_units_before_matching": len(positive),
        "eligible_control_units_before_matching": len(controls),
        "matched_positive_units": len(matched_pos),
        "unmatched_positive_units": len(positive) - len(matched_pos),
        "matched_control_rows": total_controls,
        "unique_control_units": len(set(selected_control_units)),
        "unique_control_variants": len(set(selected_control_coords)),
        "control_unit_reuse_count": total_controls - len(set(selected_control_units)),
        "control_variant_reuse_count": total_controls - len(set(selected_control_coords)),
        "same_gene_control_fraction": float(same_gene_control_count / total_controls) if total_controls else np.nan,
        "positive_with_any_same_gene_control_fraction": float(positives_with_same_gene / len(matched_pos)) if len(matched_pos) else np.nan,
        "smd_maf_before": standardized_mean_difference(positive["maf"], controls["maf"]),
        "smd_log_distance_before": standardized_mean_difference(positive["log_distance"], controls["log_distance"]),
        "smd_maf_after": standardized_mean_difference(matched_pos["maf"], selected_ctrl["maf"]),
        "smd_log_distance_after": standardized_mean_difference(matched_pos["log_distance"], selected_ctrl["log_distance"]),
    }
    balance.update(summarize_values("positive_maf_before", positive["maf"]))
    balance.update(summarize_values("control_maf_before", controls["maf"]))
    balance.update(summarize_values("positive_log_distance_before", positive["log_distance"]))
    balance.update(summarize_values("control_log_distance_before", controls["log_distance"]))
    balance.update(summarize_values("positive_maf_after", matched_pos["maf"]))
    balance.update(summarize_values("control_maf_after", selected_ctrl["maf"]))
    balance.update(summarize_values("positive_log_distance_after", matched_pos["log_distance"]))
    balance.update(summarize_values("control_log_distance_after", selected_ctrl["log_distance"]))
    max_smd = np.nanmax([abs(balance["smd_maf_after"]), abs(balance["smd_log_distance_after"])])
    balance["max_abs_smd_after"] = float(max_smd) if not pd.isna(max_smd) else np.nan
    balance["smd_after_gt_0.1_flag"] = int(max_smd > 0.1) if not pd.isna(max_smd) else 1
    return records, balance


def match_iterations(positive, controls, model, ancestry, threshold, background, unit_type, n_iterations):
    controls = controls.reset_index(drop=True)
    positive = positive.reset_index(drop=True)
    cache = _build_match_cache(controls)
    candidate_infos = _precompute_candidate_lists(positive, cache)
    for iteration in range(n_iterations):
        yield _match_once_precomputed(positive, controls, candidate_infos, model, ancestry, threshold, background, unit_type, iteration)


def matched_pairs_columns():
    return [
        "unit_type",
        "model",
        "ancestry",
        "pip_threshold",
        "background",
        "iteration",
        "match_group_id",
        "positive_unit_id",
        "role",
        "control_rank",
        "unit_id",
        "variant_id",
        "coordinate_id",
        "chromosome",
        "position",
        "ref_allele",
        "alt_allele",
        "gene_id",
        "CS",
        "pip",
        "a1_frequency",
        "maf",
        "maf_bin",
        "maf_source",
        "tss_distance",
        "log_distance",
        "tss_distance_bin",
        "same_gene_control",
        "control_reused",
    ]


def write_union_bed(variant_records, out_path):
    rows = list(variant_records.values())
    rows.sort(key=lambda r: (natural_chrom_key(r["chromosome"]), int(r["position"]), r["variant_id"]))
    with gzip.open(out_path, "wt") as handle:
        for r in rows:
            pos = int(r["position"])
            start = pos - 1
            end = pos
            if start < 0:
                raise RuntimeError("Invalid BED start for %s" % r["variant_id"])
            handle.write("%s\t%d\t%d\t%s\n" % (r["chromosome"], start, end, r["variant_id"]))
    return len(rows)


def category_for_metadata_row(row):
    data_category = str(row.get("Data Category", ""))
    path = (str(row.get("filepath", "")) + "/" + str(row.get("File name", ""))).lower()
    assay = str(row.get("Assay", "")).lower()
    if data_category == "QTL":
        if "/sqtl/" in path or "sqtl" in assay:
            return "sQTL"
        return "eQTL"
    if data_category == "Chromatin interactions":
        if "activity-by-contact" in path or "nasser_2021_abc" in path:
            return "ABC_Activity-by-Contact_model"
        if "chia-pet" in path or "chia-pet" in assay:
            return "ChIA-PET"
        if "im-pet" in path or "im-pet" in assay:
            return "IM-PET"
        if "pchi" in path or "pchi-c" in assay or ("pc" in assay and "hi-c" in assay):
            return "pcHi-C"
        if "hi-c" in path or "hi-c" in assay:
            return "Hi-C"
    return data_category.replace(" ", "_")


def load_filer_file_category_map():
    meta = pd.read_csv(FILER_METADATA, sep="\t", dtype=str)
    by_file = {}
    rows = []
    for _, row in meta.iterrows():
        file_name = row["File name"]
        category = category_for_metadata_row(row)
        by_file[file_name] = category
        rows.append(
            {
                "identifier": row.get("Identifier", ""),
                "file_name": file_name,
                "category": category,
                "data_category": row.get("Data Category", ""),
                "assay": row.get("Assay", ""),
                "filepath": row.get("filepath", ""),
            }
        )
    return by_file, pd.DataFrame(rows)


def bh_qvalues(pvalues):
    p = np.asarray([np.nan if pd.isna(x) else float(x) for x in pvalues], dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.where(~np.isnan(p))[0]
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order]
    m = float(len(ranked))
    vals = ranked * m / (np.arange(len(ranked)) + 1.0)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    vals = np.minimum(vals, 1.0)
    q[order] = vals
    return q


def mh_log_or_from_groups(df, category):
    ann = pd.to_numeric(df[category], errors="coerce").fillna(0).astype(int)
    tmp = df[["match_group_id", "role"]].copy()
    tmp["ann"] = ann.values
    numerator = 0.0
    denominator = 0.0
    obs_minus_exp = 0.0
    var_sum = 0.0
    informative = 0
    for _, g in tmp.groupby("match_group_id", sort=False):
        case = g[g["role"] == "positive"]
        ctrl = g[g["role"] == "control"]
        if len(case) == 0 or len(ctrl) == 0:
            continue
        a = float(case["ann"].sum())
        b = float(len(case) - a)
        c = float(ctrl["ann"].sum())
        d = float(len(ctrl) - c)
        n = a + b + c + d
        if n <= 1:
            continue
        numerator += a * d / n
        denominator += b * c / n
        col1 = a + c
        col0 = b + d
        row1 = a + b
        row0 = c + d
        exp_a = row1 * col1 / n
        var_a = row1 * row0 * col1 * col0 / (n * n * (n - 1.0)) if n > 1 else 0.0
        obs_minus_exp += a - exp_a
        var_sum += var_a
        if col1 > 0 and col0 > 0:
            informative += 1
    log_or = math.log((numerator + 0.5) / (denominator + 0.5))
    if var_sum > 0:
        cmh_stat = (obs_minus_exp * obs_minus_exp) / var_sum
        p_value = float(chi2.sf(cmh_stat, 1))
    else:
        p_value = np.nan
    return log_or, p_value, informative


def safe_exp(x):
    if pd.isna(x):
        return np.nan
    if x > 700:
        return np.inf
    if x < -700:
        return 0.0
    return float(math.exp(float(x)))


def empirical_p_from_boot(log_values):
    vals = np.asarray([v for v in log_values if not pd.isna(v)], dtype=float)
    if len(vals) == 0:
        return np.nan
    le = (np.sum(vals <= 0.0) + 1.0) / (len(vals) + 1.0)
    ge = (np.sum(vals >= 0.0) + 1.0) / (len(vals) + 1.0)
    return float(min(1.0, 2.0 * min(le, ge)))
