#!/usr/bin/env python3

from __future__ import print_function

import argparse
import gzip
import subprocess
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from filer_enrichment_lib import (
    BASE_DIR,
    CATEGORIES,
    FILER_METADATA,
    POSITIVE_ONLY_COUNTS,
    CACHED_VARIANT_COUNTS,
    load_filer_file_category_map,
)


VARIANT_INDEX = None
BED_PATH = None


def parse_variant_id(variant_id):
    chrom, rest = variant_id.split(":", 1)
    position, ref, alt = rest.split("_")
    return chrom, int(position), ref, alt


def load_union_bed(path):
    rows = []
    by_chrom = defaultdict(list)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, start, end, variant_id = line.rstrip("\n").split("\t")[:4]
            start_i = int(start)
            end_i = int(end)
            parsed_chrom, position, ref, alt = parse_variant_id(variant_id)
            if parsed_chrom != chrom:
                raise RuntimeError("BED chromosome does not match variant_id for %s" % variant_id)
            rows.append(
                {
                    "variant_id": variant_id,
                    "chromosome": chrom,
                    "start": start_i,
                    "end": end_i,
                    "position": position,
                    "ref_allele": ref,
                    "alt_allele": alt,
                }
            )
            by_chrom[chrom].append((start_i, variant_id))
    index = {}
    for chrom, vals in by_chrom.items():
        vals = sorted(vals)
        index[chrom] = {
            "starts": np.asarray([x[0] for x in vals], dtype=np.int64),
            "ids": np.asarray([x[1] for x in vals], dtype=object),
        }
    return pd.DataFrame(rows), index


def write_bed(variant_df, path):
    with gzip.open(path, "wt") as handle:
        for row in variant_df.itertuples(index=False):
            handle.write("%s\t%d\t%d\t%s\n" % (row.chromosome, int(row.start), int(row.end), row.variant_id))


def init_worker(index, bed_path):
    global VARIANT_INDEX, BED_PATH
    VARIANT_INDEX = index
    BED_PATH = str(bed_path)


def map_interval_to_variants(chrom, start, end):
    if chrom not in VARIANT_INDEX:
        return []
    starts = VARIANT_INDEX[chrom]["starts"]
    ids = VARIANT_INDEX[chrom]["ids"]
    lo = int(np.searchsorted(starts, int(start), side="left"))
    hi = int(np.searchsorted(starts, int(end), side="left"))
    if hi <= lo:
        return []
    return ids[lo:hi]


def query_one_track(task):
    track_path, category = task
    counts = Counter()
    seen_records = set()
    stderr_text = ""
    return_code = 0
    n_records = 0
    n_variant_hits = 0
    try:
        proc = subprocess.Popen(
            ["tabix", "-R", BED_PATH, track_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line in seen_records:
                continue
            seen_records.add(line)
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            try:
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                continue
            n_records += 1
            for variant_id in map_interval_to_variants(chrom, start, end):
                counts[(str(variant_id), category)] += 1
                n_variant_hits += 1
        stderr_text = proc.stderr.read()
        return_code = proc.wait()
    except Exception as exc:
        return_code = 99
        stderr_text = repr(exc)
    return {
        "track_path": track_path,
        "category": category,
        "return_code": return_code,
        "stderr": stderr_text.strip(),
        "n_unique_track_records": n_records,
        "n_variant_hits": n_variant_hits,
        "counts": counts,
    }


def build_track_tasks():
    by_file, track_map = load_filer_file_category_map()
    meta = pd.read_csv(FILER_METADATA, sep="\t", dtype=str)
    tasks = []
    report_rows = []
    for _, row in meta.iterrows():
        file_name = row["File name"]
        category = by_file[file_name]
        track_path = str(Path(row["filepath"]) / file_name)
        exists = Path(track_path).exists()
        tbi_exists = Path(track_path + ".tbi").exists()
        report_rows.append(
            {
                "track_path": track_path,
                "file_name": file_name,
                "category": category,
                "exists": int(exists),
                "tbi_exists": int(tbi_exists),
            }
        )
        if exists and tbi_exists:
            tasks.append((track_path, category))
    return tasks, pd.DataFrame(report_rows), track_map


def empty_counts_for_variants(variant_df):
    rows = []
    for row in variant_df.itertuples(index=False):
        rec = {
            "variant_id": row.variant_id,
            "chromosome": row.chromosome,
            "position": int(row.position),
            "ref_allele": row.ref_allele,
            "alt_allele": row.alt_allele,
        }
        for cat in CATEGORIES:
            rec["filer_raw_count_" + cat] = 0
            rec["filer_binary_" + cat] = 0
        rec["filer_any_overlap"] = 0
        rows.append(rec)
    return pd.DataFrame(rows)


def write_counts(variant_df, aggregate_counts):
    rows = []
    for _, row in variant_df.iterrows():
        rec = {
            "variant_id": row["variant_id"],
            "chromosome": row["chromosome"],
            "position": int(row["position"]),
            "ref_allele": row["ref_allele"],
            "alt_allele": row["alt_allele"],
        }
        any_overlap = 0
        for cat in CATEGORIES:
            n = int(aggregate_counts.get((row["variant_id"], cat), 0))
            rec["filer_raw_count_" + cat] = n
            rec["filer_binary_" + cat] = int(n > 0)
            any_overlap = max(any_overlap, int(n > 0))
        rec["filer_any_overlap"] = any_overlap
        rows.append(rec)
    return pd.DataFrame(rows)


def load_cached_counts():
    if not CACHED_VARIANT_COUNTS.exists():
        return pd.DataFrame(columns=["variant_id"])
    prev = pd.read_csv(CACHED_VARIANT_COUNTS, sep="\t", compression="gzip")
    if prev["variant_id"].duplicated().any():
        raise RuntimeError("Cached variant_filer_category_counts has duplicate variant_id rows")
    keep_cols = ["variant_id"] + ["filer_raw_count_" + c for c in CATEGORIES] + ["filer_binary_" + c for c in CATEGORIES] + ["filer_any_overlap"]
    return prev[keep_cols]


def merge_reused_and_queried(variant_df, queried_counts):
    prev = load_cached_counts()
    current_ids = set(variant_df["variant_id"].astype(str))
    cached_ids = set(prev["variant_id"].astype(str))
    reused_ids = current_ids & cached_ids

    current_meta = variant_df[["variant_id", "chromosome", "position", "ref_allele", "alt_allele"]].copy()
    reused = current_meta[current_meta["variant_id"].isin(reused_ids)].merge(prev, on="variant_id", how="left", validate="one_to_one")
    queried = queried_counts.copy()
    for frame, source in [(reused, "reused_cached_exact_variant_id"), (queried, "queried_tabix_new_variant")]:
        if len(frame):
            frame["annotation_source"] = source
    final = pd.concat([reused, queried], ignore_index=True, sort=False)
    final = current_meta[["variant_id"]].merge(final, on="variant_id", how="left", validate="one_to_one")
    for col in [c for c in final.columns if c.endswith("_x")]:
        base = col[:-2]
        y = base + "_y"
        if y in final.columns:
            final[base] = final[col].where(final[col].notna(), final[y])
            final = final.drop(columns=[col, y])
    count_cols = ["filer_raw_count_" + c for c in CATEGORIES] + ["filer_binary_" + c for c in CATEGORIES] + ["filer_any_overlap"]
    for col in count_cols:
        final[col] = pd.to_numeric(final[col], errors="raise").astype(int)
    final["annotation_source"] = final["annotation_source"].fillna("missing")
    if (final["annotation_source"] == "missing").any():
        raise RuntimeError("Some current union variants lack reused or queried annotations")
    return final


def validate_against_positive_only(counts_df, out_dir):
    ref = pd.read_csv(POSITIVE_ONLY_COUNTS, sep="\t")
    ref_cols = ["filer_unique_count_" + c for c in CATEGORIES]
    ref_variant = ref.groupby("variant_id")[ref_cols].max().reset_index()
    merged = ref_variant.merge(counts_df, on="variant_id", how="inner")
    rows = []
    mismatches = []
    for _, row in merged.iterrows():
        for cat in CATEGORIES:
            expected = int(row["filer_unique_count_" + cat]) > 0
            observed = int(row["filer_binary_" + cat]) > 0
            ok = expected == observed
            rows.append(
                {
                    "variant_id": row["variant_id"],
                    "category": cat,
                    "expected_binary": int(expected),
                    "observed_binary": int(observed),
                    "status": "PASS" if ok else "FAIL",
                }
            )
            if not ok:
                mismatches.append((row["variant_id"], cat, int(expected), int(observed)))
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out_dir / "logs" / "annotation_agreement.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {
                "overlapping_positive_variants": int(merged["variant_id"].nunique()),
                "category_comparisons": int(len(agreement)),
                "mismatches": int(len(mismatches)),
                "status": "PASS" if not mismatches else "FAIL",
            }
        ]
    )
    summary.to_csv(out_dir / "logs" / "annotation_agreement_summary.tsv", sep="\t", index=False)
    return mismatches, summary


def main():
    parser = argparse.ArgumentParser(description="FILER annotation query: exact-variant reuse from the annotation cache and tabix for new variants")
    parser.add_argument("--bed", default=str(BASE_DIR / "union_variants.bed.gz"))
    parser.add_argument("--out-dir", default=str(BASE_DIR / "filer_union_variants_tabix_FILER"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    variant_df, _ = load_union_bed(args.bed)
    cached_counts = load_cached_counts()
    cached_ids = set(cached_counts["variant_id"].astype(str))
    query_df = variant_df[~variant_df["variant_id"].isin(cached_ids)].copy().reset_index(drop=True)
    reused_n = int(len(variant_df) - len(query_df))
    query_bed = out_dir / "new_variants_to_query.bed.gz"
    write_bed(query_df, query_bed)

    tasks, track_report, track_map = build_track_tasks()
    track_report.to_csv(out_dir / "logs" / "filer_tabix_track_inputs.tsv", sep="\t", index=False)
    track_map.to_csv(out_dir / "logs" / "filer_category_track_map.tsv", sep="\t", index=False)
    missing = track_report[(track_report["exists"] == 0) | (track_report["tbi_exists"] == 0)]
    if len(missing):
        missing.to_csv(out_dir / "logs" / "missing_track_files.tsv", sep="\t", index=False)
        raise SystemExit("Missing FILER track files or tabix indexes; stopping")

    aggregate = Counter()
    query_rows = []
    if len(query_df):
        _, query_index = load_union_bed(query_bed)
        print("[query] %d current variants, %d reused cached annotations, %d new variants, %d tracks, %d jobs" % (len(variant_df), reused_n, len(query_df), len(tasks), args.jobs), flush=True)
        with Pool(processes=args.jobs, initializer=init_worker, initargs=(query_index, query_bed)) as pool:
            for i, result in enumerate(pool.imap_unordered(query_one_track, tasks), start=1):
                aggregate.update(result.pop("counts"))
                query_rows.append(result)
                if i % 100 == 0:
                    print("[query] completed %d / %d tracks" % (i, len(tasks)), flush=True)
        queried_counts = write_counts(query_df, aggregate)
    else:
        print("[query] %d current variants, all reused from cached exact annotations" % len(variant_df), flush=True)
        queried_counts = empty_counts_for_variants(query_df)

    query_report = pd.DataFrame(query_rows)
    query_report.to_csv(out_dir / "logs" / "filer_tabix_query_report.tsv", sep="\t", index=False)
    if len(query_report):
        failures = query_report[query_report["return_code"] != 0]
        if len(failures):
            failures.to_csv(out_dir / "logs" / "filer_tabix_query_failures.tsv", sep="\t", index=False)
            raise SystemExit("One or more tabix track queries failed; stopping")

    counts_df = merge_reused_and_queried(variant_df, queried_counts)
    counts_df.to_csv(BASE_DIR / "variant_filer_category_counts.tsv.gz", sep="\t", index=False, compression="gzip")
    source_summary = counts_df.groupby("annotation_source").size().reset_index(name="variants")
    source_summary["current_union_variants"] = int(len(variant_df))
    source_summary["tracks_checked"] = int(len(tasks))
    source_summary.to_csv(out_dir / "logs" / "filer_annotation_reuse_summary.tsv", sep="\t", index=False)

    mismatches, summary = validate_against_positive_only(counts_df, out_dir)
    print(summary.to_string(index=False), flush=True)
    if mismatches:
        print("First mismatches:", mismatches[:20], flush=True)
        raise SystemExit("FILER positive-annotation binary agreement failed; stopping inference")
    print("[PASS] FILER positive-annotation binary agreement", flush=True)


if __name__ == "__main__":
    main()
