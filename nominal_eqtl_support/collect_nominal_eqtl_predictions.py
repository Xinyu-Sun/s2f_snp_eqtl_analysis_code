#!/usr/bin/env python3
"""Collect top-up chunk outputs while preserving nominal-redo metadata columns."""

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path


def topup_root():
    return Path(__file__).resolve().parents[1]


def safe_slug(s):
    return str(s).strip().replace("/", "_").replace(" ", "_").replace(">", "gt").replace("<", "lt")


def read_jobs(jobs_tsv):
    with jobs_tsv.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def chunk_file_for(group_dir, tool, chunk_index):
    if tool == "borzoi":
        return group_dir / "chunk_{:04d}".format(chunk_index) / "sed.blood.requested_pairs.tsv"
    return group_dir / "chunk_{:04d}".format(chunk_index) / "tidy.whole_blood.requested_pairs.tsv"


def output_name_for(tool):
    if tool == "borzoi":
        return "borzoi.blood.requested_pairs.tsv.gz"
    return "alphagenome.whole_blood.requested_pairs.tsv.gz"


def collect_group(root, tool, population_group, tss_bin, jobs):
    group_dir = root / "outputs" / tool / safe_slug(population_group) / safe_slug(tss_bin)
    out_path = group_dir / output_name_for(tool)
    group_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    out_f = None
    n_rows = 0
    missing = 0
    fieldnames = None

    try:
        out_f = gzip.open(str(out_path), "wt", newline="")
        for job in sorted(jobs, key=lambda j: int(j["chunk_index"])):
            chunk_index = int(job["chunk_index"])
            chunk_path = chunk_file_for(group_dir, tool, chunk_index)
            if not chunk_path.exists():
                missing += 1
                continue

            with chunk_path.open(newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if reader.fieldnames is None:
                    continue
                prefix = [
                    "tool",
                    "seed",
                    "chunk_index",
                    "n_pairs_in_chunk",
                    "n_unique_variants_in_chunk",
                    "population_group",
                ]
                current_fields = prefix + [c for c in reader.fieldnames if c not in prefix]
                if writer is None:
                    fieldnames = current_fields
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
                    writer.writeheader()
                elif current_fields != fieldnames:
                    raise RuntimeError(
                        "Column mismatch in {}. Expected {}, observed {}".format(
                            chunk_path, fieldnames, current_fields
                        )
                    )

                for row in reader:
                    if row.get("gene_id") == "gene_id":
                        continue
                    row["tool"] = tool
                    row["seed"] = job.get("seed", "")
                    row["chunk_index"] = chunk_index
                    row["n_pairs_in_chunk"] = job.get("n_pairs", "")
                    row["n_unique_variants_in_chunk"] = job.get("n_unique_variants", "")
                    row["population_group"] = population_group
                    writer.writerow(row)
                    n_rows += 1
    finally:
        if out_f is not None:
            out_f.close()

    if writer is None:
        if out_path.exists():
            out_path.unlink()
        print("[SKIP] {} {} {}: no chunk outputs found (missing_chunks={})".format(tool, population_group, tss_bin, missing))
        return

    print("[OK] {} {} {}: wrote {:,} rows -> {} (missing_chunks={})".format(tool, population_group, tss_bin, n_rows, out_path, missing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_tsv", default=str(topup_root() / "outputs/manifests/jobs.tsv"))
    ap.add_argument("--tool", choices=["borzoi", "alphagenome", "all"], default="all")
    args = ap.parse_args()

    root = topup_root()
    jobs_tsv = Path(args.jobs_tsv).resolve()
    jobs = read_jobs(jobs_tsv)
    if args.tool != "all":
        jobs = [j for j in jobs if j["tool"] == args.tool]
    if not jobs:
        raise RuntimeError("No jobs found to collect")

    groups = defaultdict(list)
    for job in jobs:
        groups[(job["tool"], job["population_group"], job["tss_distance_bin"])].append(job)

    for (tool, population_group, tss_bin), group_jobs in sorted(groups.items()):
        collect_group(root, tool, population_group, tss_bin, group_jobs)


if __name__ == "__main__":
    main()
