#!/usr/bin/env python3
"""
Prepare only the incremental nominal-redo records needed to extend the benchmark
from 5,000 to 10,000 records per population x TSS-distance cell.

This intentionally anchors the completed 5k run: all records already present in
nominal_eqtl_redo/outputs/inputs are treated as selected and are not re-run.
Additional records are sampled from the remaining eligible records up to the new
cell cap.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_REDO_ROOT = PROJECT_ROOT / "nominal_eqtl_redo"
BASE_SCRIPT_DIR = BASE_REDO_ROOT / "scripts"
sys.path.insert(0, str(BASE_SCRIPT_DIR))

import prepare_nominal_redo_batches as base  # noqa: E402


def topup_root():
    return Path(__file__).resolve().parents[1]


def read_existing_selected_ids(base_redo_root):
    selected = set()
    pairs_paths = sorted((base_redo_root / "outputs" / "inputs" / "borzoi").glob("*/*/chunk_*/pairs.tsv"))
    if not pairs_paths:
        raise RuntimeError("No existing 5k selected pairs found under {}".format(base_redo_root))

    for path in pairs_paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rid = row.get("nominal_redo_record_id")
                if rid:
                    selected.add(rid)
    return selected


def select_topup_rows(rows, existing_ids, seed, max_pairs_per_cell):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["population"], row["tss_distance_bin"])].append(row)

    topup = []
    final_selected_ids = set(existing_ids)

    for population in base.POPULATIONS:
        for tss_bin in base.TSS_BINS:
            cell = grouped.get((population, tss_bin), [])
            target_n = min(max_pairs_per_cell, len(cell))
            old_in_cell = [row for row in cell if row["nominal_redo_record_id"] in existing_ids]
            n_needed = max(0, target_n - len(old_in_cell))
            if n_needed == 0:
                continue

            candidates = [row for row in cell if row["nominal_redo_record_id"] not in existing_ids]
            if len(candidates) < n_needed:
                raise RuntimeError(
                    "Not enough top-up candidates for {} {}: need {}, have {}".format(
                        population, tss_bin, n_needed, len(candidates)
                    )
                )

            rng = random.Random(base.stable_cell_seed(seed, population, tss_bin))
            picked_indices = sorted(rng.sample(range(len(candidates)), n_needed))
            picked = [candidates[i] for i in picked_indices]
            topup.extend(picked)
            final_selected_ids.update(row["nominal_redo_record_id"] for row in picked)

    topup.sort(key=base.sort_key)
    return topup, final_selected_ids


def count_existing_by_cell(rows, existing_ids):
    out = []
    for row in rows:
        if row["nominal_redo_record_id"] not in existing_ids:
            continue
        out.append(row)
    return base.count_by(out, ["population", "tss_distance_bin"])


def prepare(args):
    root = PROJECT_ROOT
    out_root = topup_root() / "outputs"
    inputs_root = out_root / "inputs"
    manifests_root = out_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    eqtl_path = (root / args.eqtls).resolve()
    rows, n_filtered_before_dedup = base.read_eqtl_rows(eqtl_path)
    existing_ids = read_existing_selected_ids(BASE_REDO_ROOT)
    topup_rows, final_selected_ids = select_topup_rows(
        rows,
        existing_ids,
        seed=args.seed,
        max_pairs_per_cell=args.max_pairs_per_cell,
    )

    final_rows = [row for row in rows if row["nominal_redo_record_id"] in final_selected_ids]
    final_rows.sort(key=base.sort_key)

    (manifests_root / "seed.txt").write_text("{}\n".format(args.seed))
    (manifests_root / "sampling_design.txt").write_text(
        "population_level_nominal_eqtls_10k_topup_reusing_5k_predictions\n"
        "eqtls={}\n".format(eqtl_path)
        + "base_redo_root={}\n".format(BASE_REDO_ROOT)
        + "populations={}\n".format(",".join(base.POPULATIONS))
        + "tss_bins={}\n".format(",".join(base.TSS_BINS))
        + "max_pairs_per_population_tss_cell={}\n".format(args.max_pairs_per_cell)
        + "pairs_per_chunk={}\n".format(args.pairs_per_chunk)
        + "topup_rule=preserve_existing_5k_sample_then_sample_additional_records_from_remaining_eligible_records\n"
    )

    base.write_tsv(
        manifests_root / "input_counts_by_population_tss.tsv",
        ["population", "tss_distance_bin", "record_count"],
        base.count_by(rows, ["population", "tss_distance_bin"]),
    )
    base.write_tsv(
        manifests_root / "existing_5k_counts_by_population_tss.tsv",
        ["population", "tss_distance_bin", "record_count"],
        count_existing_by_cell(rows, existing_ids),
    )
    base.write_tsv(
        manifests_root / "topup_counts_by_population_tss.tsv",
        ["population", "tss_distance_bin", "record_count"],
        base.count_by(topup_rows, ["population", "tss_distance_bin"]),
    )
    base.write_tsv(
        manifests_root / "final_10k_counts_by_population_tss.tsv",
        ["population", "tss_distance_bin", "record_count"],
        base.count_by(final_rows, ["population", "tss_distance_bin"]),
    )
    base.write_tsv(
        manifests_root / "topup_counts_by_population_sharing_class.tsv",
        ["population", "sharing_class", "record_count"],
        base.count_by(topup_rows, ["population", "sharing_class"]),
    )

    job_rows = []
    topup_by_cell = defaultdict(list)
    for row in topup_rows:
        topup_by_cell[(row["population"], row["tss_distance_bin"])].append(row)

    for population in base.POPULATIONS:
        for tss_bin in base.TSS_BINS:
            cell_rows = topup_by_cell.get((population, tss_bin), [])
            if not cell_rows:
                continue

            for chunk_index, chunk_rows in enumerate(base.chunks(cell_rows, args.pairs_per_chunk)):
                for row in chunk_rows:
                    row["vcf_id"] = base.variant_id(
                        row["chromosome"],
                        row["position"],
                        row["ref_allele"],
                        row["alt_allele"],
                    )

                for tool in base.TOOLS:
                    chunk_dir = (
                        inputs_root
                        / tool
                        / base.safe_slug(population)
                        / base.safe_slug(tss_bin)
                        / "chunk_{:04d}".format(chunk_index)
                    )
                    pairs_path = chunk_dir / "pairs.tsv"
                    vcf_path = chunk_dir / "variants.vcf"
                    manifest_path = chunk_dir / "pairs_manifest.tsv"

                    base.write_tsv(pairs_path, base.PAIR_COLUMNS + ["vcf_id"], chunk_rows)
                    n_unique_variants = base.write_vcf(vcf_path, chunk_rows)
                    base.write_tsv(
                        manifest_path,
                        [
                            "nominal_redo_record_id",
                            "chromosome",
                            "position",
                            "ref_allele",
                            "alt_allele",
                            "vcf_id",
                            "gene_id",
                            "population",
                            "ancestry_sharing",
                            "sharing_class",
                            "tss_distance_bin",
                        ],
                        chunk_rows,
                    )

                    job_rows.append(
                        {
                            "tool": tool,
                            "ancestry_sharing": population,
                            "population_group": population,
                            "tss_distance_bin": tss_bin,
                            "chunk_index": chunk_index,
                            "pairs_tsv": str(pairs_path.resolve()),
                            "variants_vcf": str(vcf_path.resolve()),
                            "pairs_manifest_tsv": str(manifest_path.resolve()),
                            "n_pairs": len(chunk_rows),
                            "n_unique_variants": n_unique_variants,
                            "seed": args.seed,
                            "max_pairs_per_cell": args.max_pairs_per_cell,
                            "pairs_per_chunk": args.pairs_per_chunk,
                            "sample_design": "population_x_tss_10k_topup_reusing_5k_predictions",
                        }
                    )

    job_fields = [
        "tool",
        "ancestry_sharing",
        "population_group",
        "tss_distance_bin",
        "chunk_index",
        "pairs_tsv",
        "variants_vcf",
        "pairs_manifest_tsv",
        "n_pairs",
        "n_unique_variants",
        "seed",
        "max_pairs_per_cell",
        "pairs_per_chunk",
        "sample_design",
    ]
    base.write_tsv(manifests_root / "jobs.tsv", job_fields, job_rows)
    base.write_tsv(
        manifests_root / "job_counts_by_tool.tsv",
        ["tool", "record_count"],
        base.count_by(job_rows, ["tool"]),
    )
    base.write_tsv(
        manifests_root / "prepare_summary.tsv",
        ["metric", "value"],
        [
            {"metric": "filtered_rows_before_exact_dedup", "value": n_filtered_before_dedup},
            {"metric": "filtered_rows_after_exact_dedup", "value": len(rows)},
            {"metric": "existing_5k_selected_rows", "value": len(existing_ids)},
            {"metric": "topup_rows", "value": len(topup_rows)},
            {"metric": "final_10k_selected_rows", "value": len(final_rows)},
            {"metric": "job_rows_total", "value": len(job_rows)},
            {"metric": "job_rows_per_tool", "value": len(job_rows) // len(base.TOOLS) if base.TOOLS else 0},
        ],
    )

    print("Input rows before exact de-duplication: {:,}".format(n_filtered_before_dedup))
    print("Input rows after exact de-duplication:  {:,}".format(len(rows)))
    print("Existing 5k selected rows:             {:,}".format(len(existing_ids)))
    print("Top-up rows:                           {:,}".format(len(topup_rows)))
    print("Final selected rows after top-up:       {:,}".format(len(final_rows)))
    print("Job rows:                              {:,}".format(len(job_rows)))
    print("Wrote manifest: {}".format(manifests_root / "jobs.tsv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqtls", default="processed_inputs/processed_eqtls.tsv.gz")
    ap.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    ap.add_argument("--max_pairs_per_cell", type=int, default=10000)
    ap.add_argument("--pairs_per_chunk", type=int, default=1000)
    args = ap.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
