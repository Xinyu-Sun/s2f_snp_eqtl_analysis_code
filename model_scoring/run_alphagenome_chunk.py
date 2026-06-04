#!/usr/bin/env python3
"""
Run AlphaGenome API scoring for one redo SuSiE chunk.

Outputs (relative to CWD):
  results/alphagenome/{ancestry}/chunk_####/
    - tidy.full.tsv
    - tidy.whole_blood.tsv
    - tidy.whole_blood.requested_pairs.tsv
    - errors.tsv (if any variants failed)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


WHOLE_BLOOD_CURIE = "UBERON:0013756"
WHOLE_BLOOD_TRACK = "UBERON:0013756 gtex Whole_Blood polyA plus RNA-seq"


def _safe_slug(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace("/", "_")
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("<", "lt")
    )


def _read_job(jobs_tsv: Path, tool: str, one_based_task_id: int) -> dict[str, Any]:
    jobs = pd.read_csv(jobs_tsv, sep="\t")
    jobs = jobs[jobs["tool"] == tool].reset_index(drop=True)
    if one_based_task_id < 1 or one_based_task_id > len(jobs):
        raise ValueError(f"task_id {one_based_task_id} out of range 1..{len(jobs)} for tool={tool}")
    return jobs.iloc[one_based_task_id - 1].to_dict()


def _parse_variant_id(variant_id: str) -> tuple[str, int, str, str]:
    chrom, pos, refalt = str(variant_id).split(":")
    ref, alt = refalt.split(">")
    return chrom, int(pos), ref, alt


def _with_retries(fn, *, max_attempts: int = 5, base_sleep_s: float = 1.0, max_sleep_s: float = 60.0):
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception:
            if attempt >= max_attempts:
                raise
            sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            time.sleep(sleep_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_tsv", default="results/manifests/jobs.tsv")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument(
        "--task_id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "1")),
        help="1-based task id (defaults to $SLURM_ARRAY_TASK_ID or 1).",
    )
    ap.add_argument("--seed_override", type=int, default=None)
    ap.add_argument("--output_type", default="RNA_SEQ")
    ap.add_argument("--whole_blood_curie", default=WHOLE_BLOOD_CURIE)
    ap.add_argument("--whole_blood_track", default=WHOLE_BLOOD_TRACK)
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--tidy_batch_size", type=int, default=25)
    args = ap.parse_args()

    api_key = os.environ.get("ALPHAGENOME_API")
    if not api_key:
        raise RuntimeError("ALPHAGENOME_API env var is not set")

    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    cwd = Path.cwd()
    jobs_tsv = (cwd / args.jobs_tsv).resolve()
    job = _read_job(jobs_tsv, tool="alphagenome", one_based_task_id=args.task_id)

    ancestry = str(job["ancestry"])
    chunk_index = int(job["chunk_index"])
    seed = int(job.get("seed", -1))
    if args.seed_override is not None:
        seed = int(args.seed_override)

    pairs_tsv = Path(str(job["pairs_tsv"]))

    out_dir = (
        cwd
        / args.results_dir
        / "alphagenome"
        / _safe_slug(ancestry)
        / f"chunk_{chunk_index:04d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "started.txt").write_text(
        f"started_epoch={int(time.time())}\n"
        f"task_id={args.task_id}\n"
        f"ancestry={ancestry}\n"
        f"chunk_index={chunk_index}\n"
        f"seed={seed}\n"
    )

    df_pairs = pd.read_csv(
        pairs_tsv,
        sep="\t",
        dtype={"chromosome": str, "position": "int64", "ref_allele": str, "alt_allele": str, "gene_id": str},
    )

    df_vars = (
        df_pairs[["chromosome", "position", "ref_allele", "alt_allele"]]
        .drop_duplicates()
        .sort_values(["chromosome", "position", "ref_allele", "alt_allele"], kind="mergesort")
        .reset_index(drop=True)
    )

    print(
        f"[alphagenome] task_id={args.task_id} ancestry={ancestry} "
        f"chunk={chunk_index} pairs={len(df_pairs):,} unique_variants={len(df_vars):,}",
        flush=True,
    )

    dna_model = dna_client.create(api_key)
    variant_scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS[args.output_type]

    errors = []
    n_success = 0

    tidy_full_path = out_dir / "tidy.full.tsv"
    tidy_wb_path = out_dir / "tidy.whole_blood.tsv"
    tidy_joined_path = out_dir / "tidy.whole_blood.requested_pairs.tsv"
    tidy_final_path = out_dir / "tidy.whole_blood.requested_pairs.final.tsv"

    for p in [tidy_full_path, tidy_wb_path, tidy_joined_path, tidy_final_path]:
        if p.exists():
            p.unlink()

    t0 = time.time()
    batch_scores = []
    wrote_full_header = False
    wrote_wb_header = False
    wrote_joined_header = False
    wrote_final_header = False

    def flush_batch() -> None:
        nonlocal wrote_full_header, wrote_wb_header, wrote_joined_header, wrote_final_header
        nonlocal batch_scores
        if not batch_scores:
            return

        df_tidy = variant_scorers.tidy_scores(batch_scores, match_gene_strand=True)
        batch_scores = []

        if df_tidy is None or len(df_tidy) == 0:
            return

        df_tidy.to_csv(
            tidy_full_path, sep="\t", index=False, mode="a", header=(not wrote_full_header),
        )
        wrote_full_header = True

        mask = False
        if "ontology_curie" in df_tidy.columns:
            mask = mask | (df_tidy["ontology_curie"] == args.whole_blood_curie)
        if "track_name" in df_tidy.columns:
            mask = mask | (df_tidy["track_name"] == args.whole_blood_track)
        df_wb = df_tidy[mask].copy()
        df_wb.to_csv(
            tidy_wb_path, sep="\t", index=False, mode="a", header=(not wrote_wb_header),
        )
        wrote_wb_header = wrote_wb_header or (len(df_wb) > 0)

        if len(df_wb) == 0:
            return

        parsed = df_wb["variant_id"].map(_parse_variant_id)
        df_wb["chromosome"] = [p[0] for p in parsed]
        df_wb["position"] = [p[1] for p in parsed]
        df_wb["ref_allele"] = [p[2] for p in parsed]
        df_wb["alt_allele"] = [p[3] for p in parsed]

        join_cols = ["chromosome", "position", "ref_allele", "alt_allele", "gene_id"]
        df_joined = df_wb.merge(df_pairs, on=join_cols, how="inner", suffixes=("", "_eqtl"))
        df_joined.to_csv(
            tidy_joined_path, sep="\t", index=False, mode="a", header=(not wrote_joined_header),
        )
        wrote_joined_header = wrote_joined_header or (len(df_joined) > 0)

        keep_cols = []
        for c in [
            "variant_id", "chromosome", "position", "ref_allele", "alt_allele",
            "gene_id", "gene_name", "gene_type", "gene_strand",
            "output_type", "variant_scorer", "track_name", "track_strand",
            "ontology_curie", "gtex_tissue", "raw_score", "quantile_score",
        ]:
            if c in df_joined.columns:
                keep_cols.append(c)
        for c in ["ancestry", "tss_distance_bin", "population", "pvalue", "beta", "std_error", "qvalue", "pip", "CS"]:
            if c in df_joined.columns and c not in keep_cols:
                keep_cols.append(c)

        df_final = df_joined[keep_cols].copy()
        df_final.insert(0, "tool", "alphagenome")
        df_final.insert(1, "seed", seed)
        df_final.insert(2, "chunk_index", chunk_index)
        df_final.insert(3, "n_pairs_in_chunk", int(job.get("n_pairs", -1)))
        df_final.insert(4, "n_unique_variants_in_chunk", int(job.get("n_unique_variants", -1)))
        df_final.to_csv(
            tidy_final_path, sep="\t", index=False, mode="a", header=(not wrote_final_header),
        )
        wrote_final_header = wrote_final_header or (len(df_final) > 0)

    for i, row in enumerate(df_vars.itertuples(index=False), start=1):
        if i == 1 or i % 25 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"[alphagenome] progress {i}/{len(df_vars)} variants (rate={rate:.2f}/s)", flush=True)
        
        v = genome.Variant(
            chromosome=str(row.chromosome),
            position=int(row.position),
            reference_bases=str(row.ref_allele),
            alternate_bases=str(row.alt_allele),
        )
        interval = v.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

        def _do_score():
            return dna_model.score_variant(interval=interval, variant=v, variant_scorers=[variant_scorer])

        try:
            scores = _with_retries(_do_score, max_attempts=args.max_attempts)
            batch_scores.append(scores)
            n_success += 1
        except Exception as e:
            errors.append({
                "chromosome": row.chromosome,
                "position": int(row.position),
                "ref_allele": row.ref_allele,
                "alt_allele": row.alt_allele,
                "error": repr(e),
            })
        
        if len(batch_scores) >= args.tidy_batch_size:
            print(f"[alphagenome] tidying/writing batch of {len(batch_scores)} variants", flush=True)
            flush_batch()

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "errors.tsv", sep="\t", index=False)
        print(f"[alphagenome] wrote errors.tsv with {len(errors)} failures", flush=True)

    if batch_scores:
        print(f"[alphagenome] tidying/writing final batch of {len(batch_scores)} variants", flush=True)
        flush_batch()

    if n_success == 0:
        tidy_full_path.write_text("")
        tidy_wb_path.write_text("")
        tidy_joined_path.write_text("")
        tidy_final_path.write_text("")
        print(f"No tidy outputs in {out_dir}", flush=True)
        return

    print(f"Wrote: {tidy_final_path}", flush=True)


if __name__ == "__main__":
    main()
