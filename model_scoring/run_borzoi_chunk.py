#!/usr/bin/env python3
"""
Run Borzoi SED scoring for one fine-mapped eQTL scoring chunk.

Outputs (relative to CWD):
  results/borzoi/{ancestry}/chunk_####/
    - sed.full.tsv
    - sed.blood.tsv
    - sed.blood.requested_pairs.tsv
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def _safe_slug(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace("/", "_")
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("<", "lt")
    )


def _read_job(jobs_tsv: Path, tool: str, one_based_task_id: int) -> dict:
    jobs = pd.read_csv(jobs_tsv, sep="\t")
    jobs = jobs[jobs["tool"] == tool].reset_index(drop=True)
    if one_based_task_id < 1 or one_based_task_id > len(jobs):
        raise ValueError(f"task_id {one_based_task_id} out of range 1..{len(jobs)} for tool={tool}")
    row = jobs.iloc[one_based_task_id - 1].to_dict()
    return row


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
    ap.add_argument(
        "--borzoi_sed_path",
        default="/mnt/vstor/Data14/xxs410_xinyu/seq2func_related/borzoi_related/borzoi/src/scripts/borzoi_sed.py",
    )
    ap.add_argument(
        "--params_pred_json",
        default="/mnt/vstor/Data14/xxs410_xinyu/seq2func_related/borzoi_related/borzoi/examples/params_pred.json",
    )
    ap.add_argument(
        "--targets_gtex",
        default="/mnt/vstor/Data14/xxs410_xinyu/seq2func_related/borzoi_related/borzoi/examples/targets_gtex.txt",
    )
    ap.add_argument(
        "--model_h5",
        default="/mnt/vstor/Data14/xxs410_xinyu/seq2func_related/borzoi_related/borzoi/tutorials/latest/analyze_sv/data/model/f3/model0_best.h5",
    )
    ap.add_argument("--stats", default="logSED,logD2,logMeanSED")
    ap.add_argument(
        "--skip_borzoi",
        action="store_true",
        help="Skip running Borzoi if sed.tsv already exists; only postprocess.",
    )
    args = ap.parse_args()

    cwd = Path.cwd()
    jobs_tsv = (cwd / args.jobs_tsv).resolve()
    job = _read_job(jobs_tsv, tool="borzoi", one_based_task_id=args.task_id)

    ancestry = str(job["ancestry"])
    chunk_index = int(job["chunk_index"])
    seed = int(job.get("seed", -1))

    pairs_tsv = Path(str(job["pairs_tsv"]))
    vcf_path = Path(str(job["variants_vcf"]))

    out_dir = cwd / args.results_dir / "borzoi" / _safe_slug(ancestry) / f"chunk_{chunk_index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Borzoi writes sed.tsv in out_dir
    sed_tsv = out_dir / "sed.tsv"
    if not (args.skip_borzoi and sed_tsv.exists()):
        cmd = [
            str(Path(args.borzoi_sed_path)),
            "--tsv_only",
            "-o",
            str(out_dir),
            "--rc",
            "--stats",
            args.stats,
            "-t",
            str(Path(args.targets_gtex)),
            str(Path(args.params_pred_json)),
            str(Path(args.model_h5)),
            str(vcf_path),
        ]
        subprocess.run(cmd, check=True)
    
    if not sed_tsv.exists():
        raise FileNotFoundError(f"Expected Borzoi output {sed_tsv} not found")

    sed_full = out_dir / "sed.full.tsv"
    shutil.copyfile(sed_tsv, sed_full)

    df_sed = pd.read_csv(
        sed_tsv,
        sep="\t",
        dtype={"chr": str, "pos": "int64", "ref": str, "alt": str, "gene_id": str},
    )

    # Normalize gene IDs (remove version)
    if "gene_id" in df_sed.columns:
        df_sed["gene_id_versioned"] = df_sed["gene_id"].astype(str)
        df_sed["gene_id"] = df_sed["gene_id"].astype(str).str.replace(r"\..*$", "", regex=True)

    # Filter tissue == blood
    if "tissue" not in df_sed.columns:
        raise ValueError(f"sed.tsv missing 'tissue' column. Columns: {list(df_sed.columns)}")
    df_blood = df_sed[df_sed["tissue"] == "blood"].copy()
    blood_path = out_dir / "sed.blood.tsv"
    df_blood.to_csv(blood_path, sep="\t", index=False)

    # Join to requested pairs
    df_pairs = pd.read_csv(
        pairs_tsv,
        sep="\t",
        dtype={"chromosome": str, "position": "int64", "ref_allele": str, "alt_allele": str, "gene_id": str},
    )

    df_pairs_key = df_pairs.rename(
        columns={
            "chromosome": "chr",
            "position": "pos",
            "ref_allele": "ref",
            "alt_allele": "alt",
        }
    )

    join_cols = ["chr", "pos", "ref", "alt", "gene_id"]
    df_joined = df_blood.merge(df_pairs_key, on=join_cols, how="inner", suffixes=("", "_eqtl"))
    joined_path = out_dir / "sed.blood.requested_pairs.tsv"
    df_joined.to_csv(joined_path, sep="\t", index=False)

    # Final compact table
    stats_cols = [c for c in args.stats.split(",") if c in df_joined.columns]
    core_cols = ["chr", "pos", "ref", "alt", "gene_id", "tissue"]
    eqtl_cols = [c for c in ["ancestry", "tss_distance_bin", "population", "pvalue", "beta", "std_error", "qvalue", "pip", "CS"] if c in df_joined.columns]
    final_cols = core_cols + stats_cols + eqtl_cols
    final = df_joined[final_cols].copy()
    final.insert(0, "tool", "borzoi")
    final.insert(1, "seed", seed)
    final.insert(2, "chunk_index", chunk_index)
    final.insert(3, "n_pairs_in_chunk", int(job.get("n_pairs", -1)))
    final.insert(4, "n_unique_variants_in_chunk", int(job.get("n_unique_variants", -1)))

    final_path = out_dir / "sed.blood.requested_pairs.final.tsv"
    final.to_csv(final_path, sep="\t", index=False)

    print(f"Wrote: {final_path} ({len(final):,} rows)")


if __name__ == "__main__":
    main()
