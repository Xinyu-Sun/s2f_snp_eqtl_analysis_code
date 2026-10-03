#!/usr/bin/env python3
"""Build one deduplicated scoring plan for the CH benchmark and model-first pairs.

Pairs with an existing Borzoi or AlphaGenome score for the exact chromosome-position-REF-ALT-gene key
reuse that score; all other pairs are written to scoring chunks for model_scoring/run_*_chunk.py."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = ("borzoi", "alphagenome")
SEED = 20260922
PRIMARY_FINEMAPPING_MODE = "regular"
PRIMARY_FINEMAPPING_PROFILE = os.environ["FINEMAPPING_PROFILE"]


def normalize_chromosome(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if text.startswith("chr") else f"chr{text}"


def add_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["chromosome"] = out["chromosome"].map(normalize_chromosome)
    out["position"] = pd.to_numeric(out["position"], errors="raise").astype("int64")
    out["pair_key"] = (
        out["chromosome"].astype(str)
        + ":"
        + out["position"].astype(str)
        + ":"
        + out["ref_allele"].astype(str)
        + ":"
        + out["alt_allele"].astype(str)
        + ":"
        + out["gene_id"].astype(str)
    )
    return out


def select_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "pair_key", "chromosome", "position", "ref_allele", "alt_allele", "gene_id",
        "ancestry", "population", "tss", "distance_to_tss", "tss_distance_bin",
        "alt_allele_freq", "maf", "maf_bin", "pvalue", "beta", "std_error", "qvalue",
        "pip", "CS", "cs_label", "profile", "mode", "gene_name", "gene_type",
        "gene_strand", "sample_size", "analysis_role", "selection_label",
    ]
    return frame[[column for column in preferred if column in frame.columns]].copy()


def load_selected_pairs(extraction_dir: Path) -> pd.DataFrame:
    pieces = []

    nominal = pd.read_csv(extraction_dir / "ch_nominal_sample.tsv.gz", sep="\t")
    nominal["analysis_role"] = "nominal_benchmark"
    pieces.append(select_pair_columns(add_pair_key(nominal)))

    finemap = pd.read_csv(
        extraction_dir / "ch_finemap_selected_pairs.tsv.gz", sep="\t", low_memory=False
    )
    required = {"mode", "profile"}
    missing = required.difference(finemap.columns)
    if missing:
        raise RuntimeError(
            "CH fine-mapping input is missing required provenance columns: "
            + ", ".join(sorted(missing))
        )
    wrong_profile = ~(
        finemap["mode"].eq(PRIMARY_FINEMAPPING_MODE)
        & finemap["profile"].eq(PRIMARY_FINEMAPPING_PROFILE)
    )
    if wrong_profile.any():
        observed = (
            finemap.loc[wrong_profile, ["mode", "profile"]]
            .value_counts(dropna=False)
            .to_dict()
        )
        raise RuntimeError(
            "CH fine-mapping input must contain only the "
            f"{PRIMARY_FINEMAPPING_MODE}/{PRIMARY_FINEMAPPING_PROFILE} profile; "
            f"observed disallowed rows: {observed}"
        )
    finemap["analysis_role"] = "finemapping_benchmark"
    pieces.append(select_pair_columns(add_pair_key(finemap)))

    panel_parts = []
    for path in sorted((extraction_dir / "model_first_fixed_panel_by_chromosome").glob("*.tsv.gz")):
        panel_parts.append(pd.read_csv(path, sep="\t", low_memory=False))
    if panel_parts:
        panel = pd.concat(panel_parts, ignore_index=True)
        panel["analysis_role"] = "model_first_fixed_panel"
        pieces.append(select_pair_columns(add_pair_key(panel)))

    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined["ancestry"] = "CH"
    combined["population"] = "CH"
    role_map = (
        combined.groupby("pair_key", sort=False)["analysis_role"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .to_dict()
    )
    priority = {
        "finemapping_benchmark": 0,
        "nominal_benchmark": 1,
        "model_first_fixed_panel": 2,
    }
    combined["role_priority"] = combined["analysis_role"].map(priority).fillna(9)
    combined = combined.sort_values(["role_priority", "pair_key"], kind="mergesort")
    combined = combined.drop_duplicates("pair_key", keep="first")
    combined["analysis_role"] = combined["pair_key"].map(role_map)
    combined = combined.drop(columns=["role_priority"])
    return combined.sort_values(
        ["chromosome", "position", "ref_allele", "alt_allele", "gene_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    chromosome = frame["chromosome"] if "chromosome" in frame else frame["chr"]
    position = frame["position"] if "position" in frame else frame["pos"]
    ref = frame["ref_allele"] if "ref_allele" in frame else frame["ref"]
    alt = frame["alt_allele"] if "alt_allele" in frame else frame["alt"]
    keys = pd.DataFrame(
        {
            "chromosome": chromosome.map(normalize_chromosome),
            "position": pd.to_numeric(position, errors="coerce"),
            "ref_allele": ref.astype(str),
            "alt_allele": alt.astype(str),
            "gene_id": frame["gene_id"].astype(str),
        },
        index=frame.index,
    ).dropna(subset=["position"])
    keys["position"] = keys["position"].astype("int64")
    return add_pair_key(keys)


def tool_rows_from_analysis_pairs(path: Path) -> pd.DataFrame:
    parts = []
    usecols = ["pair_key", "borzoi_score", "alphagenome_score"]
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=250_000):
        for tool, column in [("borzoi", "borzoi_score"), ("alphagenome", "alphagenome_score")]:
            part = chunk[["pair_key", column]].rename(columns={column: "score"})
            part = part[np.isfinite(pd.to_numeric(part["score"], errors="coerce"))].copy()
            part["tool"] = tool
            part["score_source"] = "model_first_analysis_pairs"
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def tool_rows_from_combined(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    keys = normalized_keys(frame)
    frame = frame.loc[keys.index].copy()
    frame["pair_key"] = keys["pair_key"]
    score = pd.Series(np.nan, index=frame.index, dtype="float64")
    borzoi = frame["tool"].astype(str).eq("borzoi")
    alphagenome = frame["tool"].astype(str).eq("alphagenome")
    if "logMeanSED" in frame:
        score.loc[borzoi] = pd.to_numeric(frame.loc[borzoi, "logMeanSED"], errors="coerce")
    if "raw_score" in frame:
        score.loc[alphagenome] = pd.to_numeric(frame.loc[alphagenome, "raw_score"], errors="coerce")
    result = pd.DataFrame(
        {"pair_key": frame["pair_key"], "tool": frame["tool"].astype(str), "score": score}
    )
    result = result[result["tool"].isin(TOOLS) & np.isfinite(result["score"])].copy()
    result["score_source"] = label
    return result


def tool_rows_from_nominal_long(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    keys = normalized_keys(frame)
    frame = frame.loc[keys.index].copy()
    frame["pair_key"] = keys["pair_key"]
    result = frame[["pair_key", "tool", "prediction"]].rename(columns={"prediction": "score"})
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result = result[result["tool"].isin(TOOLS) & np.isfinite(result["score"])].copy()
    result["score_source"] = "nominal_10k_long"
    return result


def build_score_cache(
    selected: pd.DataFrame,
    analysis_pairs: Path,
    combined_tables: list[Path],
    nominal_long: Path,
    diagnostics_dir: Path,
) -> pd.DataFrame:
    selected_keys = set(selected["pair_key"])
    pieces = [tool_rows_from_analysis_pairs(analysis_pairs)]
    pieces.extend(
        tool_rows_from_combined(path, f"combined:{path.parent.parent.name}")
        for path in combined_tables
        if path.is_file()
    )
    if nominal_long.is_file():
        pieces.append(tool_rows_from_nominal_long(nominal_long))
    cache = pd.concat(pieces, ignore_index=True)
    cache = cache[cache["pair_key"].isin(selected_keys)].copy()
    source_priority = {
        "model_first_analysis_pairs": 0,
        "nominal_10k_long": 1,
    }
    cache["source_priority"] = cache["score_source"].map(source_priority).fillna(2)
    cache = cache.sort_values(["source_priority", "pair_key", "tool"], kind="mergesort")

    disagreement_rows = []
    for (pair_key, tool), group in cache.groupby(["pair_key", "tool"], sort=False):
        values = pd.to_numeric(group["score"], errors="coerce").dropna().to_numpy()
        spread = float(values.max() - values.min()) if len(values) else math.nan
        if len(values) > 1 and spread > 1e-5:
            for row in group.itertuples(index=False):
                disagreement_rows.append(
                    {
                        "pair_key": pair_key,
                        "tool": tool,
                        "n_sources": len(values),
                        "score_spread": spread,
                        "score_source": row.score_source,
                        "score": row.score,
                    }
                )
    if disagreement_rows:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        disagreement = pd.DataFrame(disagreement_rows)
        disagreement.to_csv(
            diagnostics_dir / "existing_score_disagreements.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
        summary = {
            "disagreeing_tool_pair_keys": int(
                disagreement[["pair_key", "tool"]].drop_duplicates().shape[0]
            ),
            "by_tool": disagreement[["pair_key", "tool"]]
            .drop_duplicates()["tool"]
            .value_counts()
            .to_dict(),
            "by_source": disagreement["score_source"].value_counts().to_dict(),
            "spread_quantiles": disagreement[["pair_key", "tool", "score_spread"]]
            .drop_duplicates()["score_spread"]
            .quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
            .rename_axis("quantile")
            .to_dict(),
            "resolution": "excluded_from_cache_and_rescored",
        }
        (diagnostics_dir / "existing_score_disagreements_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        conflict_keys = disagreement[["pair_key", "tool"]].drop_duplicates()
        conflict_keys["exclude_from_cache"] = True
        cache = cache.merge(conflict_keys, on=["pair_key", "tool"], how="left")
        cache = cache[cache["exclude_from_cache"].isna()].drop(columns=["exclude_from_cache"])
    return cache.drop_duplicates(["pair_key", "tool"], keep="first").drop(columns=["source_priority"])


def write_vcf(path: Path, frame: pd.DataFrame) -> int:
    variants = frame[
        ["chromosome", "position", "ref_allele", "alt_allele"]
    ].drop_duplicates().sort_values(
        ["chromosome", "position", "ref_allele", "alt_allele"], kind="mergesort"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        for row in variants.itertuples(index=False):
            variant_id = f"{row.chromosome}_{row.position}_{row.ref_allele}_{row.alt_allele}"
            handle.write(
                f"{row.chromosome}\t{row.position}\t{variant_id}\t{row.ref_allele}\t{row.alt_allele}\t.\t.\n"
            )
    return len(variants)


def write_jobs(missing: pd.DataFrame, output_dir: Path, pairs_per_chunk: int) -> pd.DataFrame:
    job_rows = []
    for tool in TOOLS:
        tool_pairs = missing[missing["tool"].eq(tool)].copy()
        tool_pairs = tool_pairs.sort_values(
            ["chromosome", "position", "ref_allele", "alt_allele", "gene_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        tool_pairs["chunk_index"] = tool_pairs.index // pairs_per_chunk
        for chunk_index, chunk in tool_pairs.groupby("chunk_index", sort=True):
            chunk = chunk.drop(columns=["tool", "chunk_index", "score", "score_source"], errors="ignore")
            chunk_dir = output_dir / "scoring_inputs" / tool / f"chunk_{int(chunk_index):04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            pairs_path = chunk_dir / "pairs.tsv"
            vcf_path = chunk_dir / "variants.vcf"
            chunk.to_csv(pairs_path, sep="\t", index=False)
            n_unique = write_vcf(vcf_path, chunk)
            job_rows.append(
                {
                    "tool": tool,
                    "ancestry": "CH",
                    "chunk_index": int(chunk_index),
                    "pairs_tsv": str(pairs_path.resolve()),
                    "variants_vcf": str(vcf_path.resolve()),
                    "n_pairs": len(chunk),
                    "n_unique_variants": n_unique,
                    "seed": SEED,
                }
            )
    jobs = pd.DataFrame(job_rows)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(manifest_dir / "master_scoring_jobs.tsv", sep="\t", index=False)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-pairs", type=Path, required=True)
    parser.add_argument("--combined-score-table", type=Path, action="append", default=[])
    parser.add_argument("--nominal-long", type=Path, required=True)
    parser.add_argument("--pairs-per-chunk", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected_pairs(args.extraction_dir)
    selected.to_csv(
        args.output_dir / "master_selected_pairs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    cache = build_score_cache(
        selected,
        args.analysis_pairs,
        args.combined_score_table,
        args.nominal_long,
        args.output_dir,
    )
    cache.to_csv(
        args.output_dir / "reused_model_scores.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    requested = pd.concat([selected.assign(tool=tool) for tool in TOOLS], ignore_index=True)
    requested = requested.merge(cache, on=["pair_key", "tool"], how="left")
    missing = requested[requested["score"].isna()].copy()
    missing.to_csv(
        args.output_dir / "missing_tool_pairs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    jobs = write_jobs(missing, args.output_dir, args.pairs_per_chunk)

    roles = selected.assign(analysis_role=selected["analysis_role"].str.split(",")).explode("analysis_role")
    summary = {
        "selected_unique_pairs": len(selected),
        "selected_by_role": roles["analysis_role"].value_counts().to_dict(),
        "requested_tool_pairs": len(requested),
        "reused_tool_pairs": int(requested["score"].notna().sum()),
        "missing_tool_pairs": len(missing),
        "missing_by_tool": missing["tool"].value_counts().to_dict(),
        "jobs_by_tool": jobs["tool"].value_counts().to_dict() if not jobs.empty else {},
        "pairs_per_chunk": args.pairs_per_chunk,
        "seed": SEED,
    }
    temporary = args.output_dir / f"master_scoring_summary.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_dir / "master_scoring_summary.json")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
