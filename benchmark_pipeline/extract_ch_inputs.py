#!/usr/bin/env python3
"""Extract CH benchmark inputs from the CH eQTL and fine-mapping release (eqtl_four_population_20260922).

Builds the CH nominal-eQTL sample, the CH fine-mapping benchmark (PIP >= 0.5 positives plus sampled
low-PIP and intermediate-PIP comparison variants) and the CH pairs of the 500-gene model-first panel.
The fine-mapping profile is the regular-eQTL profile named by the FINEMAPPING_PROFILE environment
variable; other profiles in the release are only summarized and are never pooled with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


P_THRESHOLD = 2.02e-5
Q_THRESHOLD = 0.05
NOMINAL_PER_TSS_BIN = 10_000
LOW_RATIO = 100
INTERMEDIATE_RATIO = 25
MIN_LOW_PER_STRATUM = 200
MIN_INTERMEDIATE_PER_STRATUM = 100
SEED = 20260524
TSS_BINS = ("0-3kb", "3-12kb", "12-35kb", ">35kb")
MAF_BINS = ("lt0.05", "0.05-0.2", "ge0.2")
FINEMAPPING_PROFILE = os.environ["FINEMAPPING_PROFILE"]
VARIANT_RE = re.compile(r"^chr([^:]+):(\d+)_([^_]+)_([^_]+)$")


def normalize_chromosome(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if text.startswith("chr") else f"chr{text}"


def tss_bin(distance: pd.Series) -> pd.Series:
    values = pd.to_numeric(distance, errors="coerce").abs()
    result = pd.Series("unknown", index=values.index, dtype="object")
    result.loc[values <= 3_000] = "0-3kb"
    result.loc[(values > 3_000) & (values <= 12_000)] = "3-12kb"
    result.loc[(values > 12_000) & (values <= 35_000)] = "12-35kb"
    result.loc[values > 35_000] = ">35kb"
    return result


def maf_bin(maf: pd.Series) -> pd.Series:
    values = pd.to_numeric(maf, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    result.loc[(values >= 0) & (values < 0.05)] = "lt0.05"
    result.loc[(values >= 0.05) & (values < 0.2)] = "0.05-0.2"
    result.loc[(values >= 0.2) & (values <= 0.5)] = "ge0.2"
    return result


def stable_u64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def add_pair_keys(frame: pd.DataFrame) -> pd.DataFrame:
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


def load_gene_metadata(gene_locations: Path, gene_annotation: Path) -> pd.DataFrame:
    locations = pd.read_csv(gene_locations, sep="\t", dtype={"ensgid": str})
    locations = locations.rename(
        columns={"ensgid": "gene_id", "TSS": "tss", "symbol": "location_symbol"}
    )
    locations["chromosome"] = locations["chromosome_name"].map(normalize_chromosome)
    annotations = pd.read_csv(gene_annotation, sep="\t", dtype={"gene_id": str})
    annotations = annotations.rename(columns={"strand": "gene_strand"})
    keep = ["gene_id", "gene_name", "gene_type", "gene_strand"]
    merged = locations[["gene_id", "chromosome", "tss", "location_symbol"]].merge(
        annotations[keep], on="gene_id", how="left"
    )
    if merged["gene_id"].duplicated().any():
        raise RuntimeError("Gene metadata contains duplicate gene IDs")
    return merged


def normalize_association_chunk(chunk: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    out = chunk.rename(
        columns={
            "chr": "chromosome",
            "pos": "position",
            "a2": "ref_allele",
            "a1": "alt_allele",
            "af": "alt_allele_freq",
            "se": "std_error",
            "n": "sample_size",
        }
    ).copy()
    out = add_pair_keys(out)
    out = out.merge(
        genes[["gene_id", "tss", "gene_name", "gene_type", "gene_strand"]],
        on="gene_id",
        how="left",
        validate="many_to_one",
    )
    out["distance_to_tss"] = (out["position"] - out["tss"]).abs()
    out["tss_distance_bin"] = tss_bin(out["distance_to_tss"])
    out["ancestry"] = "CH"
    out["population"] = "CH"
    out["maf"] = np.minimum(
        pd.to_numeric(out["alt_allele_freq"], errors="coerce"),
        1 - pd.to_numeric(out["alt_allele_freq"], errors="coerce"),
    )
    out["maf_bin"] = maf_bin(out["maf"])
    return out


def update_reservoir(
    current: pd.DataFrame | None,
    incoming: pd.DataFrame,
    n_keep: int,
    priority_col: str,
) -> pd.DataFrame:
    if incoming.empty:
        return current if current is not None else incoming
    combined = incoming if current is None or current.empty else pd.concat([current, incoming])
    if len(combined) <= n_keep:
        return combined
    return combined.nsmallest(n_keep, priority_col, keep="first")


def extract_association_inputs(
    release_root: Path,
    output_dir: Path,
    genes: pd.DataFrame,
    general_gene_ids: set[str],
    chunksize: int,
) -> dict[str, object]:
    export_root = release_root / "database_proposal/export_full/CH"
    nominal_reservoirs: dict[str, pd.DataFrame] = {}
    significant_rows = 0
    significant_genes: set[str] = set()
    tested_rows = 0
    model_first_rows = 0
    model_first_genes: set[str] = set()
    chromosome_summaries = []
    model_first_dir = output_dir / "model_first_fixed_panel_by_chromosome"
    model_first_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "chr", "pos", "a1", "a2", "gene_id", "var_id", "af", "ma_samples",
        "ma_count", "pvalue", "beta", "se", "n", "qvalue",
    ]
    for chromosome in range(1, 23):
        path = export_root / f"chr{chromosome}/magenta_ch_no_int_ad_mac10_ad_con_only.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        chromosome_tested = 0
        chromosome_significant = 0
        chromosome_model_first = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            chunk = normalize_association_chunk(chunk, genes)
            chromosome_tested += len(chunk)
            tested_rows += len(chunk)

            panel_pairs = chunk[
                chunk["gene_id"].isin(general_gene_ids)
                & chunk["distance_to_tss"].le(100_000)
            ].copy()
            if not panel_pairs.empty:
                chromosome_model_first.append(panel_pairs)
                model_first_rows += len(panel_pairs)
                model_first_genes.update(panel_pairs["gene_id"].astype(str))

            significant = chunk[
                pd.to_numeric(chunk["pvalue"], errors="coerce").lt(P_THRESHOLD)
                & pd.to_numeric(chunk["qvalue"], errors="coerce").lt(Q_THRESHOLD)
                & chunk["tss_distance_bin"].isin(TSS_BINS)
            ].copy()
            if significant.empty:
                continue
            chromosome_significant += len(significant)
            significant_rows += len(significant)
            significant_genes.update(significant["gene_id"].astype(str))
            significant["sampling_priority"] = significant["pair_key"].map(
                lambda value: stable_u64(f"nominal|{SEED}|{value}")
            )
            for label, part in significant.groupby("tss_distance_bin", sort=False):
                nominal_reservoirs[label] = update_reservoir(
                    nominal_reservoirs.get(label),
                    part,
                    NOMINAL_PER_TSS_BIN,
                    "sampling_priority",
                )

        if chromosome_model_first:
            panel_frame = pd.concat(chromosome_model_first, ignore_index=True)
            panel_frame = panel_frame.drop_duplicates("pair_key", keep="first")
            panel_frame.to_csv(
                model_first_dir / f"CH.chr{chromosome}.pairs.tsv.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )
        chromosome_summaries.append(
            {
                "chromosome": chromosome,
                "tested_rows": chromosome_tested,
                "significant_rows": chromosome_significant,
                "model_first_panel_rows": sum(len(x) for x in chromosome_model_first),
            }
        )

    nominal = pd.concat(
        [nominal_reservoirs[label] for label in TSS_BINS if label in nominal_reservoirs],
        ignore_index=True,
    )
    nominal = nominal.drop(columns=["sampling_priority"]).sort_values(
        ["tss_distance_bin", "chromosome", "position", "ref_allele", "alt_allele", "gene_id"],
        kind="mergesort",
    )
    nominal.to_csv(
        output_dir / "ch_nominal_sample.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(chromosome_summaries).to_csv(
        output_dir / "ch_association_chromosome_summary.tsv", sep="\t", index=False
    )
    return {
        "tested_rows": tested_rows,
        "significant_rows": significant_rows,
        "significant_genes": len(significant_genes),
        "nominal_sample_rows": len(nominal),
        "nominal_sample_by_tss_bin": nominal["tss_distance_bin"].value_counts().to_dict(),
        "model_first_panel_rows": model_first_rows,
        "model_first_panel_genes": len(model_first_genes),
    }


def load_score_presence(analysis_pairs: Path, combined_score_tables: list[Path]) -> dict[str, int]:
    presence: dict[str, int] = {}
    if analysis_pairs.is_file():
        usecols = ["pair_key", "borzoi_score", "alphagenome_score"]
        for chunk in pd.read_csv(analysis_pairs, sep="\t", usecols=usecols, chunksize=250_000):
            for row in chunk.itertuples(index=False):
                flags = 0
                if pd.notna(row.borzoi_score):
                    flags |= 1
                if pd.notna(row.alphagenome_score):
                    flags |= 2
                if flags:
                    presence[row.pair_key] = presence.get(row.pair_key, 0) | flags

    for path in combined_score_tables:
        if not path.is_file():
            continue
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        chromosome = frame["chromosome"] if "chromosome" in frame else frame["chr"]
        position = frame["position"] if "position" in frame else frame["pos"]
        ref = frame["ref_allele"] if "ref_allele" in frame else frame["ref"]
        alt = frame["alt_allele"] if "alt_allele" in frame else frame["alt"]
        keys = (
            chromosome.map(normalize_chromosome).astype(str)
            + ":"
            + pd.to_numeric(position, errors="coerce").astype("Int64").astype(str)
            + ":"
            + ref.astype(str)
            + ":"
            + alt.astype(str)
            + ":"
            + frame["gene_id"].astype(str)
        )
        for tool, key in zip(frame["tool"].astype(str), keys):
            flag = 1 if tool == "borzoi" else 2 if tool == "alphagenome" else 0
            if flag:
                presence[key] = presence.get(key, 0) | flag
    return presence


def parse_variant_id(value: str) -> tuple[str, int, str, str]:
    match = VARIANT_RE.match(str(value))
    if not match:
        raise ValueError(f"Unrecognized variant ID: {value}")
    chrom, position, ref, alt = match.groups()
    return f"chr{chrom}", int(position), ref, alt


def normalize_pip_frame(frame: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(
        columns={
            "chrom": "chromosome",
            "pos": "position",
            "a2": "ref_allele",
            "a1": "alt_allele",
            "af": "alt_allele_freq",
            "se": "std_error",
            "n": "sample_size",
            "qvalue": "qvalue",
        }
    ).copy()
    out = add_pair_keys(out)
    out = out.merge(
        genes[["gene_id", "tss", "gene_name", "gene_type", "gene_strand"]],
        on="gene_id",
        how="left",
        validate="many_to_one",
    )
    out["distance_to_tss"] = (out["position"] - out["tss"]).abs()
    out["tss_distance_bin"] = tss_bin(out["distance_to_tss"])
    out["maf"] = np.minimum(
        pd.to_numeric(out["alt_allele_freq"], errors="coerce"),
        1 - pd.to_numeric(out["alt_allele_freq"], errors="coerce"),
    )
    out["maf_bin"] = maf_bin(out["maf"])
    out["ancestry"] = "CH"
    out["population"] = "CH"
    out["profile"] = FINEMAPPING_PROFILE
    out["mode"] = "regular"
    return out


def trim_reused(frame: pd.DataFrame, target: int, pip_class: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    ascending_pip = pip_class == "low"
    return frame.sort_values(
        ["reuse_tool_count", "CS", "pip", "pair_key"],
        ascending=[False, True, ascending_pip, True],
        kind="mergesort",
    ).drop_duplicates("pair_key").head(target)


def extract_finemapping_inputs(
    release_root: Path,
    output_dir: Path,
    genes: pd.DataFrame,
    score_presence: dict[str, int],
) -> dict[str, object]:
    summary_dir = release_root / "finemapping/summary/CH"
    members = pd.read_csv(summary_dir / "credible_set_members.tsv.gz", sep="\t")
    loci = pd.read_csv(summary_dir / "locus_summary.tsv", sep="\t", low_memory=False)

    profile_summary = (
        members.groupby(["mode", "profile"], observed=True)
        .agg(rows=("variant_id", "size"), genes=("gene_id", "nunique"), max_pip=("pip", "max"))
        .reset_index()
    )
    profile_summary.to_csv(output_dir / "ch_finemapping_profile_summary.tsv", sep="\t", index=False)

    positive_keys = members[
        members["mode"].eq("regular")
        & members["profile"].eq(FINEMAPPING_PROFILE)
        & pd.to_numeric(members["pip"], errors="coerce").ge(0.5)
    ][["gene_id", "variant_id", "pip", "cs", "converged"]].copy()
    positive_keys = positive_keys.sort_values("pip", ascending=False).drop_duplicates(
        ["gene_id", "variant_id"], keep="first"
    )

    regular_loci = loci[loci["mode"].eq("regular")].copy()
    path_col = f"{FINEMAPPING_PROFILE}_output"
    regular_loci = regular_loci[
        regular_loci[path_col].notna()
        & regular_loci[f"{FINEMAPPING_PROFILE}_fit_available"].astype(str).str.lower().eq("true")
    ].drop_duplicates("gene_id", keep="last")
    gene_to_pip = {
        row.gene_id: Path(str(getattr(row, path_col))) / "pip.tsv.gz"
        for row in regular_loci.itertuples(index=False)
    }

    positive_parts = []
    missing_positive_genes = []
    for gene_id, wanted in positive_keys.groupby("gene_id", sort=True):
        path = gene_to_pip.get(gene_id)
        if path is None or not path.is_file():
            missing_positive_genes.append(gene_id)
            continue
        frame = pd.read_csv(path, sep="\t")
        frame = frame[frame["variant_id"].astype(str).isin(set(wanted["variant_id"].astype(str)))]
        if frame.empty:
            continue
        frame = normalize_pip_frame(frame, genes)
        frame = frame.merge(
            wanted[["variant_id", "cs"]].rename(columns={"cs": "cs_label"}),
            on="variant_id",
            how="left",
        )
        frame["selection_label"] = "positive_pip_ge_0.5"
        frame["pip_class"] = "positive"
        positive_parts.append(frame)

    if missing_positive_genes:
        raise RuntimeError(f"Missing positive PIP files for {len(missing_positive_genes)} genes")
    positives = pd.concat(positive_parts, ignore_index=True).drop_duplicates("pair_key", keep="first")
    positives = positives[
        positives["tss_distance_bin"].isin(TSS_BINS) & positives["maf_bin"].isin(MAF_BINS)
    ].copy()
    positives["reuse_flags"] = positives["pair_key"].map(score_presence).fillna(0).astype(int)
    positives["reuse_tool_count"] = positives["reuse_flags"].map(
        lambda value: int(bool(value & 1)) + int(bool(value & 2))
    )

    positive_counts = positives.groupby(
        ["tss_distance_bin", "maf_bin"], observed=True
    ).size().to_dict()
    targets: dict[tuple[str, str, str], int] = {}
    for (distance_label, maf_label), count in positive_counts.items():
        targets[("low", distance_label, maf_label)] = max(count * LOW_RATIO, MIN_LOW_PER_STRATUM)
        targets[("intermediate", distance_label, maf_label)] = max(
            count * INTERMEDIATE_RATIO, MIN_INTERMEDIATE_PER_STRATUM
        )

    reused: dict[tuple[str, str, str], pd.DataFrame] = {}
    unscored: dict[tuple[str, str, str], pd.DataFrame] = {}
    scanned_rows = 0
    scanned_genes = 0
    available_by_stratum = defaultdict(int)
    for gene_id, path in sorted(gene_to_pip.items()):
        if not path.is_file():
            continue
        frame = pd.read_csv(path, sep="\t")
        if frame.empty:
            continue
        scanned_genes += 1
        scanned_rows += len(frame)
        frame = normalize_pip_frame(frame, genes)
        frame["pip"] = pd.to_numeric(frame["pip"], errors="coerce")
        frame["CS"] = pd.to_numeric(frame.get("CS", 0), errors="coerce").fillna(0).astype(int)
        frame["pip_class"] = "drop"
        frame.loc[frame["pip"].lt(0.01), "pip_class"] = "low"
        frame.loc[frame["pip"].ge(0.01) & frame["pip"].lt(0.5), "pip_class"] = "intermediate"
        frame = frame[
            frame["pip_class"].isin(["low", "intermediate"])
            & frame["tss_distance_bin"].isin(TSS_BINS)
            & frame["maf_bin"].isin(MAF_BINS)
        ].copy()
        if frame.empty:
            continue
        frame["reuse_flags"] = frame["pair_key"].map(score_presence).fillna(0).astype(int)
        frame["reuse_tool_count"] = frame["reuse_flags"].map(
            lambda value: int(bool(value & 1)) + int(bool(value & 2))
        )
        frame["sampling_priority"] = [
            -math.log(max((stable_u64(f"background|{SEED}|{cls}|{key}") + 1) / (2**64 + 1), 1e-300))
            / (2.0 if int(cs) == 0 else 1.0)
            for cls, key, cs in zip(frame["pip_class"], frame["pair_key"], frame["CS"])
        ]
        for (cls, distance_label, maf_label), part in frame.groupby(
            ["pip_class", "tss_distance_bin", "maf_bin"], observed=True, sort=False
        ):
            key = (cls, distance_label, maf_label)
            if key not in targets:
                continue
            available_by_stratum[key] += len(part)
            target = targets[key]
            reuse_part = part[part["reuse_tool_count"].gt(0)]
            unscored_part = part[part["reuse_tool_count"].eq(0)]
            if not reuse_part.empty:
                combined = reuse_part if key not in reused else pd.concat([reused[key], reuse_part])
                reused[key] = trim_reused(combined, target, cls)
            if not unscored_part.empty:
                unscored[key] = update_reservoir(
                    unscored.get(key), unscored_part, target, "sampling_priority"
                )

    background_parts = []
    selection_rows = []
    for key, target in sorted(targets.items()):
        cls, distance_label, maf_label = key
        reuse_part = trim_reused(reused.get(key, pd.DataFrame()), target, cls)
        remaining = max(0, target - len(reuse_part))
        new_part = unscored.get(key, pd.DataFrame())
        if remaining and not new_part.empty:
            new_part = new_part.nsmallest(remaining, "sampling_priority", keep="first")
        else:
            new_part = new_part.iloc[0:0] if not new_part.empty else new_part
        selected = pd.concat([reuse_part, new_part], ignore_index=True)
        selected = selected.drop_duplicates("pair_key", keep="first").head(target)
        selected["selection_label"] = (
            "sampled_negative_pip_lt_0.01"
            if cls == "low"
            else "sampled_intermediate_pip_0.01_to_0.5"
        )
        background_parts.append(selected)
        selection_rows.append(
            {
                "pip_class": cls,
                "tss_distance_bin": distance_label,
                "maf_bin": maf_label,
                "positive_count": positive_counts.get((distance_label, maf_label), 0),
                "target": target,
                "available": available_by_stratum.get(key, 0),
                "selected": len(selected),
                "reused_selected": int(selected["reuse_tool_count"].gt(0).sum()),
            }
        )

    selected = pd.concat([positives, *background_parts], ignore_index=True, sort=False)
    selected = selected.drop(columns=["sampling_priority"], errors="ignore")
    selected = selected.drop_duplicates("pair_key", keep="first").sort_values(
        ["selection_label", "tss_distance_bin", "maf_bin", "chromosome", "position", "gene_id"],
        kind="mergesort",
    )
    selected.to_csv(
        output_dir / "ch_finemap_selected_pairs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    positives.to_csv(
        output_dir / "ch_finemap_positives.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "ch_finemap_background_selection.tsv", sep="\t", index=False
    )
    selected.groupby("selection_label", observed=True).size().reset_index(name="n").to_csv(
        output_dir / "ch_finemap_selected_summary.tsv", sep="\t", index=False
    )
    return {
        "credible_set_profile_rows": len(profile_summary),
        "positive_pairs_pip_ge_0_5": len(positives),
        "positive_pairs_pip_ge_0_9": int(positives["pip"].ge(0.9).sum()),
        "positive_genes": int(positives["gene_id"].nunique()),
        "all_variant_rows_scanned": scanned_rows,
        "all_variant_genes_scanned": scanned_genes,
        "selected_pairs": len(selected),
        "selected_pairs_with_both_scores_reusable": int(selected["reuse_flags"].eq(3).sum()),
    }


def audit_fixed_panel_overlap(
    output_dir: Path, analysis_pairs: Path, score_presence: dict[str, int]
) -> dict[str, object]:
    release_parts = []
    for path in sorted((output_dir / "model_first_fixed_panel_by_chromosome").glob("*.tsv.gz")):
        release_parts.append(pd.read_csv(path, sep="\t", usecols=["pair_key", "gene_id"]))
    release = pd.concat(release_parts, ignore_index=True).drop_duplicates("pair_key")
    locked = pd.read_csv(
        analysis_pairs,
        sep="\t",
        usecols=["pair_key", "gene_id", "general", "CH_present", "abs_distance_to_tss"],
    )
    locked = locked[
        locked["general"].eq(1)
        & locked["CH_present"].eq(1)
        & pd.to_numeric(locked["abs_distance_to_tss"], errors="coerce").le(100_000)
    ].drop_duplicates("pair_key")
    release_keys = set(release["pair_key"])
    locked_keys = set(locked["pair_key"])
    release["reuse_flags"] = release["pair_key"].map(score_presence).fillna(0).astype(int)
    summary = {
        "release_panel_pairs": len(release_keys),
        "release_panel_genes": int(release["gene_id"].nunique()),
        "locked_ch_native_pairs": len(locked_keys),
        "locked_ch_native_genes": int(locked["gene_id"].nunique()),
        "pair_overlap": len(release_keys & locked_keys),
        "locked_only_pairs": len(locked_keys - release_keys),
        "release_only_pairs": len(release_keys - locked_keys),
        "release_pairs_with_both_scores_reusable": int(release["reuse_flags"].eq(3).sum()),
        "release_pairs_missing_at_least_one_score": int(release["reuse_flags"].ne(3).sum()),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "model_first_fixed_panel_overlap_audit.tsv", sep="\t", index=False
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gene-locations", type=Path, required=True)
    parser.add_argument("--gene-annotation", type=Path, required=True)
    parser.add_argument("--general-genes", type=Path, required=True)
    parser.add_argument("--analysis-pairs", type=Path, required=True)
    parser.add_argument("--combined-score-table", type=Path, action="append", default=[])
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--skip-association", action="store_true")
    parser.add_argument("--skip-finemapping", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    genes = load_gene_metadata(args.gene_locations, args.gene_annotation)
    general = pd.read_csv(args.general_genes, sep="\t", usecols=["gene_id"])
    general_gene_ids = set(general["gene_id"].astype(str))
    if len(general_gene_ids) != 500:
        raise RuntimeError(f"Expected 500 locked general genes, found {len(general_gene_ids)}")

    summary: dict[str, object] = {
        "release_root": str(args.release_root),
        "fine_mapping_primary_profile": FINEMAPPING_PROFILE,
        "other_fine_mapping_profiles": "summarized_separately_not_pooled",
        "interaction_eqtl_in_scope": False,
        "sample_size": 209,
        "seed": SEED,
        "software": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    if not args.skip_association:
        summary["association"] = extract_association_inputs(
            args.release_root,
            args.output_dir,
            genes,
            general_gene_ids,
            args.chunksize,
        )

    score_presence = load_score_presence(args.analysis_pairs, args.combined_score_table)
    summary["score_presence_keys"] = len(score_presence)
    if not args.skip_finemapping:
        summary["fine_mapping"] = extract_finemapping_inputs(
            args.release_root, args.output_dir, genes, score_presence
        )
    if (args.output_dir / "model_first_fixed_panel_by_chromosome").is_dir():
        summary["model_first_fixed_panel"] = audit_fixed_panel_overlap(
            args.output_dir, args.analysis_pairs, score_presence
        )

    temporary = args.output_dir / f"extraction_summary.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_dir / "extraction_summary.json")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
