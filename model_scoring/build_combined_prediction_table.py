#!/usr/bin/env python3
"""
Build the combined prediction table for the fine-mapped eQTL AUROC benchmark.

Combines, in priority order:
1. prior scored predictions supplied by --prior_combined
2. current scored predictions supplied by --current_combined
3. newly generated chunk-level scores in the selected results directory

Only selected pairs from <results_dir>/selected_pairs.tsv.gz are kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = ["borzoi", "alphagenome"]
ANCESTRIES = ["AA", "CH", "NHW"]
def _safe_slug(s: str) -> str:
    return str(s).strip().replace("/", "_").replace(" ", "_").replace(">", "gt").replace("<", "lt")


def _normalize_chr(chromosome: str) -> str:
    chrom = str(chromosome)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def key_cols() -> list[str]:
    return ["chromosome_key", "position_key", "ref_key", "alt_key", "gene_id", "ancestry"]


def coalesce_column(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in df.columns and fallback in df.columns:
        return df[preferred].where(df[preferred].notna(), df[fallback])
    if preferred in df.columns:
        return df[preferred]
    return df[fallback]


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["chromosome_key"] = coalesce_column(out, "chromosome", "chr")
    out["position_key"] = coalesce_column(out, "position", "pos")
    out["ref_key"] = coalesce_column(out, "ref_allele", "ref")
    out["alt_key"] = coalesce_column(out, "alt_allele", "alt")
    out["chromosome_key"] = pd.Series(out["chromosome_key"]).map(_normalize_chr)
    out["position_key"] = pd.to_numeric(out["position_key"], errors="coerce")
    return out


def load_selected(results_dir: Path) -> pd.DataFrame:
    selected = pd.read_csv(results_dir / "selected_pairs.tsv.gz", sep="\t", low_memory=False)
    selected = selected.rename(
        columns={
            "chromosome": "chromosome_key",
            "position": "position_key",
            "ref_allele": "ref_key",
            "alt_allele": "alt_key",
        }
    )
    selected["chromosome_key"] = selected["chromosome_key"].map(_normalize_chr)
    selected["position_key"] = pd.to_numeric(selected["position_key"], errors="coerce")
    return selected[key_cols()].drop_duplicates()


def subset_existing(path: Path, selected_keys: pd.DataFrame, source_label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] missing existing combined table: {path}")
        return None
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["tool"].isin(TOOLS) & df["ancestry"].isin(ANCESTRIES)].copy()
    keyed = add_keys(df)
    subset = keyed.merge(selected_keys, on=key_cols(), how="inner")
    subset["redo_score_source"] = source_label
    print(f"[reuse] {source_label}: {len(subset):,}")
    return subset


def collect_new(results_dir: Path) -> list[pd.DataFrame]:
    dfs = []
    for tool in TOOLS:
        for ancestry in ANCESTRIES:
            root = results_dir / tool / _safe_slug(ancestry)
            if not root.exists():
                continue
            pattern = (
                "chunk_*/sed.blood.requested_pairs.final.tsv"
                if tool == "borzoi"
                else "chunk_*/tidy.whole_blood.requested_pairs.final.tsv"
            )
            for path in sorted(root.glob(pattern)):
                try:
                    df = pd.read_csv(path, sep="\t", low_memory=False)
                except Exception as exc:
                    print(f"[warn] skipping unreadable {path}: {exc}")
                    continue
                if len(df):
                    dfs.append(df)
    return dfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results_name", default="fine_mapped_eqtl_auroc_results")
    parser.add_argument("--prior_combined", type=Path, default=None)
    parser.add_argument("--current_combined", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.base_dir.resolve() / args.results_name
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = load_selected(results_dir)

    pieces = []
    reusable_sources = []
    if args.prior_combined:
        reusable_sources.append((args.prior_combined, "reused_prior_prediction"))
    if args.current_combined:
        reusable_sources.append((args.current_combined, "reused_current_prediction"))

    for path, label in reusable_sources:
        subset = subset_existing(path, selected_keys, label)
        if subset is not None and len(subset):
            pieces.append(subset)

    new_dfs = collect_new(results_dir)
    if new_dfs:
        new = pd.concat(new_dfs, ignore_index=True)
        new["redo_score_source"] = "new_auroc_expanded_prediction"
        pieces.append(new)
        print(f"[new] expanded predictions collected: {len(new):,}")
    else:
        print("[new] no expanded chunk predictions found yet")

    if not pieces:
        raise RuntimeError("No prediction rows available to combine")

    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined = add_keys(combined)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["tool"] + key_cols(), keep="first")
    print(f"[dedupe] {before:,} -> {len(combined):,} tool/pair rows")

    out_path = analysis_dir / "susie_s2f_combined.tsv.gz"
    combined.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
