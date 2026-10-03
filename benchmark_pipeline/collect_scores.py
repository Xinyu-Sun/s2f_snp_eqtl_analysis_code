#!/usr/bin/env python3
"""Collect reused and newly generated S2F model scores for the scoring plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = ("borzoi", "alphagenome")


def normalize_chromosome(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if text.startswith("chr") else f"chr{text}"


def add_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    chromosome = out["chromosome"] if "chromosome" in out else out["chr"]
    position = out["position"] if "position" in out else out["pos"]
    ref = out["ref_allele"] if "ref_allele" in out else out["ref"]
    alt = out["alt_allele"] if "alt_allele" in out else out["alt"]
    out["chromosome_key"] = chromosome.map(normalize_chromosome)
    out["position_key"] = pd.to_numeric(position, errors="coerce").astype("Int64")
    out["ref_key"] = ref.astype(str)
    out["alt_key"] = alt.astype(str)
    out["pair_key"] = (
        out["chromosome_key"].astype(str)
        + ":"
        + out["position_key"].astype(str)
        + ":"
        + out["ref_key"]
        + ":"
        + out["alt_key"]
        + ":"
        + out["gene_id"].astype(str)
    )
    return out


def collect_new_scores(new_root: Path) -> pd.DataFrame:
    pieces = []
    for path in sorted((new_root / "borzoi/CH").glob("chunk_*/sed.blood.requested_pairs.final.tsv")):
        frame = add_key(pd.read_csv(path, sep="\t"))
        frame["score"] = pd.to_numeric(frame["logMeanSED"], errors="coerce")
        frame["tool"] = "borzoi"
        frame["score_source"] = f"new:{path.parent.name}"
        pieces.append(frame[["pair_key", "tool", "score", "score_source"]])
    for path in sorted(
        (new_root / "alphagenome/CH").glob("chunk_*/tidy.whole_blood.requested_pairs.final.tsv")
    ):
        frame = add_key(pd.read_csv(path, sep="\t"))
        frame["score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
        frame["tool"] = "alphagenome"
        frame["score_source"] = f"new:{path.parent.name}"
        pieces.append(frame[["pair_key", "tool", "score", "score_source"]])
    if not pieces:
        return pd.DataFrame(columns=["pair_key", "tool", "score", "score_source"])
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    selected = pd.read_csv(args.scoring_dir / "master_selected_pairs.tsv.gz", sep="\t", low_memory=False)
    reused = pd.read_csv(args.scoring_dir / "reused_model_scores.tsv.gz", sep="\t")
    new = collect_new_scores(args.scoring_dir / "new_scores")
    scores = pd.concat([reused, new], ignore_index=True)
    scores = scores[np.isfinite(pd.to_numeric(scores["score"], errors="coerce"))].copy()
    scores["source_priority"] = scores["score_source"].astype(str).str.startswith("new:").astype(int)
    scores = scores.sort_values(["source_priority", "pair_key", "tool"], kind="mergesort")

    duplicate_spread = (
        scores.groupby(["pair_key", "tool"], observed=True)["score"]
        .agg(lambda values: float(pd.to_numeric(values).max() - pd.to_numeric(values).min()))
    )
    conflicts = duplicate_spread[duplicate_spread.gt(1e-5)]
    if len(conflicts):
        raise RuntimeError(f"Conflicting scores for {len(conflicts)} exact tool/pair keys")
    scores = scores.drop_duplicates(["pair_key", "tool"], keep="first").drop(
        columns=["source_priority"]
    )

    requested = pd.MultiIndex.from_product(
        [selected["pair_key"].astype(str), TOOLS], names=["pair_key", "tool"]
    ).to_frame(index=False)
    audit = requested.merge(scores, on=["pair_key", "tool"], how="left")
    missing = audit[audit["score"].isna()].copy()
    missing.to_csv(args.scoring_dir / "missing_scores_after_collection.tsv", sep="\t", index=False)
    if len(missing) and not args.allow_incomplete:
        raise RuntimeError(f"Missing {len(missing)} requested tool/pair scores after collection")

    scores.to_csv(
        args.scoring_dir / "master_model_scores.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    wide = scores.pivot(index="pair_key", columns="tool", values="score").reset_index()
    wide = wide.rename(columns={"borzoi": "borzoi_score", "alphagenome": "alphagenome_score"})
    combined = selected.merge(wide, on="pair_key", how="left", validate="one_to_one")
    combined.to_csv(
        args.scoring_dir / "master_scored_pairs_wide.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    summary = {
        "selected_pairs": len(selected),
        "requested_tool_pairs": len(requested),
        "reused_tool_pairs": len(reused.drop_duplicates(["pair_key", "tool"])),
        "new_tool_pairs": len(new.drop_duplicates(["pair_key", "tool"])),
        "available_tool_pairs": len(audit) - len(missing),
        "missing_tool_pairs": len(missing),
        "complete_pairs_both_models": int(
            combined[["borzoi_score", "alphagenome_score"]].notna().all(axis=1).sum()
        ),
    }
    temporary = args.scoring_dir / f"score_collection_summary.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.scoring_dir / "score_collection_summary.json")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
