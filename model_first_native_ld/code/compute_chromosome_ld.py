#!/usr/bin/env python3
"""Extract exact target variants from cohort QC PLINK and compute sparse LD."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LD_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_dataframe_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def run(command: list[str], log: list[str], stdin=None, stdout=None) -> None:
    result = subprocess.run(
        command,
        check=False,
        stdin=stdin,
        stdout=stdout if stdout is not None else subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=stdout is None,
    )
    if stdout is None and result.stdout:
        log.append(result.stdout)
    if result.stderr:
        log.append(result.stderr if isinstance(result.stderr, str) else result.stderr.decode())
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def count_ld_edges(path: Path) -> tuple[int, float, float]:
    rows = 0
    minimum = 1.0
    maximum = 0.0
    with gzip.open(path, "rt") as handle:
        header = handle.readline().split()
        if "R2" not in header:
            raise RuntimeError(f"LD output lacks R2 column: {path}")
        r2_index = header.index("R2")
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            value = float(fields[r2_index])
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            rows += 1
    return rows, minimum if rows else float("nan"), maximum if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-manifest", type=Path, default=LD_ROOT / "manifests/ld_chromosome_jobs.tsv")
    parser.add_argument("--job-index", type=int, required=True)
    args = parser.parse_args()

    jobs = pd.read_csv(args.job_manifest, sep="\t")
    selected = jobs.loc[jobs["job_index"].eq(args.job_index)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one job-index row, found {len(selected)}")
    job = selected.iloc[0]
    ancestry = str(job["ancestry"])
    chromosome = int(job["chromosome"])
    expected = pd.read_csv(job["variant_manifest"], sep="\t")
    expected_ids = set(expected["variant_id"].astype(str))
    if len(expected_ids) != int(job["expected_variants"]):
        raise RuntimeError("Expected-variant manifest count mismatch")

    final_prefix = Path(job["output_prefix"])
    qc_path = Path(job["qc_output"])
    coverage_path = qc_path.with_name(qc_path.stem + ".variant_coverage.tsv.gz")
    log_path = LD_ROOT / f"logs/ld_{ancestry}_chr{chromosome}.log"
    final_ld = Path(str(final_prefix) + ".ld.gz")
    final_freq = Path(str(final_prefix) + ".frq")
    final_lmiss = Path(str(final_prefix) + ".lmiss")
    for path in (final_prefix.parent, qc_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    log: list[str] = []
    started = datetime.now(timezone.utc)
    temporary_parent = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(prefix=f"native_ld_{ancestry}_{chromosome}_", dir=temporary_parent) as temp_dir:
        temp = Path(temp_dir)
        plink_prefix = temp / "target"
        freq_prefix = temp / "freq"
        ld_prefix = temp / "sparse"

        run([
            "plink", "--bfile", str(job["bfile_prefix"]),
            "--keep", str(job["sample_file"]),
            "--extract", str(job["extract_ids_file"]), "--keep-allele-order",
            "--make-bed", "--out", str(plink_prefix),
        ], log)
        run([
            "plink", "--bfile", str(plink_prefix), "--keep-allele-order",
            "--freq", "--missing", "--out", str(freq_prefix),
        ], log)
        run([
            "plink", "--bfile", str(plink_prefix), "--keep-allele-order",
            "--r2", "gz", "--ld-window", "999999", "--ld-window-kb", "200",
            "--ld-window-r2", "0.1", "--out", str(ld_prefix),
        ], log)

        bim = pd.read_csv(
            Path(str(plink_prefix) + ".bim"), sep=r"\s+", header=None,
            names=["chromosome", "variant_id", "cm", "position", "a1", "a2"],
        )
        if bim["variant_id"].duplicated().any():
            raise RuntimeError("Duplicate exact variant IDs after PLINK extraction")
        observed_ids = set(bim["variant_id"].astype(str))
        unexpected_ids = observed_ids - expected_ids
        if unexpected_ids:
            raise RuntimeError(f"Unexpected exact variants after PLINK extraction: {len(unexpected_ids)}")
        allele_check = expected[["variant_id", "ref_allele", "alt_allele"]].merge(
            bim[["variant_id", "a1", "a2"]], on="variant_id", how="left", validate="one_to_one"
        )
        allele_check["plink_alt_ref_order_valid"] = (
            allele_check["a1"].astype(str).eq(allele_check["alt_allele"].astype(str))
            & allele_check["a2"].astype(str).eq(allele_check["ref_allele"].astype(str))
        )

        freq = pd.read_csv(Path(str(freq_prefix) + ".frq"), sep=r"\s+")
        # With --keep-allele-order, PLINK keeps A1 as the source ALT allele.
        # The legacy .frq column is still named MAF, but in this mode it is the
        # A1 (ALT) frequency and can exceed 0.5. Convert it explicitly to MAF.
        freq = freq.rename(columns={
            "SNP": "variant_id", "MAF": "genotype_alt_frequency",
            "NCHROBS": "observed_alleles",
        })
        freq["genotype_maf"] = np.minimum(
            freq["genotype_alt_frequency"].astype(float),
            1.0 - freq["genotype_alt_frequency"].astype(float),
        )
        missing = pd.read_csv(Path(str(freq_prefix) + ".lmiss"), sep=r"\s+")
        missing = missing.rename(columns={"SNP": "variant_id", "F_MISS": "genotype_missing_fraction"})
        coverage = expected.merge(
            freq[["variant_id", "genotype_alt_frequency", "genotype_maf", "observed_alleles"]],
            on="variant_id", how="left", validate="one_to_one"
        ).merge(
            missing[["variant_id", "genotype_missing_fraction"]], on="variant_id", how="left", validate="one_to_one"
        ).merge(
            allele_check[["variant_id", "plink_alt_ref_order_valid"]],
            on="variant_id", how="left", validate="one_to_one",
        )
        coverage["ld_genotype_present"] = coverage["variant_id"].isin(observed_ids).astype(int)
        coverage["estimated_genotype_variance"] = (
            2.0 * coverage["genotype_maf"].astype(float)
            * (1.0 - coverage["genotype_maf"].astype(float))
        )
        coverage["ld_calculable"] = (
            coverage["ld_genotype_present"].eq(1)
            & coverage["plink_alt_ref_order_valid"].fillna(False)
            & coverage["observed_alleles"].ge(40)
            & coverage["estimated_genotype_variance"].gt(1.0e-8)
        ).astype(int)
        coverage["maf_absolute_difference"] = (
            coverage["genotype_maf"].astype(float) - coverage["eqtl_maf"].astype(float)
        ).abs()
        atomic_dataframe_gzip(coverage, coverage_path)

        source_ld = Path(str(ld_prefix) + ".ld.gz")
        source_freq = Path(str(freq_prefix) + ".frq")
        source_lmiss = Path(str(freq_prefix) + ".lmiss")
        if not source_ld.is_file():
            raise RuntimeError("PLINK did not create sparse LD output")
        atomic_copy(source_ld, final_ld)
        atomic_copy(source_freq, final_freq)
        atomic_copy(source_lmiss, final_lmiss)

    matched = int(coverage["ld_genotype_present"].sum())
    allele_valid = int(coverage["plink_alt_ref_order_valid"].fillna(False).sum())
    ld_calculable = int(coverage["ld_calculable"].sum())
    maf_valid = coverage[["eqtl_maf", "genotype_maf", "maf_absolute_difference"]].dropna()
    maf_corr = float(maf_valid[["eqtl_maf", "genotype_maf"]].corr().iloc[0, 1]) if len(maf_valid) >= 2 else float("nan")
    maf_median_abs_diff = float(maf_valid["maf_absolute_difference"].median())
    maf_q95_abs_diff = float(maf_valid["maf_absolute_difference"].quantile(0.95))
    edge_rows, edge_min_r2, edge_max_r2 = count_ld_edges(final_ld)
    sample_count = sum(1 for line in Path(job["sample_file"]).read_text().splitlines() if line.strip())
    ready = (
        matched / len(expected_ids) >= 0.95
        and allele_valid == matched
        and np.isfinite(maf_corr) and maf_corr >= 0.90
        and maf_median_abs_diff <= 0.05
        and (edge_rows == 0 or (edge_min_r2 >= 0.099999 and edge_max_r2 <= 1.000001))
    )
    completed = datetime.now(timezone.utc)
    qc = {
        "ancestry": ancestry,
        "chromosome": chromosome,
        "created_at": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "sample_count": sample_count,
        "ch_definition": "IHG_only" if ancestry == "CH" else None,
        "per_excluded": ancestry == "CH",
        "expected_exact_variants": len(expected_ids),
        "matched_exact_variants": matched,
        "unmatched_exact_variants": len(expected_ids) - matched,
        "variant_match_fraction": matched / len(expected_ids),
        "exact_plink_alt_ref_order_variants": allele_valid,
        "exact_plink_alt_ref_order_fraction_of_matched": allele_valid / matched if matched else 0.0,
        "plink_a1_interpreted_as_alt_frequency": True,
        "genotype_maf_definition": "min(PLINK_A1_ALT_frequency, 1-PLINK_A1_ALT_frequency)",
        "minimum_nonmissing_samples": 20,
        "minimum_genotype_variance": 1.0e-8,
        "ld_calculable_variants": ld_calculable,
        "ld_calculable_fraction_of_expected": ld_calculable / len(expected_ids),
        "excluded_low_nonmissing_or_negligible_variance": matched - ld_calculable,
        "maf_valid_variants": int(len(maf_valid)),
        "maf_pearson_correlation_with_eqtl": maf_corr,
        "maf_median_absolute_difference": maf_median_abs_diff,
        "maf_q95_absolute_difference": maf_q95_abs_diff,
        "maximum_genotype_missing_fraction": float(coverage["genotype_missing_fraction"].max()),
        "sparse_ld_edge_floor_r2": 0.1,
        "maximum_physical_distance_bp": 200000,
        "sparse_ld_edges": edge_rows,
        "minimum_reported_r2": edge_min_r2,
        "maximum_reported_r2": edge_max_r2,
        "ld_output": str(final_ld),
        "ld_output_sha256": sha256_file(final_ld),
        "frequency_output": str(final_freq),
        "frequency_output_sha256": sha256_file(final_freq),
        "missingness_output": str(final_lmiss),
        "missingness_output_sha256": sha256_file(final_lmiss),
        "variant_coverage_output": str(coverage_path),
        "variant_coverage_output_sha256": sha256_file(coverage_path),
        "ready": bool(ready),
    }
    atomic_text(qc_path, json.dumps(qc, indent=2, sort_keys=True) + "\n")
    atomic_text(log_path, "\n".join(log))
    if not ready:
        raise RuntimeError(f"Chromosome LD QC failed: {json.dumps(qc, sort_keys=True)}")
    print(json.dumps(qc, sort_keys=True))


if __name__ == "__main__":
    main()
