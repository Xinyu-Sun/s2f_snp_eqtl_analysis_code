#!/usr/bin/env python3
"""Write the source TSVs of the follow-up supplementary tables (Tables S2-S7, S11-S17, S21, S22 and S25).

Each table source is one TSV; a table drawn from several result files stacks them with a `source_table` column.
Table S2 holds the study-design counts of the eQTL mapping and fine-mapping runs (listed below); Table S3 is the
eQTLGen replication summary written by followup_analyses/external_replication/r1_eqtlgen_replication.py.

Usage: write_followup_table_sources.py --results <followup results dir> --eqtlgen <eqtlgen_replication.tsv> --output-dir <dir>
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--eqtlgen", type=Path, required=True)
ap.add_argument("--output-dir", type=Path, required=True)
args = ap.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)


def rd(name):
    return pd.read_csv(args.results / f"{name}.tsv", sep="\t")


def stack(parts):
    frames = []
    for name, frame in parts:
        frame = frame.dropna(axis=1, how="all").copy()
        frame.insert(0, "source_table", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def stack_files(names):
    return stack([(n, rd(n)) for n in names])


ld = rd("a5_ld_partner_models_and_strata")
TABLES = {
    "tableS04_joint_bootstrap.tsv": stack_files(["a2_auroc_bootstrap", "a2_group_differences", "a2_cross_pool_auroc"]),
    "tableS05_logistic_slopes.tsv": stack_files(["b3_group_slopes", "b3_slope_contrasts", "b3_leave_one_gene_out_AA"]),
    "tableS06_subsets_pip_weighting.tsv": stack_files(["b7_random_subsets", "b7_pip_weighted_auroc", "b7_pip_weighted_differences"]),
    "tableS07_score_anatomy.tsv": stack_files(["a1_score_by_class", "a1_pos_vs_low_ratio", "a1_group_contrasts", "a1_positive_characteristics"]),
    "tableS11_shared_genes.tsv": rd("a7_shared_gene_auroc"),
    "tableS12_within_gene.tsv": stack_files(["b2_within_gene", "b2_within_gene_differences"]),
    "tableS13_matched_positives.tsv": stack_files(["a4_matched_positives", "a4b_matching_covariate_sets"]),
    "tableS14_decile_enrichment.tsv": rd("a3_decile_enrichment"),
    # Table S15 has two parts: (A) LD proxies and AUROC by LD stratum, (B) group regressions adjusted for LD proxies
    "tableS15_ld_proxies.tsv": stack([
        ("a5_ld_partner_summary", rd("a5_ld_partner_summary")),
        ("a5_ld_partner_strata", ld[ld["stratum"].notna()]),
        ("a5_ld_partner_regressions", ld[ld["contrast"].notna()]),
    ]),
    "tableS16_freq_class.tsv": rd("a6_freq_class_summary"),
    # Table S17 has two parts: (A) label transfer and AUROC by sharing, (B) matched shared versus group-specific positives
    "tableS17_label_transfer.tsv": stack_files(["b1_label_transfer_summary", "b1_auroc_by_sharing", "b8_matched_auroc", "b8_pooled", "b8_regression"]),
    "tableS21_accessible.tsv": stack_files(["a9_accessible_summary", "a9_accessible_regression", "a9_accessible_auroc"]),
    "tableS22_annotation_baselines.tsv": stack_files(["b4_baseline_auroc", "b4_baseline_group_differences"]),
}
for name, frame in TABLES.items():
    frame.to_csv(args.output_dir / name, sep="\t", index=False)
shutil.copyfile(args.results / "b5_model_first_lift_contrasts.tsv", args.output_dir / "tableS25_model_first_lift.tsv")
shutil.copyfile(args.eqtlgen, args.output_dir / "tableS03_eqtlgen_replication.tsv")

# Study design of the eQTL mapping and fine-mapping runs (participant files and per-group association results are not public)
design = [
    ("Participants in the association model", "n", 224, 209, 235),
    ("Alzheimer disease cases", "n", 111, 108, 118),
    ("Alzheimer disease controls", "n", 113, 101, 117),
    ("Female participants", "percent", 72, 73, 53),
    ("Age at examination, mean", "years", 76.6, 78.4, 77.6),
    ("Age at examination, SD", "years", 7.8, 7.3, 7.4),
    ("Genes tested", "n", 21049, 20960, 20955),
    ("Variant-gene pairs tested", "million pairs", 332.8, 244.0, 225.8),
    ("Median variants per tested gene", "n", 13790, 10130, 9298),
    ("Covariates, total", "n", 61, 60, 62),
    ("Covariates, age, sex and AD status", "n", 3, 3, 3),
    ("Covariates, genotype principal components", "n", 3, 3, 3),
    ("Covariates, cell fractions", "n", 14, 18, 14),
    ("Covariates, hidden factors", "n", 41, 36, 42),
    ("Residual degrees of freedom", "n", 161, 147, 171),
    ("Significant variant-gene pairs (p < 2.02e-5 and within-gene q < 0.05)", "n", 64150, 401287, 290846),
    ("eGenes", "n", 2421, 4913, 3722),
    ("eGenes with a credible set", "n", 654, 2595, 1839),
    ("Credible-set variants (unique variant-gene pairs in a 95% credible set)", "n", 15433, 106702, 85885),
    ("Credible-set variants per gene, median", "n", 12, 12, 18),
    ("High-PIP variant-gene pairs (PIP >= 0.9, in a credible set)", "n", 83, 347, 152),
]
pd.DataFrame(
    [[c, u] + [str(v) for v in vals] for c, u, *vals in design],
    columns=["characteristic", "unit", "AA", "CH", "NHW"],
).to_csv(args.output_dir / "tableS02_group_design.tsv", sep="\t", index=False)
print("wrote", len(TABLES) + 3, "table sources to", args.output_dir)
