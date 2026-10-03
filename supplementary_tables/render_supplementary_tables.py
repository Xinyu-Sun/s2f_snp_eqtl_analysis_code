#!/usr/bin/env python3
"""Render supplementary-table source TSVs and LaTeX tables from the analysis outputs.

Writes tableS01, S08, S09, S10, S19, S20, S21, S22, S25 and S26 (current Supplementary Table numbering)
to --data-dir and the corresponding LaTeX tables to --tex-dir."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = ["borzoi", "alphagenome"]
TOOL_LABEL = {"borzoi": "Borzoi", "alphagenome": "AlphaGenome", "consensus": "Consensus"}
ANCESTRIES = ["AA", "CH", "NHW"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
TSS_BINS = ["0-3kb", "3-12kb", "12-35kb", ">35kb"]
MAF_BINS = ["lt0.05", "0.05-0.2", "ge0.2"]
MAF_LABEL_TEX = {
    "lt0.05": "MAF $<$ 0.05",
    "0.05-0.2": "0.05 $\\leq$ MAF $<$ 0.2",
    "ge0.2": "MAF $\\geq$ 0.2",
}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def tex_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    if value == 0:
        return "$<10^{-300}$"
    if value < 0.001:
        exponent = math.floor(math.log10(value))
        coefficient = value / (10**exponent)
        return f"${coefficient:.2f} \\times 10^{{{exponent}}}$"
    return f"{value:.3f}"


def tex_ci(mean: float, low: float, high: float, *, separator: str = "--") -> str:
    return f"{mean:.3f} ({low:.3f}{separator}{high:.3f})"


def write_tex(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def table_s1(benchmark: Path, tex_dir: Path, data_dir: Path) -> None:
    frame = read(benchmark / "finemapped_eqtl_pair_counts.tsv")
    frame.to_csv(data_dir / "tableS01_auroc_positive_counts.tsv", sep="\t", index=False)
    order = {tool: i for i, tool in enumerate(TOOLS)}
    anc_order = {ancestry: i for i, ancestry in enumerate(ANCESTRIES)}
    frame["_tool"] = frame["tool"].map(order)
    frame["_anc"] = frame["Group"].map(anc_order)
    frame = frame.sort_values(["_tool", "_anc", "Threshold"])
    lines = [
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Group} & \\textbf{PIP} & \\textbf{Total} & \\textbf{0--3 kb} & \\textbf{3--12 kb} & \\textbf{12--35 kb} & \\textbf{$>$35 kb} \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        threshold = str(row["Threshold"]).replace("≥", "").replace(">=", "")
        distance_cells = [str(row[column]).replace("%", "\\%") for column in TSS_BINS]
        lines.append(
            f"{TOOL_LABEL[row['tool']]} & {row['Group']} & $\\geq$ {threshold} & {int(row['Total'])} & "
            + " & ".join(distance_cells) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    # itertuples renames non-identifier distance columns positionally.
    write_tex(tex_dir / "auroc_positive_eqtl_counts_actual.tex", lines)


def table_s2(benchmark: Path, tex_dir: Path, data_dir: Path) -> None:
    frame = read(benchmark / "positive_variant_maf_distribution.tsv")
    frame.to_csv(data_dir / "tableS09_maf_positive_summary.tsv", sep="\t", index=False)
    indexed = frame.set_index(["ancestry", "pip_threshold", "tool"])
    lines = [
        "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
        "\\textbf{Group} & \\textbf{PIP} & \\multicolumn{3}{c}{\\textbf{Borzoi}} & \\multicolumn{3}{c}{\\textbf{AlphaGenome}} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
        " & & \\textbf{$n$} & \\textbf{Median MAF} & \\textbf{MAF $<$ 0.05 (\\%)} & \\textbf{$n$} & \\textbf{Median MAF} & \\textbf{MAF $<$ 0.05 (\\%)} \\\\",
        "\\midrule",
    ]
    for ancestry in ANCESTRIES:
        for threshold in THRESHOLDS:
            b = indexed.loc[(ancestry, threshold, "borzoi")]
            a = indexed.loc[(ancestry, threshold, "alphagenome")]
            lines.append(
                f"{ancestry} & $\\geq$ {threshold} & {int(b.n_positives)} & {b.maf_median:.3f} & {b.pct_maf_lt_0_05:.1f} & "
                f"{int(a.n_positives)} & {a.maf_median:.3f} & {a.pct_maf_lt_0_05:.1f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_tex(tex_dir / "maf_positive_summary.tex", lines)


def table_s3(benchmark: Path, tex_dir: Path, data_dir: Path) -> None:
    raw = read(benchmark / "susie_auroc_maf_stratified_results.tsv")
    rows = []
    for ancestry in ANCESTRIES:
        for maf_bin in MAF_BINS:
            row = {"ancestry": ancestry, "maf_bin": maf_bin}
            for tool in TOOLS:
                part = raw[(raw["ancestry"] == ancestry) & (raw["maf_bin"] == maf_bin) & (raw["tool"] == tool)]
                row[f"{tool}_mean_auroc"] = part["auc_mean"].mean()
                row[f"{tool}_n_min"] = int(part["n_pos_raw"].min())
                row[f"{tool}_n_max"] = int(part["n_pos_raw"].max())
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "tableS10_maf_auroc_summary.tsv", sep="\t", index=False)
    lines = [
        "\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
        "\\textbf{Group} & \\textbf{MAF bin} & \\multicolumn{2}{c}{\\textbf{Borzoi}} & \\multicolumn{2}{c}{\\textbf{AlphaGenome}} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        " & & \\textbf{Mean AUROC} & \\textbf{$n$ range} & \\textbf{Mean AUROC} & \\textbf{$n$ range} \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.ancestry} & {MAF_LABEL_TEX[row.maf_bin]} & {row.borzoi_mean_auroc:.3f} & "
            f"{row.borzoi_n_min}--{row.borzoi_n_max} & {row.alphagenome_mean_auroc:.3f} & "
            f"{row.alphagenome_n_min}--{row.alphagenome_n_max} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_tex(tex_dir / "maf_auroc_summary.tex", lines)


def table_s4_s6(composition_dir: Path, tex_dir: Path, data_dir: Path) -> None:
    samples = read(composition_dir / "filer_positive_variant_sample_sizes.tsv")
    composition = read(composition_dir / "filer_annotation_overlap_key.tsv")
    pairwise = read(composition_dir / "filer_pairwise_comparisons_key.tsv")
    for name, frame in [
        ("tableS20_functional_annotation_sample_sizes.tsv", samples),
        ("tableS21_functional_annotation_overlap.tsv", composition),
        ("tableS22_functional_annotation_pairwise.tsv", pairwise),
    ]:
        frame.to_csv(data_dir / name, sep="\t", index=False)

    samples["_model"] = samples["model"].map({"alphagenome": 0, "borzoi": 1})
    samples = samples.sort_values(["_model", "pip_threshold"])
    lines = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
             "\\textbf{Model} & \\textbf{PIP threshold} & \\textbf{AA} & \\textbf{CH} & \\textbf{NHW} \\\\", "\\midrule"]
    for row in samples.itertuples(index=False):
        lines.append(f"{TOOL_LABEL[row.model]} & $\\geq {row.pip_threshold}$ & {int(row.AA)} & {int(row.CH)} & {int(row.NHW)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_tex(tex_dir / "functional_annotation_sample_sizes.tex", lines)

    composition["_model"] = composition["model"].map({"alphagenome": 0, "borzoi": 1})
    composition["_category"] = composition["category"].map({c: i for i, c in enumerate(["Accessible_chromatin", "Hi-C", "IM-PET", "pcHi-C", "eQTL", "sQTL"])})
    composition = composition.sort_values(["_model", "pip_threshold", "_category"])
    lines = [
        "\\begin{table}[p]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{\\textbf{Key FILER functional annotation overlap among high-PIP variants.} Values are percentages of high-PIP positive variants overlapping selected FILER categories emphasized in the manuscript. Chi-square $q$ values are Benjamini--Hochberg FDR-adjusted 3$\\times$2 tests comparing AA, CH, and NHW within each model and PIP threshold. For readability, this compact table reports the least and most stringent PIP thresholds.}",
        "\\label{stab:functional_annotation_overlap}",
        "\\begin{tabular}{@{}llp{3.2cm}rrrr@{}}", "\\toprule",
        "\\textbf{Model} & \\textbf{PIP} & \\textbf{Category} & \\textbf{AA (\\%)} & \\textbf{CH (\\%)} & \\textbf{NHW (\\%)} & \\textbf{Chi-square $q$} \\\\", "\\midrule",
    ]
    last_group = None
    for row in composition.itertuples(index=False):
        group = (row.model, row.pip_threshold)
        if last_group is not None and group != last_group:
            lines.append("\\addlinespace")
        lines.append(
            f"{TOOL_LABEL[row.model]} & $\\geq {row.pip_threshold}$ & {row.category_display} & "
            f"{row.AA_percent:.1f} & {row.CH_percent:.1f} & {row.NHW_percent:.1f} & {tex_p(row.chi_square_q)} \\\\"
        )
        last_group = group
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    write_tex(tex_dir / "functional_annotation_overlap_key.tex", lines)

    selected = pairwise[pairwise["fisher_q"].lt(0.10)].copy()
    selected["_model"] = selected["model"].map({"alphagenome": 0, "borzoi": 1})
    selected["_category"] = selected["category"].map({c: i for i, c in enumerate(["Accessible_chromatin", "Hi-C", "IM-PET", "pcHi-C", "eQTL", "sQTL"])})
    selected = selected.sort_values(["pip_threshold", "_category", "_model", "comparison"])
    lines = [
        "\\begin{table}[p]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{\\textbf{Selected pairwise group comparisons for key FILER annotation categories.} Rows show pairwise Fisher exact tests with FDR-adjusted $q < 0.10$ for the compact set of categories discussed in the manuscript at PIP thresholds shown in \\cref{stab:functional_annotation_overlap}. Fractions are percentages of high-PIP variants overlapping each category; difference is ancestry 1 minus ancestry 2 in percentage points.}",
        "\\label{stab:functional_annotation_pairwise}", "\\resizebox{0.98\\textwidth}{!}{%", "\\begin{tabular}{@{}llp{3.0cm}lrrrr@{}}", "\\toprule",
        "\\textbf{Model} & \\textbf{PIP} & \\textbf{Category} & \\textbf{Comparison} & \\textbf{Frac. 1 (\\%)} & \\textbf{Frac. 2 (\\%)} & \\textbf{Diff.} & \\textbf{$q$} \\\\", "\\midrule",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"{TOOL_LABEL[row.model]} & $\\geq {row.pip_threshold}$ & {row.category_display} & {row.comparison} & "
            f"{row.frac1_percent:.1f} & {row.frac2_percent:.1f} & {row.diff_percent_points:+.1f} & {tex_p(row.fisher_q)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    write_tex(tex_dir / "functional_annotation_pairwise_key.tex", lines)


def table_s7_s8(model_first: Path, tex_dir: Path, data_dir: Path) -> None:
    adjusted = read(model_first / "native_ld_adjusted_primary.tsv")
    top = read(model_first / "native_ld_top_vs_rest.tsv")
    regional = read(model_first / "native_ld_regional_pvalues.tsv")
    filters = (
        adjusted["panel"].eq("general") & adjusted["deployment"].eq("native")
        & adjusted["clump_threshold_r2"].eq(0.2) & adjusted["rank_view"].eq("global")
    )
    adjusted = adjusted[filters].copy()
    top = top[
        top["panel"].eq("general") & top["deployment"].eq("native")
        & top["clump_threshold_r2"].eq(0.2) & top["rank_view"].eq("global")
        & top["cutoff_percent"].eq(1.0)
    ].copy()
    adjusted.to_csv(data_dir / "tableS25_native_ld_adjusted.tsv", sep="\t", index=False)
    rows = []
    for ancestry in ANCESTRIES:
        for method in ["borzoi", "alphagenome", "consensus"]:
            a = adjusted[(adjusted["ancestry"] == ancestry) & (adjusted["method"] == method) & (adjusted["evidence_type"] == "association")].iloc[0]
            d = adjusted[(adjusted["ancestry"] == ancestry) & (adjusted["method"] == method) & (adjusted["evidence_type"] == "directional")].iloc[0]
            ta = top[(top["ancestry"] == ancestry) & (top["method"] == method) & (top["evidence_type"] == "association")].iloc[0]
            td = top[(top["ancestry"] == ancestry) & (top["method"] == method) & (top["evidence_type"] == "directional")].iloc[0]
            rows.append((ancestry, method, a, d, ta, td))
    lines = [
        "\\begin{table}[p]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{\\textbf{Adjusted top-versus-rest tests for the model-first analysis after within-group LD clumping.} Association tests use $|z|$ as the outcome, and directional tests use model-sign-aligned $z$ scores. Effect estimates are the adjusted difference between global top-1\\% lead pairs and the remaining LD-clumped background after inverse gene-inclusion weighting, gene fixed effects, TSS-distance terms, MAF terms, and gene-clustered sandwich uncertainty. Reported $p$ values are Holm-adjusted across the nine group-by-method contrasts within each outcome family.}",
        "\\label{stab:native_ld_adjusted}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
        "\\textbf{Group} & \\textbf{Method} & \\textbf{Association probability of superiority} & \\textbf{Adj. $\\Delta |z|$ (95\\% CI)} & \\textbf{Assoc. Holm $p$} & \\textbf{Directional probability of superiority} & \\textbf{Adj. $\\Delta$ aligned $z$ (95\\% CI)} & \\textbf{Dir. Holm $p$} \\\\", "\\midrule",
    ]
    for ancestry, method, a, d, ta, td in rows:
        lines.append(
            f"{ancestry} & {TOOL_LABEL[method]} & {ta.auc_probability_top_has_smaller_p:.3f} & "
            f"{a.top_indicator_estimate:.3f} ({a.ci_low:.3f}, {a.ci_high:.3f}) & {tex_p(a.holm_pvalue_within_9_test_family)} & "
            f"{td.auc_probability_top_has_smaller_p:.3f} & {d.top_indicator_estimate:.3f} ({d.ci_low:.3f}, {d.ci_high:.3f}) & "
            f"{tex_p(d.holm_pvalue_within_9_test_family)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    write_tex(tex_dir / "native_ld_adjusted_top_vs_rest.tex", lines)

    regional = regional[
        regional["panel"].eq("general") & regional["deployment"].eq("native")
        & regional["clump_threshold_r2"].eq(0.2) & regional["rank_view"].eq("global")
        & regional["cutoff_percent"].eq(1.0)
    ].copy()
    summary = regional.groupby("regional_statistic")["auc_probability_top_has_smaller_p"].agg(["min", "max"]).reset_index()
    summary.to_csv(data_dir / "tableS26_native_ld_regional_summaries.tsv", sep="\t", index=False)
    definitions = {
        "minimum_p": ("Minimum regional $p$", "Smallest two-sided regular-eQTL $p$ value among close proxies for the same gene", "Sensitive summary of whether any close proxy carries strong same-gene association evidence; affected by proxy count"),
        "simes_p": ("Simes $p$", "Ordered $p$-value combination across close proxies for the same gene", "Checks whether regional evidence persists beyond selecting the single smallest proxy $p$ value"),
        "acat_p": ("ACAT $p$", "Cauchy combination of close-proxy $p$ values for the same gene", "Summarizes correlated regional evidence while remaining sensitive to one strong or several moderate proxy signals"),
    }
    lines = [
        "\\begin{table}[p]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{\\textbf{Close-proxy summaries for the model-first analysis.} For each lead pair, the close-proxy neighborhood included the lead pair and same-gene variants in within-group LD with the lead SNP ($r^2 \\geq 0.8$). These regional summaries ask whether top-ranked lead pairs tend to fall near same-gene eQTL evidence even when the lead pair itself lacks exact support. Values are descriptive top-versus-rest probability-of-superiority ranges across the nine group-by-method contrasts at the primary top-1\\% cutoff; values greater than 0.5 indicate stronger regional eQTL evidence among top-ranked lead pairs. These summaries are contextual sensitivity analyses rather than primary significance tests.}",
        "\\label{stab:native_ld_regional_summaries}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}llll@{}}", "\\toprule",
        "\\textbf{Regional summary} & \\textbf{Definition} & \\textbf{Top-versus-rest probability-of-superiority range} & \\textbf{Purpose} \\\\", "\\midrule",
    ]
    by_stat = summary.set_index("regional_statistic")
    for statistic in ["minimum_p", "simes_p", "acat_p"]:
        label, definition, purpose = definitions[statistic]
        row = by_stat.loc[statistic]
        lines.append(f"{label} & {definition} & {row['min']:.3f}--{row['max']:.3f} & {purpose} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    write_tex(tex_dir / "native_ld_regional_summaries.tex", lines)


def table_s9(benchmark: Path, tex_dir: Path, data_dir: Path) -> None:
    results = read(benchmark / "credible_set_size_sensitivity_bootstrap.tsv")
    sizes = read(benchmark / "positive_credible_set_size_summary.tsv")
    results.to_csv(data_dir / "tableS08_cs_size_sensitivity.tsv", sep="\t", index=False)
    indexed = results.set_index(["tool", "ancestry", "positive_subset", "mode"])
    lines = [
        "\\begin{tabular}{@{}lllrrcc@{}}", "\\toprule",
        "\\textbf{Model} & \\textbf{Group} & \\textbf{Positive subset} & \\textbf{$n_+$} & \\textbf{CS size} & \\textbf{vs. PIP $<0.01$} & \\textbf{vs. intermediate PIP} \\\\", "\\midrule",
    ]
    for tool in TOOLS:
        for ancestry in ANCESTRIES:
            for label, display, size in [("all", "All", "Any"), ("singleton", "Singleton CS", "1")]:
                standard = indexed.loc[(tool, ancestry, label, "standard")]
                rest = indexed.loc[(tool, ancestry, label, "rest")]
                lines.append(
                    f"{TOOL_LABEL[tool]} & {ancestry} & {display} & {int(standard.n_pos)} & {size} & "
                    f"{tex_ci(standard.auc_mean, standard.auc_ci_low, standard.auc_ci_high)} & "
                    f"{tex_ci(rest.auc_mean, rest.auc_ci_low, rest.auc_ci_high)} \\\\"
                )
            if ancestry != ANCESTRIES[-1]:
                lines.append("\\addlinespace")
        if tool != TOOLS[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_tex(tex_dir / "credible_set_size_sensitivity.tex", lines)


def table_s10(filer_root: Path, tex_dir: Path, data_dir: Path) -> None:
    enrich = read(filer_root / "annotation_enrichment_results.tsv")
    interact = read(filer_root / "ancestry_enrichment_interactions.tsv")
    key = enrich[
        enrich["unit_type"].eq("pair") & enrich["background"].eq("low_pip_lt_0.01")
        & enrich["category"].eq("Accessible_chromatin") & enrich["pip_threshold"].isin([0.5, 0.9])
    ].copy()
    inter = interact[
        interact["unit_type"].eq("pair") & interact["background"].eq("low_pip_lt_0.01")
        & interact["category"].eq("Accessible_chromatin") & interact["pip_threshold"].isin([0.5, 0.9])
    ].copy()
    rows = []
    for model in ["alphagenome", "borzoi"]:
        for threshold in [0.5, 0.9]:
            row = {"model": model, "pip_threshold": threshold, "background": "low_pip_lt_0.01"}
            for ancestry in ANCESTRIES:
                current = key[(key["model"] == model) & (key["pip_threshold"] == threshold) & (key["ancestry"] == ancestry)].iloc[0]
                for field in ["odds_ratio", "ci95_low", "ci95_high"]:
                    row[f"{ancestry}_{field}"] = current[field]
            for contrast in ["AA_vs_CH", "AA_vs_NHW"]:
                current = inter[(inter["model"] == model) & (inter["pip_threshold"] == threshold) & (inter["contrast"] == contrast)].iloc[0]
                for field in ["interaction_odds_ratio", "ci95_low", "ci95_high", "q_value_bh"]:
                    row[f"{contrast}_{field}"] = current[field]
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "tableS19_filer_matched_accessible.tsv", sep="\t", index=False)
    lines = [
        "\\begin{table}[p]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{\\textbf{Matched accessible-chromatin enrichment and group interactions.} Pair-level Mantel--Haenszel odds ratios compare variants above the indicated PIP threshold with group-, model-, TSS-distance-, and PLINK-MAF-matched PIP $<0.01$ comparison variants. Values in the group columns are enrichment odds ratios with 95\\% bootstrap confidence intervals. Interaction values are ratios of group-specific enrichment odds ratios with 95\\% bootstrap confidence intervals; $q$ values are Benjamini--Hochberg adjusted within each unit, model, threshold, and background family across 15 categories and three group contrasts. Interaction ratios above one indicate numerically stronger enrichment in AA.}",
        "\\label{stab:filer_matched_accessible}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
        "\\textbf{Model} & \\textbf{PIP} & \\textbf{AA OR (95\\% CI)} & \\textbf{CH OR (95\\% CI)} & \\textbf{NHW OR (95\\% CI)} & \\textbf{AA/CH ratio (95\\% CI); $q$} & \\textbf{AA/NHW ratio (95\\% CI); $q$} \\\\", "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        ancestry_cells = []
        for ancestry in ANCESTRIES:
            ancestry_cells.append(
                f"{getattr(row, ancestry + '_odds_ratio'):.2f} ({getattr(row, ancestry + '_ci95_low'):.2f}--{getattr(row, ancestry + '_ci95_high'):.2f})"
            )
        contrast_cells = []
        for contrast in ["AA_vs_CH", "AA_vs_NHW"]:
            contrast_cells.append(
                f"{getattr(row, contrast + '_interaction_odds_ratio'):.2f} ({getattr(row, contrast + '_ci95_low'):.2f}--{getattr(row, contrast + '_ci95_high'):.2f}); {tex_p(getattr(row, contrast + '_q_value_bh'))}"
            )
        lines.append(
            f"{TOOL_LABEL[row.model]} & $\\geq {row.pip_threshold}$ & " + " & ".join(ancestry_cells + contrast_cells) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    write_tex(tex_dir / "filer_matched_accessible_chromatin.tex", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--composition-dir", type=Path, required=True)
    parser.add_argument("--model-first-results", type=Path, required=True)
    parser.add_argument("--filer-root", type=Path, required=True)
    parser.add_argument("--tex-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.tex_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    table_s1(args.benchmark_dir, args.tex_dir, args.data_dir)
    table_s2(args.benchmark_dir, args.tex_dir, args.data_dir)
    table_s3(args.benchmark_dir, args.tex_dir, args.data_dir)
    table_s4_s6(args.composition_dir, args.tex_dir, args.data_dir)
    table_s7_s8(args.model_first_results, args.tex_dir, args.data_dir)
    table_s9(args.benchmark_dir, args.tex_dir, args.data_dir)
    table_s10(args.filer_root, args.tex_dir, args.data_dir)
    print(f"[OK] Rendered supplementary tables to {args.tex_dir} and {args.data_dir}")


if __name__ == "__main__":
    main()
