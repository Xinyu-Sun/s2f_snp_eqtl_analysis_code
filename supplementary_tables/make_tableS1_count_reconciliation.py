#!/usr/bin/env python3
"""Build a compact attrition table for heatmap and redo AUROC counts.

The output is intended for manuscript/reviewer reporting. Counts are
SNP-gene pairs, not unique variants.
"""

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path


TOOLS = ("borzoi", "alphagenome")
ANCESTRIES = ("AA", "CH", "NHW")
TSS_BINS = ("0-3kb", "3-12kb", "12-35kb", ">35kb")
NOMINAL_ANCESTRY_MAP = {
    "AA_specific": "AA",
    "CH_specific": "CH",
    "NHW_specific": "NHW",
}


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), "rt")
    return open(str(path), "r")


def read_tsv(path):
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def int_value(value):
    if value in (None, ""):
        return 0
    return int(float(value))


def format_count(value):
    if value in (None, ""):
        return ""
    return str(int(value))


def count_nominal_starting_pairs(path):
    rows_by_ancestry = defaultdict(int)
    row_keys_by_ancestry = defaultdict(list)
    ancestry_sets_by_key = defaultdict(set)
    key_cols = ("chr", "pos", "a2", "a1", "gene_id")

    for row in read_tsv(path):
        ancestry = row.get("population", "")
        if ancestry not in ("AA", "CH", "NHW", "PER"):
            continue
        rows_by_ancestry[ancestry] += 1
        key = tuple(row.get(col, "") for col in key_cols)
        row_keys_by_ancestry[ancestry].append(key)
        ancestry_sets_by_key[key].add(ancestry)

    counts = {}
    notes = {}
    for ancestry in ANCESTRIES:
        n_rows = rows_by_ancestry.get(ancestry, 0)
        n_specific = sum(
            1
            for key in row_keys_by_ancestry.get(ancestry, [])
            if ancestry_sets_by_key.get(key, set()) == set([ancestry])
        )
        counts[ancestry] = n_specific
        notes[ancestry] = (
            "Starting count is %s-specific row-level SNP-gene pairs derived from the raw nominal file; "
            "raw %s population rows before ancestry-sharing classification = %d."
        ) % (ancestry, ancestry, n_rows)
    return counts, notes


def sum_heatmap_counts(path, ancestry_map):
    counts = defaultdict(int)
    cells = {}

    for row in read_tsv(path):
        tss_bin = row.get("tss_distance_bin", "")
        if tss_bin not in TSS_BINS:
            continue
        tool = row.get("tool", "")
        ancestry = ancestry_map.get(row.get("ancestry", ""), row.get("ancestry", ""))
        if tool not in TOOLS or ancestry not in ANCESTRIES:
            continue
        n = int_value(row.get("n"))
        key = (tool, ancestry, tss_bin)
        cells[key] = n
        counts[(tool, ancestry)] += n

    return counts, cells


def load_heatmap_counts(correlation_path, concordance_path, ancestry_map, dataset_label):
    corr_counts, corr_cells = sum_heatmap_counts(correlation_path, ancestry_map)
    conc_counts, conc_cells = sum_heatmap_counts(concordance_path, ancestry_map)

    validation_rows = []
    totals = {}
    has_mismatch = False

    for tool in TOOLS:
        for ancestry in ANCESTRIES:
            corr_total = 0
            conc_total = 0
            cell_mismatches = []
            for tss_bin in TSS_BINS:
                key = (tool, ancestry, tss_bin)
                corr_n = corr_cells.get(key, 0)
                conc_n = conc_cells.get(key, 0)
                corr_total += corr_n
                conc_total += conc_n
                if corr_n != conc_n:
                    cell_mismatches.append("%s:%d vs %d" % (tss_bin, corr_n, conc_n))

            status = "pass" if corr_total == conc_total and not cell_mismatches else "fail"
            if status == "fail":
                has_mismatch = True
            validation_rows.append(
                {
                    "validation": "correlation_vs_concordance_heatmap_n",
                    "dataset": dataset_label,
                    "tool": tool,
                    "ancestry": ancestry,
                    "correlation_n": corr_total,
                    "concordance_n": conc_total,
                    "delta": corr_total - conc_total,
                    "status": status,
                    "notes": "; ".join(cell_mismatches)
                    if cell_mismatches
                    else "Correlation and concordance heatmap counts match across plotted TSS bins.",
                }
            )
            totals[(tool, ancestry)] = corr_total

    return totals, validation_rows, has_mismatch


def load_finemapped_starting_pairs(manifest_path):
    by_tool = defaultdict(dict)
    for row in read_tsv(manifest_path):
        tool = row.get("tool", "")
        ancestry = row.get("ancestry", "")
        if tool not in TOOLS or ancestry not in ANCESTRIES:
            continue
        by_tool[ancestry][tool] = by_tool[ancestry].get(tool, 0) + int_value(row.get("n_pairs"))

    counts = {}
    notes = {}
    for ancestry in ANCESTRIES:
        tool_counts = by_tool.get(ancestry, {})
        values = [tool_counts.get(tool, 0) for tool in TOOLS]
        counts[ancestry] = max(values) if values else 0
        if len(set(values)) <= 1:
            notes[ancestry] = "Both model manifests requested the same fine-mapped pair count."
        else:
            notes[ancestry] = "Model manifests differ: borzoi=%d, alphagenome=%d." % (
                tool_counts.get("borzoi", 0),
                tool_counts.get("alphagenome", 0),
            )
    return counts, notes


def load_redo_positive_starting_pairs(summary_path):
    counts = {}
    for row in read_tsv(summary_path):
        if row.get("redo_label") != "positive_pip_ge_0.5_in_original_table":
            continue
        ancestry = row.get("ancestry", "")
        if ancestry in ANCESTRIES:
            counts[ancestry] = int_value(row.get("n"))
    return counts


def load_redo_pip09_counts(path):
    counts = {}
    for row in read_tsv(path):
        threshold = row.get("Threshold", "")
        tool = row.get("tool", "")
        ancestry = row.get("Group", "")
        if "0.9" not in threshold or tool not in TOOLS or ancestry not in ANCESTRIES:
            continue
        counts[(tool, ancestry)] = int_value(row.get("Total"))
    return counts


def make_table_rows(
    nominal_start,
    nominal_notes,
    nominal_heatmap,
    fine_start,
    fine_notes,
    fine_heatmap,
    redo_start,
    redo_pip09,
):
    rows = []

    for ancestry in ANCESTRIES:
        rows.append(
            {
                "dataset": "Nominal eQTLs",
                "ancestry": ancestry,
                "starting_pairs": nominal_start.get(ancestry, 0),
                "filter_step": "Retained in correlation/sign-concordance heatmaps after model score availability and plotted TSS-bin filtering",
                "borzoi_pairs": nominal_heatmap.get(("borzoi", ancestry), 0),
                "alphagenome_pairs": nominal_heatmap.get(("alphagenome", ancestry), 0),
                "borzoi_reconciliation_delta": "",
                "alphagenome_reconciliation_delta": "",
                "notes": nominal_notes.get(ancestry, "")
                + " Unknown TSS bins are excluded because they are not plotted in the heatmaps.",
            }
        )

    for ancestry in ANCESTRIES:
        rows.append(
            {
                "dataset": "Fine-mapped PIP >= 0.9",
                "ancestry": ancestry,
                "starting_pairs": fine_start.get(ancestry, 0),
                "filter_step": "Retained in correlation/sign-concordance heatmaps after model score availability and plotted TSS-bin filtering",
                "borzoi_pairs": fine_heatmap.get(("borzoi", ancestry), 0),
                "alphagenome_pairs": fine_heatmap.get(("alphagenome", ancestry), 0),
                "borzoi_reconciliation_delta": "",
                "alphagenome_reconciliation_delta": "",
                "notes": fine_notes.get(ancestry, "")
                + " Fine-mapped heatmap scripts omit plotted cells with n < 10.",
            }
        )

    for ancestry in ANCESTRIES:
        rows.append(
            {
                "dataset": "Redo AUROC fine-mapped positives",
                "ancestry": ancestry,
                "starting_pairs": redo_start.get(ancestry, 0),
                "filter_step": "Retained as PIP >= 0.9 positives in redo AUROC table",
                "borzoi_pairs": redo_pip09.get(("borzoi", ancestry), 0),
                "alphagenome_pairs": redo_pip09.get(("alphagenome", ancestry), 0),
                "borzoi_reconciliation_delta": "",
                "alphagenome_reconciliation_delta": "",
                "notes": "Starting count is the redo AUROC PIP >= 0.5 positive source. PIP >= 0.9 follows the source table threshold label.",
            }
        )

    for ancestry in ANCESTRIES:
        borzoi_delta = fine_heatmap.get(("borzoi", ancestry), 0) - redo_pip09.get(("borzoi", ancestry), 0)
        alphagenome_delta = (
            fine_heatmap.get(("alphagenome", ancestry), 0)
            - redo_pip09.get(("alphagenome", ancestry), 0)
        )
        rows.append(
            {
                "dataset": "Fine-mapped heatmap minus redo AUROC PIP >= 0.9",
                "ancestry": ancestry,
                "starting_pairs": "",
                "filter_step": "Reconciliation delta: heatmap retained count minus redo AUROC PIP >= 0.9 count",
                "borzoi_pairs": "",
                "alphagenome_pairs": "",
                "borzoi_reconciliation_delta": borzoi_delta,
                "alphagenome_reconciliation_delta": alphagenome_delta,
                "notes": "Positive values mean the heatmap retained more pairs; negative values mean the redo AUROC table retained more pairs.",
            }
        )

    return rows


def append_reconciliation_validation(validation_rows, fine_heatmap, redo_pip09):
    for tool in TOOLS:
        for ancestry in ANCESTRIES:
            heatmap_n = fine_heatmap.get((tool, ancestry), 0)
            redo_n = redo_pip09.get((tool, ancestry), 0)
            delta = heatmap_n - redo_n
            validation_rows.append(
                {
                    "validation": "fine_mapped_heatmap_vs_redo_auroc_pip_ge_0.9",
                    "dataset": "Fine-mapped PIP >= 0.9",
                    "tool": tool,
                    "ancestry": ancestry,
                    "correlation_n": heatmap_n,
                    "concordance_n": redo_n,
                    "delta": delta,
                    "status": "match" if delta == 0 else "delta_flagged",
                    "notes": "comparison is heatmap retained count minus redo AUROC PIP >= 0.9 count",
                }
            )


def write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows, fieldnames):
    display_rows = []
    for row in rows:
        display_rows.append([str(row.get(field, "")) for field in fieldnames])

    widths = []
    for i, field in enumerate(fieldnames):
        widths.append(max([len(field)] + [len(row[i]) for row in display_rows]))

    def fmt(values):
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    lines = [fmt(fieldnames), fmt(["-" * width for width in widths])]
    for row in display_rows:
        lines.append(fmt(row))
    return "\n".join(lines) + "\n"


def write_markdown(path, rows, fieldnames):
    with open(str(path), "w") as handle:
        handle.write(markdown_table(rows, fieldnames))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--common-base",
        type=Path,
        default=None,
        help="Path to the companion eQTL benchmark base directory. Defaults to sibling of base-dir.",
    )
    parser.add_argument(
        "--fine-mapped-results-name",
        default="fine_mapped_eqtl_auroc_results",
        help="Fine-mapped AUROC result directory under base-dir.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    common_base = args.common_base.resolve() if args.common_base else base_dir.parent / "companion_eqtl_benchmark"
    redo_analysis = base_dir / args.fine_mapped_results_name / "analysis"

    nominal_start, nominal_notes = count_nominal_starting_pairs(
        common_base / "inputs" / "MAGENTA_All_Significant_Nominal_eQTLs_v2.tsv.gz"
    )
    nominal_heatmap, nominal_validation, nominal_mismatch = load_heatmap_counts(
        common_base / "outputs" / "analysis" / "stratified_correlations.tsv",
        common_base / "outputs" / "analysis" / "stratified_concordance.tsv",
        NOMINAL_ANCESTRY_MAP,
        "Nominal eQTLs",
    )
    fine_start, fine_notes = load_finemapped_starting_pairs(
        common_base / "outputs_finemapped" / "manifests" / "jobs.tsv"
    )
    fine_heatmap, fine_validation, fine_mismatch = load_heatmap_counts(
        common_base / "outputs_finemapped" / "analysis" / "finemapped_stratified_correlations.tsv",
        common_base / "outputs_finemapped" / "analysis" / "finemapped_stratified_concordance.tsv",
        {},
        "Fine-mapped PIP >= 0.9",
    )
    redo_start = load_redo_positive_starting_pairs(redo_analysis / "selected_pair_summary.tsv")
    redo_pip09 = load_redo_pip09_counts(
        redo_analysis / "manuscript_positive_tss_threshold_counts_deduplicated.tsv"
    )

    rows = make_table_rows(
        nominal_start,
        nominal_notes,
        nominal_heatmap,
        fine_start,
        fine_notes,
        fine_heatmap,
        redo_start,
        redo_pip09,
    )
    validation_rows = nominal_validation + fine_validation
    append_reconciliation_validation(validation_rows, fine_heatmap, redo_pip09)

    table_fields = [
        "dataset",
        "ancestry",
        "starting_pairs",
        "filter_step",
        "borzoi_pairs",
        "alphagenome_pairs",
        "borzoi_reconciliation_delta",
        "alphagenome_reconciliation_delta",
        "notes",
    ]
    validation_fields = [
        "validation",
        "dataset",
        "tool",
        "ancestry",
        "correlation_n",
        "concordance_n",
        "delta",
        "status",
        "notes",
    ]

    table_path = redo_analysis / "tableS1_count_reconciliation.tsv"
    markdown_path = redo_analysis / "tableS1_count_reconciliation.md"
    validation_path = redo_analysis / "tableS1_count_reconciliation_validation.tsv"
    write_tsv(table_path, rows, table_fields)
    write_markdown(markdown_path, rows, table_fields)
    write_tsv(validation_path, validation_rows, validation_fields)

    print("[ok] wrote %s" % table_path)
    print("[ok] wrote %s" % markdown_path)
    print("[ok] wrote %s" % validation_path)

    if nominal_mismatch or fine_mismatch:
        raise SystemExit("Correlation and concordance heatmap n values do not match; see validation TSV.")

    flagged = [row for row in validation_rows if row["status"] == "delta_flagged"]
    if flagged:
        print("[warn] %d fine-mapped heatmap vs redo AUROC deltas flagged; see validation TSV." % len(flagged))


if __name__ == "__main__":
    main()
