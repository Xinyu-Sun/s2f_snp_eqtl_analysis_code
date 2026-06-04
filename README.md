# Code Availability

This folder is organized by manuscript output rather than by internal analysis
workspace.

Figures 1 and 2 are not included here because they were generated on the local
machine and were explicitly skipped for this packaging pass.

## Main Entry Point

`reproduce_published_figures_and_tables.sh` redraws the packaged Figure 3 and
Supplementary Figure S1 panels from the packaged data files, then regenerates
the derived Supplementary Tables S2-S4.

The runner defaults to `python3`; set `PYTHON=/path/to/python` before
running it to use a different compatible Python.

The plotting scripts are:

- `figure3_distance_matched_auroc/make_figure3_distance_matched_auroc.py`
- `figureS1_maf_stratified_auroc/make_figureS1_maf_stratified_auroc.py`

## Folder Guide

- `figure3_distance_matched_auroc/`: code for the distance-matched AUROC
  bootstrap boxplot in Figure 3.
- `figureS1_maf_stratified_auroc/`: code for the MAF-stratified
  distance-matched AUROC analysis and Supplementary Figure S1.
- `supplementary_tables/`: scripts for the positive-set summaries and
  manuscript table summaries.
- `model_scoring/`: batch preparation, model-scoring wrappers, and result
  combination code for the fine-mapped eQTL benchmark.
- `nominal_eqtl_support/`: companion nominal-eQTL benchmark support code from
  the same manuscript project. Final Figure 1/2 plotting is not included here.
- `negative_control_sensitivity/`: scripts for negative-control contamination
  and rerun-confirmed sensitivity summaries.

## Notes

The code uses the paper-facing file names in this package. Some source tables
retain historical provenance columns used during analysis; those columns are
not part of the public folder or file naming.
