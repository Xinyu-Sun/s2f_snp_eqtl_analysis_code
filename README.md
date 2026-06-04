# Sequence-to-Function SNP-eQTL Analysis Code

This repository contains the analysis and plotting code for the manuscript
benchmarking Borzoi and AlphaGenome variant-effect predictions against
MAGENTA ancestry-stratified eQTL results.

The repository is organized by manuscript output. Scripts expect the companion
data-availability package to provide the corresponding input tables.

## Reproducing Outputs

Run the top-level script from the repository root:

```bash
bash reproduce_published_figures_and_tables.sh
```

By default, the runner expects the companion data package at
`../data_availability`. To use a different location:

```bash
DATA_DIR=/path/to/data_availability bash reproduce_published_figures_and_tables.sh
```

The runner uses `python3` by default. Set `PYTHON=/path/to/python` to use a
specific Python interpreter.

## Manuscript Figure Scripts

- `figure1_corr_concordance/make_figure1_corr_concordance.py`: regenerates
  Figure 1, showing S2F model correlation and direction concordance with eQTL
  effect sizes across TSS-distance and ancestry strata.
- `figure2_model_convergence/make_figure2_model_convergence.py`: regenerates
  Figure 2, showing inter-model convergence between Borzoi and AlphaGenome.
- `figure3_distance_matched_auroc/make_figure3_distance_matched_auroc.py`:
  regenerates Figure 3, the distance-matched AUROC bootstrap boxplot.
- `figureS1_maf_stratified_auroc/make_figureS1_maf_stratified_auroc.py`:
  regenerates Supplementary Figure S1, the MAF-stratified distance-matched
  AUROC analysis.

## Supporting Analysis Code

- `model_scoring/`: batch preparation, model-scoring wrappers, and prediction
  table construction for the fine-mapped eQTL benchmark.
- `nominal_eqtl_support/`: nominal-eQTL benchmark batch preparation, result
  collection, and plot-input construction.
- `negative_control_sensitivity/`: scripts for negative-control contamination
  and rerun-confirmed sensitivity analyses.
- `supplementary_tables/`: scripts for derived manuscript summary tables.

## Python Dependencies

The plotting and table scripts use standard scientific Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`

Some model-scoring wrappers require additional model-specific environments for
Borzoi or AlphaGenome inference. The plotting scripts only require the summary
tables distributed with the companion data package.
