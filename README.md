# Sequence-to-Function SNP-eQTL Analysis Code

Analysis and plotting code for the manuscript *Sequence-to-function models prioritize fine-mapped
blood eQTLs in African American, Caribbean Hispanic, and Non-Hispanic White participants*. The code
benchmarks Borzoi and AlphaGenome variant-effect scores against MAGENTA cis-eQTL mapping and SuSiE
fine-mapping in African American (AA), Caribbean Hispanic (CH), and Non-Hispanic White (NHW)
participants.

The repository is organized by manuscript output. Every figure and supplementary table can be
regenerated from the companion data package (`reproduce_published_figures_and_tables.sh`); the
upstream pipeline steps (model scoring, LD computation, FILER annotation) need the cluster resources
named in each script and are provided to document how the package files were produced.

## Quick start

```bash
# data package unpacked next to this repository as ../data_availability
bash reproduce_published_figures_and_tables.sh

# or point to it explicitly
DATA_DIR=/path/to/data_availability OUT_DIR=/tmp/s2f_outputs PYTHON=/path/to/python \
  bash reproduce_published_figures_and_tables.sh
```

The script reads only from `DATA_DIR`, writes figures and tables to `OUT_DIR`
(default `./regenerated_outputs`), and finishes by comparing every regenerated supplementary-table
source with the file in `DATA_DIR/supplementary_tables/`.

## Layout by manuscript output

| Manuscript item | Code | Data package input |
| --- | --- | --- |
| Figure 1 | `figure1_corr_concordance/make_figure1_corr_concordance.py` | `figure1_corr_concordance/` |
| Figure 2 | `figure2_model_convergence/make_figure2_model_convergence.py` (plotting in `model_convergence_plotting.py`) | `figure2_model_convergence/` |
| Figure 3 | `figure3_distance_matched_auroc/make_figure3_distance_matched_auroc.py` | `figure3_distance_matched_auroc/` |
| Figure 4 | `followup_analyses/a8_make_anatomy_figure.py` | `figure4_score_anatomy/`, `detailed_predictions/` |
| Figure 5 | `model_first_native_ld/code/make_native_ld_figure.py` | `figure5_model_first/` |
| Figure S1 | `figureS1_maf_stratified_auroc/make_figureS1_maf_stratified_auroc.py` | `figureS1_maf_stratified_auroc/` |
| Figure S2 | `model_first_native_ld/code/make_native_ld_distribution_figure.py` | `figureS2_model_first_distributions/` |
| Tables S1, S8-S10, S19-S22, S25, S26 | `supplementary_tables/render_supplementary_tables.py` | `benchmark_results/`, `matched_filer_enrichment/`, `model_first_native_ld/results/` |
| Tables S20-S22 (sources) | `supplementary_tables/build_filer_composition_tables.py` | `matched_filer_enrichment/inputs/`, `matched_filer_enrichment/variant_filer_category_counts.tsv.gz` |
| Table S2 | study-design counts from the eQTL mapping and fine-mapping runs | `supplementary_tables/tableS02_group_design.tsv` |
| Table S3 | `followup_analyses/external_replication/r1_eqtlgen_replication.py` | `followup_analyses/external_replication/` |
| Tables S4-S7, S11-S18, S23, S24 | `followup_analyses/` (analysis scripts) and `followup_analyses/make_followup_tables.py` (LaTeX) | `followup_analyses/results/` |
| Table S27 | `followup_analyses/b5_model_first_lift_contrasts.py` | `model_first_native_ld/results/native_ld_yields.tsv` |

Supplementary tables are numbered as in the Supplementary Information (Tables S1-S27). The data
package holds one source table per supplementary table in `supplementary_tables/`; its README lists
the table titles.

## Pipeline code (run order)

1. **Base benchmark, all groups** (`model_scoring/`). `prepare_base_finemapped_eqtl_batches.py` and
   `prepare_finemapped_eqtl_auroc_batches.py` select fine-mapped positives (PIP >= 0.5) and sampled
   low-PIP and intermediate-PIP comparison variants within group x TSS-distance x MAF strata and write
   scoring chunks; `run_borzoi_chunk.py` / `run_alphagenome_chunk.py` (with the `slurm_*.sh` array
   wrappers) score the chunks; `build_combined_prediction_table.py` assembles the scored table.
   `nominal_eqtl_support/` does the same for the stratified nominal-eQTL sample (10,000 pairs per group
   x TSS-distance cell).
2. **CH inputs and final benchmark tables** (`benchmark_pipeline/`). The CH eQTL and fine-mapping results
   come from the `eqtl_four_population_20260922` release (N = 209):
   1. `extract_ch_inputs.py` - CH nominal sample, CH fine-mapping benchmark, CH pairs of the model-first panel;
   2. `build_scoring_plan.py` - one deduplicated scoring plan; pairs already scored are reused;
   3. `launch_scoring.sh` - Borzoi and AlphaGenome arrays (`slurm/run_*_array.slurm`, which call the
      `model_scoring/` runners), then collection, the CH model-first setup and the benchmark build;
   4. `collect_scores.py` - merges reused and new scores;
   5. `build_benchmark_results.py` - row-level prediction tables and the summary tables behind
      Figures 1-3, S1 and Tables S1, S8-S10;
   6. `build_supporting_inputs.py` - compact supporting inputs (`supporting_inputs/` in the data package);
   7. `build_prediction_row_counts.py` - model-availability flags and row counts of the prediction tables.

   The `slurm/` launchers take site paths from environment variables (`PROJECT_DIR`, `REPO_DIR`,
   `EQTL_RELEASE_DIR`, ...) and `FINEMAPPING_PROFILE`, the label of the regular-eQTL SuSiE profile in the
   release.
3. **Matched FILER enrichment** (`matched_filer_enrichment/`). `plink_maf/` computes exact-QTL-cohort
   PLINK MAF for the CH benchmark variants; `build_filer_inputs.py` stages `inputs/`;
   `run_chain.sh` runs `scripts/01_prepare_and_match.py`, `02_query_filer_tabix.py`,
   `03_run_enrichment.py` and `04_validate_outputs.py` (shared code in `filer_enrichment_lib.py`)
   from a directory that holds `scripts/` and `inputs/`. FILER tracks and `tabix` are required for step 02.
4. **Supplementary tables** (`supplementary_tables/`). `build_filer_composition_tables.py`, then
   `render_supplementary_tables.py`.
5. **Follow-up analyses** (`followup_analyses/`). `a56_make_variant_lists.py` and
   `a56_plink_ld_freq.sbatch` (PLINK LD and allele frequencies; needs genotypes), then
   `run_followup_analyses.sbatch`, which runs B3, B4, B2, B7, A1, A3, A5, A6, A7, A9, A4, A4b, A2, B1, B5,
   A8 (Figure 4) and `make_followup_tables.py` in that order. All scripts share `fu_common.py`
   (data loading and the distance-matched AUROC). `external_replication/r1_eqtlgen_replication.py`
   produces Table S3.
6. **Model-first analysis with within-group LD clumping** (`model_first_native_ld/`). Configuration in
   `config/` (`design.yaml`, `analysis_spec.md`, `ch_inputs.yaml`). `code/prepare_native_ld_inputs.py`
   prepares the AA and NHW LD inputs of the locked 500-gene design; `code/ch_inputs/` builds the CH pair
   manifests and CH LD inputs (`launch_ch_model_first.sh`); then `compute_chromosome_ld.py`,
   `compute_chromosome_clumps.py`, `merge_native_ld_results.py`, `analyze_native_ld_results.py`,
   `fit_native_ld_adjusted.py`, `summarize_native_ld_architecture.py` and the two figure scripts
   (`code/ch_inputs/run_postprocess.slurm` shows the order). The code directory is copied into the
   analysis directory and run from there.

`archive/` holds code that does not produce any manuscript output.

## Data package layout expected by the scripts

```
data_availability/
  figure1_corr_concordance/  figure2_model_convergence/  figure3_distance_matched_auroc/
  figure4_score_anatomy/     figure5_model_first/        figureS1_maf_stratified_auroc/
  figureS2_model_first_distributions/
  supplementary_tables/      tableS01_*.tsv ... tableS27_*.tsv
  benchmark_results/         build_benchmark_results.py summary outputs
  detailed_predictions/      row-level prediction tables
  supporting_inputs/         build_supporting_inputs.py outputs
  matched_filer_enrichment/  inputs/, results, composition/, logs/
  followup_analyses/         results/, external_replication/
  model_first_native_ld/     config/, qc/, results/, figures/, manifests/
```

Default paths inside individual scripts point to the original cluster locations or to placeholders
(`/path/to/...`); when running a script directly, pass the corresponding data-package file on the
command line or through the environment variables documented in the script.

## Dependencies

- Python 3.10 or later with `numpy`, `pandas`, `scipy`, `statsmodels`, `patsy`, `scikit-learn`,
  `matplotlib` and `seaborn`. These are sufficient for `reproduce_published_figures_and_tables.sh`.
- `pyarrow` for the parquet inputs read by `followup_analyses/b1_label_transfer.py` and
  `external_replication/r1_eqtlgen_replication.py`.
- Upstream steps only: PLINK 1.9 (LD and allele frequencies), `tabix` and the FILER track collection
  (annotation queries), Borzoi with TensorFlow on a GPU, and the AlphaGenome Python client with an API key
  (model scoring).

## License

MIT (see `LICENSE`).
