#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../data_availability}"
PYTHON="${PYTHON:-python3}"

FIGURE_OUT="${DATA_DIR}/regenerated_figures"
TABLE_OUT="${DATA_DIR}/regenerated_tables"

mkdir -p "${FIGURE_OUT}/figure1_corr_concordance"
mkdir -p "${FIGURE_OUT}/figure2_model_convergence"
mkdir -p "${FIGURE_OUT}/figure3_distance_matched_auroc"
mkdir -p "${FIGURE_OUT}/figureS1_maf_stratified_auroc"
mkdir -p "${TABLE_OUT}"

"${PYTHON}" "${SCRIPT_DIR}/figure1_corr_concordance/make_figure1_corr_concordance.py" \
  --nominal_tool_performance "${DATA_DIR}/figure1_corr_concordance/nominal_tool_performance_results.tsv" \
  --finemapped_correlations "${DATA_DIR}/figure1_corr_concordance/finemapped_stratified_correlations.tsv" \
  --finemapped_concordance "${DATA_DIR}/figure1_corr_concordance/finemapped_stratified_concordance.tsv" \
  --output_dir "${FIGURE_OUT}/figure1_corr_concordance"

"${PYTHON}" "${SCRIPT_DIR}/figure2_model_convergence/make_figure2_model_convergence.py" \
  --model_convergence_results "${DATA_DIR}/figure2_model_convergence/model_convergence_results.tsv" \
  --output_dir "${FIGURE_OUT}/figure2_model_convergence"

"${PYTHON}" "${SCRIPT_DIR}/figure3_distance_matched_auroc/make_figure3_distance_matched_auroc.py" \
  --plot_from_results "${DATA_DIR}/figure3_distance_matched_auroc/figure3_distance_matched_auroc_bootstrap.tsv" \
  --output_dir "${FIGURE_OUT}/figure3_distance_matched_auroc"

"${PYTHON}" "${SCRIPT_DIR}/figureS1_maf_stratified_auroc/make_figureS1_maf_stratified_auroc.py" \
  --plot_from_bootstrap_values "${DATA_DIR}/figureS1_maf_stratified_auroc/figureS1_maf_stratified_auroc_bootstrap.tsv.gz" \
  --output_dir "${FIGURE_OUT}/figureS1_maf_stratified_auroc"

"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/make_tableS2_positive_maf_summary.py" \
  --input "${DATA_DIR}/supporting_inputs/deduplicated_positive_variants_for_filer.tsv" \
  --output "${TABLE_OUT}/tableS2_positive_variant_maf_distribution.tsv"

"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/make_tableS3_maf_stratified_summary.py" \
  --input "${DATA_DIR}/figureS1_maf_stratified_auroc/figureS1_maf_stratified_auroc_by_threshold.tsv" \
  --output "${TABLE_OUT}/tableS3_maf_stratified_auroc_summary.tsv"

"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/make_tableS4_filer_variant_sample_sizes.py" \
  --input "${DATA_DIR}/supporting_inputs/deduplicated_positive_variants_for_filer.tsv" \
  --output "${TABLE_OUT}/tableS4_filer_positive_variant_sample_sizes.tsv"

echo "Regenerated figures: ${FIGURE_OUT}"
echo "Regenerated tables: ${TABLE_OUT}"
