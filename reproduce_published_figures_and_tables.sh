#!/usr/bin/env bash
# Regenerate the manuscript figures and supplementary-table sources from the data package.
#
#   DATA_DIR   data package (default: ../data_availability next to this repository)
#   OUT_DIR    where regenerated outputs are written (default: ./regenerated_outputs)
#   PYTHON     Python interpreter (default: python3)
#
# Inputs are read only from DATA_DIR; nothing in DATA_DIR is modified. At the end, regenerated
# table sources are compared with the files in DATA_DIR/supplementary_tables.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DATA_DIR="$(cd -- "${DATA_DIR:-${SCRIPT_DIR}/../data_availability}" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/regenerated_outputs}"
PYTHON="${PYTHON:-python3}"
export MPLBACKEND=Agg

mkdir -p "${OUT_DIR}"/{figure1,figure2,figure3,figure4,figure5,figureS1,figureS2,table_sources,table_tex,filer_composition}

echo "== Figure 1"
S2F_RESULTS="${DATA_DIR}/figure1_corr_concordance" S2F_FIGURE_OUTPUT="${OUT_DIR}/figure1" \
  "${PYTHON}" "${SCRIPT_DIR}/figure1_corr_concordance/make_figure1_corr_concordance.py"

echo "== Figure 2"
S2F_RESULTS="${DATA_DIR}/figure2_model_convergence" S2F_FIGURE_OUTPUT="${OUT_DIR}/figure2" \
  "${PYTHON}" "${SCRIPT_DIR}/figure2_model_convergence/make_figure2_model_convergence.py"

echo "== Figure 3"
S2F_RESULTS="${DATA_DIR}/figure3_distance_matched_auroc" S2F_FIGURE_OUTPUT="${OUT_DIR}/figure3" \
  "${PYTHON}" "${SCRIPT_DIR}/figure3_distance_matched_auroc/make_figure3_distance_matched_auroc.py"

echo "== Figure 4"
(
  cd "${SCRIPT_DIR}/followup_analyses"
  FU_GENE_TSS="${DATA_DIR}/matched_filer_enrichment/inputs/hg38_gene_locations.txt" \
  FU_MAF="${DATA_DIR}/matched_filer_enrichment/inputs/benchmark_plink_maf_exact_qtl_cohorts.tsv.gz" \
    "${PYTHON}" a8_make_anatomy_figure.py \
      "${DATA_DIR}/detailed_predictions/finemapped_auroc_selected_pairs_predictions_wide.tsv.gz" \
      "${DATA_DIR}/figure4_score_anatomy" \
      "${OUT_DIR}/figure4/fig_aa_advantage_anatomy.pdf"
)

echo "== Figure 5"
"${PYTHON}" "${SCRIPT_DIR}/model_first_native_ld/code/make_native_ld_figure.py" \
  --yields "${DATA_DIR}/figure5_model_first/native_ld_yields.tsv" \
  --distributions "${DATA_DIR}/figure5_model_first/native_ld_top_vs_rest.tsv" \
  --output "${OUT_DIR}/figure5/native_ld_primary_summary.png" \
  --summary "${OUT_DIR}/figure5/native_ld_figure_summary.json"

echo "== Figure S1"
S2F_RESULTS="${DATA_DIR}/figureS1_maf_stratified_auroc" S2F_FIGURE_OUTPUT="${OUT_DIR}/figureS1" \
  "${PYTHON}" "${SCRIPT_DIR}/figureS1_maf_stratified_auroc/make_figureS1_maf_stratified_auroc.py"

echo "== Figure S2"
"${PYTHON}" "${SCRIPT_DIR}/model_first_native_ld/code/make_native_ld_distribution_figure.py" \
  --ecdf "${DATA_DIR}/figureS2_model_first_distributions/native_ld_primary_ecdf.tsv.gz" \
  --output "${OUT_DIR}/figureS2/native_ld_top_vs_rest_distributions.png"

echo "== FILER composition (sources of Tables S20-S22)"
"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/build_filer_composition_tables.py" \
  --combined "${DATA_DIR}/matched_filer_enrichment/inputs/susie_s2f_combined.tsv.gz" \
  --annotations "${DATA_DIR}/matched_filer_enrichment/variant_filer_category_counts.tsv.gz" \
  --output-dir "${OUT_DIR}/filer_composition"

echo "== Tables S1, S8-S10, S19-S22, S25, S26"
"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/render_supplementary_tables.py" \
  --benchmark-dir "${DATA_DIR}/benchmark_results" \
  --composition-dir "${OUT_DIR}/filer_composition" \
  --model-first-results "${DATA_DIR}/model_first_native_ld/results" \
  --filer-root "${DATA_DIR}/matched_filer_enrichment" \
  --tex-dir "${OUT_DIR}/table_tex" \
  --data-dir "${OUT_DIR}/table_sources"

echo "== Tables S4-S7, S11-S18, S23, S24 (LaTeX bodies from the follow-up results)"
(
  cd "${SCRIPT_DIR}/followup_analyses"
  "${PYTHON}" make_followup_tables.py "${DATA_DIR}/followup_analyses/results" "${OUT_DIR}/table_tex"
)

echo "== Table S27"
"${PYTHON}" "${SCRIPT_DIR}/followup_analyses/b5_model_first_lift_contrasts.py" \
  "${DATA_DIR}/model_first_native_ld/results/native_ld_yields.tsv" \
  "${OUT_DIR}/table_sources/tableS27_model_first_lift.tsv"

echo "== Comparison with the data package"
status=0
for f in "${OUT_DIR}"/table_sources/*.tsv; do
  name="$(basename "${f}")"
  if cmp -s "${f}" "${DATA_DIR}/supplementary_tables/${name}"; then
    echo "identical  supplementary_tables/${name}"
  else
    echo "DIFFERENT  supplementary_tables/${name}"
    status=1
  fi
done
for name in filer_positive_variant_sample_sizes.tsv filer_annotation_overlap_key.tsv filer_pairwise_comparisons_key.tsv \
            filer_positive_composition_all_categories.tsv filer_positive_pairwise_all_categories.tsv; do
  if cmp -s "${OUT_DIR}/filer_composition/${name}" "${DATA_DIR}/matched_filer_enrichment/composition/${name}"; then
    echo "identical  matched_filer_enrichment/composition/${name}"
  else
    echo "DIFFERENT  matched_filer_enrichment/composition/${name}"
    status=1
  fi
done

echo "Regenerated figures and tables: ${OUT_DIR}"
exit "${status}"
