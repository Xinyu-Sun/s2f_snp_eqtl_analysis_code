#!/usr/bin/env bash
# Regenerate the manuscript figures and every supplementary-table source from the data package.
#
#   DATA_DIR   data package (default: ../data_availability next to this repository)
#   OUT_DIR    where regenerated outputs are written (default: ./regenerated_outputs)
#   PYTHON     Python interpreter (default: python3)
#
# Inputs are read only from DATA_DIR; nothing in DATA_DIR is modified. At the end, each of the 25 supplementary-table
# sources (Tables S1-S25) and the five FILER composition files are compared with the files in DATA_DIR; a missing or
# different file makes the script exit with status 1.
#
# How each table source is produced here:
#   S1, S8-S10           from benchmark_results/ (build_benchmark_results.py outputs)
#   S18                  from the matched FILER enrichment results
#   S19, S20             from the FILER composition recomputed from matched_filer_enrichment/ inputs and annotation calls
#   S23, S24             from model_first_native_ld/results/
#   S4-S7, S11-S17, S21, S22
#                        from the stored follow-up results in followup_analyses/results/ (the analyses themselves are
#                        in followup_analyses/; run_followup_analyses.sbatch runs them)
#   S25                  recomputed from model_first_native_ld/results/native_ld_yields.tsv
#   S2, S3               study-design counts and the eQTLGen replication summary; they need the individual-level
#                        association results and eQTLGen, so they are written from the values stored in the package
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DATA_DIR="$(cd -- "${DATA_DIR:-${SCRIPT_DIR}/../data_availability}" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/regenerated_outputs}"
PYTHON="${PYTHON:-python3}"
export MPLBACKEND=Agg
export FU_GENE_TSS="${DATA_DIR}/matched_filer_enrichment/inputs/hg38_gene_locations.txt"
export FU_MAF="${DATA_DIR}/matched_filer_enrichment/inputs/benchmark_plink_maf_exact_qtl_cohorts.tsv.gz"
PAIRS="${DATA_DIR}/detailed_predictions/finemapped_auroc_selected_pairs_predictions_wide.tsv.gz"

mkdir -p "${OUT_DIR}"/{figure1,figure2,figure3,figure4,figure5,figureS1,figureS2,figureS3,table_sources,table_tex,filer_composition}

echo "== Figure 1"
S2F_RESULTS="${DATA_DIR}/figure1_corr_concordance" S2F_FIGURE_OUTPUT="${OUT_DIR}/figure1" \
  "${PYTHON}" "${SCRIPT_DIR}/figure1_corr_concordance/make_figure1_corr_concordance.py"

echo "== Figure 2"
S2F_RESULTS="${DATA_DIR}/figure2_distance_matched_auroc" S2F_FIGURE_OUTPUT="${OUT_DIR}/figure2" \
  "${PYTHON}" "${SCRIPT_DIR}/figure2_distance_matched_auroc/make_figure2_distance_matched_auroc.py"

echo "== Figure 3"
(
  cd "${SCRIPT_DIR}/followup_analyses"
  "${PYTHON}" a8_make_anatomy_figure.py "${PAIRS}" "${DATA_DIR}/figure3_score_anatomy" \
    "${OUT_DIR}/figure3/fig_aa_advantage_anatomy.pdf"
)

echo "== Figure 4"
(
  cd "${SCRIPT_DIR}/followup_analyses"
  "${PYTHON}" a10_make_accessible_figure.py "${PAIRS}" \
    "${DATA_DIR}/matched_filer_enrichment/variant_filer_category_counts.tsv.gz" \
    "${DATA_DIR}/figure4_accessible_chromatin" "${OUT_DIR}/figure4/fig_accessible_chromatin.pdf"
)

echo "== Figure 5"
"${PYTHON}" "${SCRIPT_DIR}/model_first_native_ld/code/make_native_ld_figure.py" \
  --yields "${DATA_DIR}/figure5_model_first/native_ld_yields.tsv" \
  --distributions "${DATA_DIR}/figure5_model_first/native_ld_top_vs_rest.tsv" \
  --output "${OUT_DIR}/figure5/native_ld_primary_summary.png" \
  --summary "${OUT_DIR}/figure5/native_ld_figure_summary.json"

echo "== Figure S1"
S2F_RESULTS="${DATA_DIR}/figureS1_model_convergence" S2F_FIGURE_OUTPUT="${OUT_DIR}/figureS1" \
  "${PYTHON}" "${SCRIPT_DIR}/figureS1_model_convergence/make_figureS1_model_convergence.py"

echo "== Figure S2"
S2F_RESULTS="${DATA_DIR}/figureS2_maf_stratified_auroc" S2F_FIGURE_OUTPUT="${OUT_DIR}/figureS2" \
  "${PYTHON}" "${SCRIPT_DIR}/figureS2_maf_stratified_auroc/make_figureS2_maf_stratified_auroc.py"

echo "== Figure S3"
"${PYTHON}" "${SCRIPT_DIR}/model_first_native_ld/code/make_native_ld_distribution_figure.py" \
  --ecdf "${DATA_DIR}/figureS3_model_first_distributions/native_ld_primary_ecdf.tsv.gz" \
  --output "${OUT_DIR}/figureS3/native_ld_top_vs_rest_distributions.png"

echo "== FILER composition (sources of Tables S19 and S20)"
"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/build_filer_composition_tables.py" \
  --combined "${DATA_DIR}/matched_filer_enrichment/inputs/susie_s2f_combined.tsv.gz" \
  --annotations "${DATA_DIR}/matched_filer_enrichment/variant_filer_category_counts.tsv.gz" \
  --output-dir "${OUT_DIR}/filer_composition"

echo "== Tables S1, S8-S10, S18-S20, S23, S24"
"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/render_supplementary_tables.py" \
  --benchmark-dir "${DATA_DIR}/benchmark_results" \
  --composition-dir "${OUT_DIR}/filer_composition" \
  --model-first-results "${DATA_DIR}/model_first_native_ld/results" \
  --filer-root "${DATA_DIR}/matched_filer_enrichment" \
  --tex-dir "${OUT_DIR}/table_tex" \
  --data-dir "${OUT_DIR}/table_sources"

echo "== Table S25 (lift contrasts recomputed from the model-first yields)"
mkdir -p "${OUT_DIR}/followup_results"
"${PYTHON}" "${SCRIPT_DIR}/followup_analyses/b5_model_first_lift_contrasts.py" \
  "${DATA_DIR}/model_first_native_ld/results/native_ld_yields.tsv" \
  "${OUT_DIR}/followup_results/b5_model_first_lift_contrasts.tsv"

echo "== Tables S2-S7, S11-S17, S21, S22, S25 (sources) and the follow-up LaTeX bodies"
for f in "${DATA_DIR}"/followup_analyses/results/*.tsv; do
  name="$(basename "${f}")"
  [ "${name}" = b5_model_first_lift_contrasts.tsv ] || cp "${f}" "${OUT_DIR}/followup_results/${name}"
done
"${PYTHON}" "${SCRIPT_DIR}/supplementary_tables/write_followup_table_sources.py" \
  --results "${OUT_DIR}/followup_results" \
  --eqtlgen "${DATA_DIR}/followup_analyses/external_replication/eqtlgen_replication.tsv" \
  --output-dir "${OUT_DIR}/table_sources"
(
  cd "${SCRIPT_DIR}/followup_analyses"
  "${PYTHON}" make_followup_tables.py "${OUT_DIR}/followup_results" "${OUT_DIR}/table_tex"
)

echo "== Comparison with the data package"
status=0
expected=(
  tableS01_auroc_positive_counts.tsv tableS02_group_design.tsv tableS03_eqtlgen_replication.tsv
  tableS04_joint_bootstrap.tsv tableS05_logistic_slopes.tsv tableS06_subsets_pip_weighting.tsv
  tableS07_score_anatomy.tsv tableS08_cs_size_sensitivity.tsv tableS09_maf_positive_summary.tsv
  tableS10_maf_auroc_summary.tsv tableS11_shared_genes.tsv tableS12_within_gene.tsv
  tableS13_matched_positives.tsv tableS14_decile_enrichment.tsv tableS15_ld_proxies.tsv
  tableS16_freq_class.tsv tableS17_label_transfer.tsv tableS18_filer_matched_accessible.tsv
  tableS19_functional_annotation_sample_sizes.tsv tableS20_functional_annotation_overlap.tsv
  tableS21_accessible.tsv tableS22_annotation_baselines.tsv tableS23_native_ld_adjusted.tsv
  tableS24_native_ld_regional_summaries.tsv tableS25_model_first_lift.tsv
)
for name in "${expected[@]}"; do
  if [ ! -f "${OUT_DIR}/table_sources/${name}" ]; then
    echo "MISSING    table_sources/${name}"; status=1
  elif cmp -s "${OUT_DIR}/table_sources/${name}" "${DATA_DIR}/supplementary_tables/${name}"; then
    echo "identical  supplementary_tables/${name}"
  else
    echo "DIFFERENT  supplementary_tables/${name}"; status=1
  fi
done
if cmp -s "${OUT_DIR}/followup_results/b5_model_first_lift_contrasts.tsv" "${DATA_DIR}/followup_analyses/results/b5_model_first_lift_contrasts.tsv"; then
  echo "identical  followup_analyses/results/b5_model_first_lift_contrasts.tsv"
else
  echo "DIFFERENT  followup_analyses/results/b5_model_first_lift_contrasts.tsv"; status=1
fi
for name in filer_positive_variant_sample_sizes.tsv filer_annotation_overlap_key.tsv filer_pairwise_comparisons_key.tsv \
            filer_positive_composition_all_categories.tsv filer_positive_pairwise_all_categories.tsv; do
  if cmp -s "${OUT_DIR}/filer_composition/${name}" "${DATA_DIR}/matched_filer_enrichment/composition/${name}"; then
    echo "identical  matched_filer_enrichment/composition/${name}"
  else
    echo "DIFFERENT  matched_filer_enrichment/composition/${name}"; status=1
  fi
done

echo "Regenerated figures and tables: ${OUT_DIR}"
exit "${status}"
