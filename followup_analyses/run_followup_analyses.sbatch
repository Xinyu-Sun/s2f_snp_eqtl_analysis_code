#!/bin/bash
#SBATCH --job-name=followup
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/followup_%j.out
#SBATCH --error=logs/followup_%j.err
# Invocation order of the follow-up analyses behind Figures 3 and 4 and Tables S4-S7, S11-S17, S21, S22 and S25
# (Table S3: external_replication/r1_eqtlgen_replication.py, which needs the per-group nominal tables and eQTLGen).
# Run a56_make_variant_lists.py and a56_plink_ld_freq.sbatch first (LD and allele-frequency outputs in $F/ld, $F/freq).
# Inputs from the data package:
#   PAIRS   detailed_predictions/finemapped_auroc_selected_pairs_predictions_wide.tsv.gz
#   FILER   matched_filer_enrichment/variant_filer_category_counts.tsv.gz
#   FU_GENE_TSS, FU_MAF   matched_filer_enrichment/inputs/hg38_gene_locations.txt, benchmark_plink_maf_exact_qtl_cohorts.tsv.gz
#   YIELDS  model_first_native_ld/results/native_ld_yields.tsv
set -uo pipefail
F=${FOLLOWUP_DIR:-/path/to/followup_project}
DATA=${DATA_DIR:-/path/to/data_availability}
CODE=${CODE_DIR:-$(pwd)}
PY=${PYTHON:-python3}
P=${PAIRS:-$DATA/detailed_predictions/finemapped_auroc_selected_pairs_predictions_wide.tsv.gz}
FILER=${FILER:-$DATA/matched_filer_enrichment/variant_filer_category_counts.tsv.gz}
YIELDS=${YIELDS:-$DATA/model_first_native_ld/results/native_ld_yields.tsv}
O=${OUT:-$F/results}
mkdir -p $O
export FU_GENE_TSS=${FU_GENE_TSS:-$DATA/matched_filer_enrichment/inputs/hg38_gene_locations.txt}
export FU_MAF=${FU_MAF:-$DATA/matched_filer_enrichment/inputs/benchmark_plink_maf_exact_qtl_cohorts.tsv.gz}
cd $CODE
run() { echo "== $* $(date +%T)"; "$@" || echo "FAILED: $*"; }
run $PY b3_primary_test.py $P $O
run $PY b4_baselines.py $P $FILER $O
run $PY b2_within_gene.py $P $O
run $PY b7_subsets_pipweighted.py $P $O
run $PY a1_score_anatomy.py $P $O
run $PY a3_decile_enrichment.py $P $O
run $PY a5_summarize_ld.py $P $F/ld $O $F/freq
run $PY a6_summarize_freq.py $P $F/freq $O
run $PY a7_shared_gene_control.py $P $O
run $PY a9_accessible_chromatin.py $P $FILER $O/a5_positive_ld_partners.tsv $O
run $PY a4_matched_positives.py $P $FU_MAF $O
run $PY a4b_matching_covariate_sets.py $P $O
run $PY a2_joint_bootstrap_auroc.py $P $O
# B1 needs the AA/NHW per-gene SuSiE outputs, the CH compact PIP table and fine-mapped gene list, and FU_NOMINAL_DIR
run $PY b1_label_transfer.py $P ${CH_COMPACT:-$F/inputs/ch_pip_ge_0.001_or_cs.tsv.gz} ${CH_GENES:-$F/inputs/ch_finemapped_genes.tsv} $O
run $PY b8_shared_matched.py $P $O/b1_label_transfer_detail.tsv $FILER $O
run $PY b5_model_first_lift_contrasts.py $YIELDS $O/b5_model_first_lift_contrasts.tsv
run $PY a8_make_anatomy_figure.py $P $O $O/fig_aa_advantage_anatomy.pdf
run $PY a10_make_accessible_figure.py $P $FILER $O $O/fig_accessible_chromatin.pdf
# LaTeX bodies of the follow-up supplementary tables (captions of existing table files are kept); Table S15 joins the
# LD-proxy table and the LD-adjusted regressions as parts A and B, and Table S17 joins the label-transfer table and the
# matched shared-versus-group-specific comparison
T=${TABLES_DIR:-$F/tables}
run $PY make_followup_tables.py $O $T
run $PY combine_tables.py --out $T/aa_advantage_ld_combined.tex --label stab:ld_proxies \
  --caption "LD structure of high-PIP variants and its contribution to the group difference in score magnitude, related to Figure~3." \
  --part $T/aa_advantage_ld_proxies.tex --part $T/aa_advantage_ld_proxy_regression.tex
run $PY combine_tables.py --out $T/review_label_transfer_combined.tex --label stab:label_transfer \
  --caption "Status of each group's high-PIP variants in the fine-mapping of the other two groups, and separation of shared and group-specific high-PIP variants, related to Figure~3." \
  --part "$T/review_label_transfer.tex::Status of each group's high-PIP variants in the fine-mapping of the other two groups::High-PIP variants are PIP \$\\geq 0.9\$ and in a credible set; the lower block gives their distance-matched AUROC by sharing (positives resampled, 1,000 iterations). \`\`Variant not tested'' means the gene has a credible set in the other group but the variant did not pass its allele-count filter." \
  --part $T/review_shared_matched.tex
echo "ALL DONE $(date)"
