#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
release=${EQTL_RELEASE_DIR:-/path/to/eqtl_four_population_20260922}
# AA and NHW base benchmark tables (model_scoring/ outputs)
inputs=${BASE_TABLES_DIR:-${project}/reference_inputs}
export FINEMAPPING_PROFILE=${FINEMAPPING_PROFILE:?set to the regular-eQTL fine-mapping profile label of the CH release}

python -u "${repo}/benchmark_pipeline/build_benchmark_results.py" \
  --master-scored "${project}/scoring/master_scored_pairs_wide.tsv.gz" \
  --ch-members "${release}/finemapping/summary/CH/credible_set_members.tsv.gz" \
  --base-nominal "${inputs}/nominal_eqtl_10k_predictions_wide.tsv.gz" \
  --base-high-pip "${inputs}/finemapped_pip_ge_0.9_predictions_wide.tsv.gz" \
  --base-auroc-pool "${inputs}/finemapped_auroc_selected_pairs_predictions_wide.tsv.gz" \
  --base-auroc-results "${inputs}/susie_auroc_bootstrap_results_manuscript_full100.tsv" \
  --base-maf-summary "${inputs}/susie_auroc_maf_stratified_results_full100.tsv" \
  --base-maf-bootstrap "${inputs}/figureS1_maf_stratified_auroc_bootstrap.tsv.gz" \
  --base-positive-maf-summary "${inputs}/positive_variant_maf_distribution.tsv" \
  --base-cs-results "${inputs}/credible_set_size_sensitivity_bootstrap.tsv" \
  --base-cs-sizes "${inputs}/positive_credible_set_size_summary.tsv" \
  --output-dir "${project}/benchmark_results"
