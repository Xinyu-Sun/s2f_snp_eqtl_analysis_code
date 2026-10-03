#!/bin/bash
#SBATCH --job-name=ch_plan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/plan_%j.out
#SBATCH --error=logs/plan_%j.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
common=${COMMON_DIR:-/path/to/common_eqtl_snp_proj}
base=${BASE_BENCHMARK_DIR:-/path/to/finemapped_eqtl_benchmark}
nominal=${NOMINAL_BENCHMARK_DIR:-/path/to/nominal_eqtl_10k_benchmark}
followup=${base}/model_first_eqtl_followup
export FINEMAPPING_PROFILE=${FINEMAPPING_PROFILE:?set to the regular-eQTL fine-mapping profile label of the CH release}

mkdir -p "${project}/logs" "${project}/scoring"

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-NA}"

python -u \
  "${repo}/benchmark_pipeline/build_scoring_plan.py" \
  --extraction-dir "${project}/outputs" \
  --output-dir "${project}/scoring" \
  --analysis-pairs "${followup}/results/analysis_pairs.tsv.gz" \
  --combined-score-table "${common}/outputs_susie/analysis/susie_s2f_combined.tsv.gz" \
  --combined-score-table "${base}/results_auroc_expanded_100x_tablepos_full100/analysis/susie_s2f_combined.tsv.gz" \
  --nominal-long "${nominal}/outputs/analysis/nominal_10k_predictions_long.tsv.gz" \
  --pairs-per-chunk 500

echo "finished_at=$(date --iso-8601=seconds)"
