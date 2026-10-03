#!/bin/bash
#SBATCH --job-name=ch_extract
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
release=${EQTL_RELEASE_DIR:-/path/to/eqtl_four_population_20260922}
common=${COMMON_DIR:-/path/to/common_eqtl_snp_proj}
base=${BASE_BENCHMARK_DIR:-/path/to/finemapped_eqtl_benchmark}
followup=${base}/model_first_eqtl_followup
export FINEMAPPING_PROFILE=${FINEMAPPING_PROFILE:?set to the regular-eQTL fine-mapping profile label of the CH release}

mkdir -p "${project}/logs" "${project}/outputs"

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-NA}"

python -u \
  "${repo}/benchmark_pipeline/extract_ch_inputs.py" \
  --release-root "${release}" \
  --output-dir "${project}/outputs" \
  --gene-locations "${common}/inputs/hg38_gene_locations.txt" \
  --gene-annotation "${followup}/manifests/gencode_v46_gene_annotation.tsv" \
  --general-genes "${followup}/manifests/general_benchmark_genes.tsv" \
  --analysis-pairs "${followup}/results/analysis_pairs.tsv.gz" \
  --combined-score-table "${common}/outputs_susie/analysis/susie_s2f_combined.tsv.gz" \
  --combined-score-table "${base}/results_auroc_expanded_100x_tablepos_full100/analysis/susie_s2f_combined.tsv.gz"

echo "finished_at=$(date --iso-8601=seconds)"
