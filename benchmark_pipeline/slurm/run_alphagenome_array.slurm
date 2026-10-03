#!/bin/bash
#SBATCH --job-name=ch_alphagenome
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/alphagenome_%A_%a.out
#SBATCH --error=logs/alphagenome_%A_%a.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}

# ALPHAGENOME_API must hold the AlphaGenome API key; the alphagenome environment must be on PATH.
if [ -z "${ALPHAGENOME_API:-}" ]; then
  echo "ERROR: ALPHAGENOME_API is unavailable" >&2
  exit 1
fi
mkdir -p "${project}/logs" "${project}/scoring/new_scores"

python -u "${repo}/model_scoring/run_alphagenome_chunk.py" \
  --jobs_tsv "${project}/scoring/manifests/master_scoring_jobs.tsv" \
  --results_dir "${project}/scoring/new_scores" \
  --task_id "${SLURM_ARRAY_TASK_ID}"
