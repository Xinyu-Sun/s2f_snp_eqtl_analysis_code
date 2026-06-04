#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-12:30:00
#SBATCH --job-name=borzoi_auroc100
#SBATCH --output=logs/borzoi_%A_%a.out
#SBATCH --error=logs/borzoi_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/fine_mapped_eqtl_project}"
RESULTS_DIR="${RESULTS_DIR:-fine_mapped_eqtl_auroc_results}"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_DIR}/logs"

echo "started_at=$(date)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-NA}"

init_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return 0
  fi
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      return 0
    fi
  done
  echo "ERROR: conda not found" >&2
  return 1
}

init_conda
set +u
conda activate borzoi_py310
set -u
python -V

python -u model_scoring/run_borzoi_chunk.py \
  --jobs_tsv "${RESULTS_DIR}/manifests/jobs.tsv" \
  --results_dir "${RESULTS_DIR}" \
  --task_id "${SLURM_ARRAY_TASK_ID}"

echo "finished_at=$(date)"
