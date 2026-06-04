#!/bin/bash
#SBATCH --job-name=ag_nom10k
#SBATCH --output=logs/alphagenome_%A_%a.out
#SBATCH --error=logs/alphagenome_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-12:30:00
#SBATCH --mem=32G
#SBATCH --partition=batch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/companion_eqtl_benchmark}"
TOPUP_ROOT="${TOPUP_ROOT:-${PROJECT_ROOT}/nominal_eqtl_support}"

echo "============================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "ArrayTaskID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "ALPHAGENOME_API set?: $([ -n \"${ALPHAGENOME_API:-}\" ] && echo yes || echo no)"
echo "Top-up root: ${TOPUP_ROOT}"
echo "============================================"

cd "${TOPUP_ROOT}"
mkdir -p outputs/logs

init_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return 0
  fi
  for base in \
    "/usr/local/easybuild_allnodes/software/Miniconda3/23.10.0-1" \
    "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "$base/etc/profile.d/conda.sh"
      return 0
    fi
  done
  echo "ERROR: conda not found on PATH and no conda.sh under common locations." >&2
  return 1
}

echo "conda_on_path: $(command -v conda || echo NO)"
init_conda
echo "conda_base: $(conda info --base 2>/dev/null || echo UNKNOWN)"
conda activate alphagenome-env
echo "python: $(command -v python)"
python -V

python -u "${PROJECT_ROOT}/scripts/run_alphagenome_chunk.py" \
  --jobs_tsv outputs/manifests/jobs.tsv \
  --task_id "${SLURM_ARRAY_TASK_ID}"
