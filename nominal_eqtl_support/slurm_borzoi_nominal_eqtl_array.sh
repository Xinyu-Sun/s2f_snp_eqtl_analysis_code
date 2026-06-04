#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-12:30:00
#SBATCH --job-name=borzoi_nom10k
#SBATCH --output=logs/borzoi_%A_%a.out
#SBATCH --error=logs/borzoi_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/companion_eqtl_benchmark}"
TOPUP_ROOT="${TOPUP_ROOT:-${PROJECT_ROOT}/nominal_eqtl_support}"

echo "============================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "ArrayTaskID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Top-up root: ${TOPUP_ROOT}"
echo "============================================"

cd "${TOPUP_ROOT}"
mkdir -p outputs/logs

init_lmod_cuda() {
  if [ -f /etc/profile.d/z00_lmod.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/z00_lmod.sh
  fi
  if ! command -v ml >/dev/null 2>&1; then
    echo "WARN: lmod 'ml' command not found; proceeding without module-loaded CUDA/cuDNN" >&2
    return 0
  fi
  ml purge >/dev/null 2>&1 || true
  ml CUDA/12.2.0 cuDNN/8.9.2.26-CUDA-12.2.0 >/dev/null 2>&1 || {
    echo "ERROR: failed to load CUDA/12.2.0 and cuDNN/8.9.2.26-CUDA-12.2.0 via Lmod" >&2
    return 1
  }
  return 0
}

init_lmod_cuda
export TF_FORCE_GPU_ALLOW_GROWTH=true
if [ -n "${CUDA_HOME:-}" ]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}${XLA_FLAGS:+ ${XLA_FLAGS}}"
fi

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

export PYTHONPATH="${PYTHONPATH:-}"
set +u
conda activate borzoi_py310
set -u

echo "python: $(command -v python)"
python -V
echo "CUDA_HOME: ${CUDA_HOME:-UNSET}"
echo "cuDNN root: ${EBROOTCUDNN:-UNSET}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-UNSET}"
if command -v ml >/dev/null 2>&1; then
  ml list 2>&1 || true
fi
nvidia-smi || true

python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"tensorflow_gpus: {gpus}")
if not gpus:
    raise SystemExit("ERROR: TensorFlow did not detect a GPU; refusing Borzoi CPU fallback.")
PY

python "${PROJECT_ROOT}/scripts/run_borzoi_chunk.py" \
  --jobs_tsv outputs/manifests/jobs.tsv \
  --task_id "${SLURM_ARRAY_TASK_ID}"
