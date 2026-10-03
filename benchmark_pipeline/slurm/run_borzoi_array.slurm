#!/bin/bash
#SBATCH --job-name=ch_borzoi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/borzoi_%A_%a.out
#SBATCH --error=logs/borzoi_%A_%a.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}

module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
# Borzoi environment (borzoi_py310) must be on PATH; BORZOI_HG38 points at the Borzoi hg38 resources.
export BORZOI_HG38=${BORZOI_HG38:-/path/to/borzoi/examples/hg38}
mkdir -p "${project}/logs" "${project}/scoring/new_scores"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import tensorflow as tf; devices=tf.config.list_physical_devices('GPU'); print('tensorflow_gpus='+str(devices)); assert devices, 'TensorFlow did not register the allocated GPU'"

python -u "${repo}/model_scoring/run_borzoi_chunk.py" \
  --jobs_tsv "${project}/scoring/manifests/master_scoring_jobs.tsv" \
  --results_dir "${project}/scoring/new_scores" \
  --task_id "${SLURM_ARRAY_TASK_ID}"
