#!/bin/bash
#SBATCH --job-name=ch_mfld
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/model_first_ld_%A_%a.out
#SBATCH --error=logs/model_first_ld_%A_%a.err

set -euo pipefail

project=${PROJECT_DIR:-/path/to/project}
native_root=${project}/model_first_native_ld
# PLINK 1.9 and Python must be on PATH.
python -u "${native_root}/code/compute_chromosome_ld.py" \
  --job-manifest "${native_root}/manifests/ch_ld_jobs.tsv" \
  --job-index "${SLURM_ARRAY_TASK_ID}"
