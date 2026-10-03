#!/bin/bash
#SBATCH --job-name=ch_mfclump
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/model_first_clump_%A_%a.out
#SBATCH --error=logs/model_first_clump_%A_%a.err

set -euo pipefail

project=${PROJECT_DIR:-/path/to/project}
native_root=${project}/model_first_native_ld
python -u "${native_root}/code/compute_chromosome_clumps.py" \
  --ancestry CH \
  --chromosome "${SLURM_ARRAY_TASK_ID}"
