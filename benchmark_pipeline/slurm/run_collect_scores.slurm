#!/bin/bash
#SBATCH --job-name=ch_collect
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}

python -u "${repo}/benchmark_pipeline/collect_scores.py" \
  --scoring-dir "${project}/scoring" \
  --allow-incomplete
