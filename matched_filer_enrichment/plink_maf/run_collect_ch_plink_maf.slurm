#!/bin/bash
#SBATCH --job-name=ch_mafcollect
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/maf_collect_%j.out
#SBATCH --error=logs/maf_collect_%j.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
python -u "${repo}/matched_filer_enrichment/plink_maf/collect_ch_plink_maf.py" \
  --work-dir "${project}/filer_plink_maf"
