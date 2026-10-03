#!/bin/bash
#SBATCH --job-name=ch_mafsetup
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/maf_setup_%j.out
#SBATCH --error=logs/maf_setup_%j.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
work=${project}/filer_plink_maf
mkdir -p "${work}" "${project}/logs"
python -u "${repo}/matched_filer_enrichment/plink_maf/prepare_ch_plink_maf.py" \
  --selected-pairs "${project}/outputs/ch_finemap_selected_pairs.tsv.gz" \
  --output-dir "${work}"
maf_job=$(sbatch --parsable "${repo}/matched_filer_enrichment/plink_maf/run_ch_plink_maf.slurm")
collect_job=$(sbatch --parsable --dependency="afterok:${maf_job}" "${repo}/matched_filer_enrichment/plink_maf/run_collect_ch_plink_maf.slurm")
printf 'stage\tjob_id\nmaf\t%s\nmaf_collect\t%s\n' "${maf_job}" "${collect_job}" \
  > "${project}/filer_plink_maf_submission.tsv"
cat "${project}/filer_plink_maf_submission.tsv"
