#!/bin/bash
#SBATCH --job-name=ch_maf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/maf_%j.out
#SBATCH --error=logs/maf_%j.err

set -euo pipefail

# Site-specific locations (override in the environment).
project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
work=${project}/filer_plink_maf
prefix=/mnt/vstor/Data14/metabrain_lasso/rna_seq_norm/MAGENTA_Hispanic/geno/imp_fil/geno_final/ci/MAGENTA_HISP_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.IHG.plink_qc
keep=/mnt/vstor/Data14/metabrain_lasso/filer_maf_current_manuscript_20260831/keep/CH.exact_qtl_cohort.keep
plink=${PLINK:-plink}

mkdir -p "${work}/plink"
"${plink}" \
  --bfile "${prefix}" \
  --keep "${keep}" \
  --extract "${work}/CH.all.extract_ids.txt" \
  --keep-allele-order \
  --freq \
  --out "${work}/plink/CH.exact_qtl_cohort"
