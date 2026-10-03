#!/bin/bash
#SBATCH --job-name=followup_plink
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/plink_%j.out
#SBATCH --error=logs/plink_%j.err
# PLINK 1.9 LD (A5) and allele frequencies (A6) in each group's exact QTL cohort. Needs individual-level genotypes,
# which are not distributed. Variant lists come from a56_make_variant_lists.py.
# Optional overrides: LISTS (directory with ld_lists/ and freq_lists/, default inputs), GRPS (default "AA CH NHW"),
# STEPS (default "freq ld").
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load PLINK/1.9b_6.21-x86_64
F=${FOLLOWUP_DIR:-/path/to/followup_project}
G=/mnt/vstor/Data14/metabrain_lasso/rna_seq_norm
K=${KEEP_DIR:-/path/to/sample_keep}
L=${LISTS:-inputs}
# each group's eQTL genotype file and eQTL participants (AA 224, CH 209, NHW 235)
declare -A PFX=( [AA]="$G/aa_233/geno/imp_fil/geno_final/plink_by_chr/MAGENTA_AA_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.no_outlier_MAC10.plink_qc"
                 [CH]="$G/MAGENTA_Hispanic/geno/imp_fil/geno_final/ihg_ad/plink_by_chr/MAGENTA_HISP_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.CH_MAC10.plink_qc"
                 [NHW]="$G/nhw_239/geno/imp_fil/geno_final/plink_by_chr/MAGENTA_NHW_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.no_outlier_MAC10.plink_qc" )
declare -A KEEP=( [AA]="$K/AA.samples.txt" [CH]="$K/CH.samples.txt" [NHW]="$K/NHW.samples.txt" )
cd $F
for g in ${GRPS:-AA CH NHW}; do
  for c in $(seq 1 22); do
    bf="${PFX[$g]}.$c"
    [ -f "$bf.bed" ] || { echo "missing $bf"; continue; }
    # A6: frequencies of every group's positives in this cohort
    if [[ " ${STEPS:-freq ld} " == *" freq "* ]] && [ -s $L/freq_lists/all.chr$c.txt ]; then
      plink --bfile "$bf" --keep ${KEEP[$g]} --extract $L/freq_lists/all.chr$c.txt --keep-allele-order --freq --out freq/$g.chr$c --threads 4 --silent || true
    fi
    # A5: LD of this group's own positives with all variants within 1 Mb, r2 >= 0.1
    if [[ " ${STEPS:-freq ld} " == *" ld "* ]] && [ -s $L/ld_lists/$g.chr$c.txt ]; then
      plink --bfile "$bf" --keep ${KEEP[$g]} --ld-snp-list $L/ld_lists/$g.chr$c.txt --r2 --ld-window-kb 1000 --ld-window 999999 --ld-window-r2 0.1 --keep-allele-order --out ld/$g.chr$c --threads 4 --silent || true
    fi
  done
done
echo "PLINK done $(date)"
