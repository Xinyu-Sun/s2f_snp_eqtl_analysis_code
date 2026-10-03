#!/bin/bash
#SBATCH --job-name=ch_mfsetup
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/model_first_setup_%j.out
#SBATCH --error=logs/model_first_setup_%j.err

set -euo pipefail

project=${PROJECT_DIR:-/path/to/project}
native_root=${project}/model_first_native_ld
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
# Locked model-first design (500-gene panel, model scores, AA/NHW manifests and LD) and its pair table
followup=${MODEL_FIRST_DESIGN_DIR:-/path/to/model_first_eqtl_followup}
locked_native=${followup}/native_ld_followup

if [ ! -d "${native_root}" ]; then
  # Work on a copy of the locked design directory: downstream scripts write manifests, results,
  # QC tables, and figures in place. Reflink when supported; otherwise make a full copy.
  cp -a --reflink=auto "${locked_native}" "${native_root}"
fi

python -u "${repo}/model_first_native_ld/code/ch_inputs/build_ch_pair_manifests.py" \
  --master-scored "${project}/scoring/master_scored_pairs_wide.tsv.gz" \
  --locked-analysis-pairs "${followup}/results/analysis_pairs.tsv.gz" \
  --native-root "${native_root}" \
  --allow-ld-recompute

reuse_ld=$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["locked_ld_reuse_permitted"]).lower())' \
  "${native_root}/qc/ch_manifest_audit.json")
if [ "${reuse_ld}" = "true" ]; then
  ld_job=""
  clump_job=$(sbatch --parsable --array=1-22%22 "${repo}/model_first_native_ld/code/ch_inputs/run_ch_clump_array.slurm")
else
  python -u "${repo}/model_first_native_ld/code/ch_inputs/prepare_ch_ld_inputs.py" \
    --native-root "${native_root}"
  ld_job=$(sbatch --parsable --array=1-22%22 "${repo}/model_first_native_ld/code/ch_inputs/run_ch_ld_array.slurm")
  clump_job=$(sbatch --parsable --dependency="afterok:${ld_job}" --array=1-22%22 \
    "${repo}/model_first_native_ld/code/ch_inputs/run_ch_clump_array.slurm")
fi
post_job=$(sbatch --parsable --dependency="afterok:${clump_job}" "${repo}/model_first_native_ld/code/ch_inputs/run_postprocess.slurm")
printf 'stage\tjob_id\n' > "${project}/model_first_submission.tsv"
if [ -n "${ld_job}" ]; then
  printf 'ld\t%s\n' "${ld_job}" >> "${project}/model_first_submission.tsv"
fi
printf 'clump\t%s\npostprocess\t%s\n' "${clump_job}" "${post_job}" \
  >> "${project}/model_first_submission.tsv"
cat "${project}/model_first_submission.tsv"
