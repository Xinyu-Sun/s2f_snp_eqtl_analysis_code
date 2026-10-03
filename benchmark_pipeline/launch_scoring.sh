#!/bin/bash
# Submit the scoring arrays for the scoring plan, then score collection, the CH model-first setup,
# and the benchmark build (each after collection).
set -euo pipefail

project=${PROJECT_DIR:-/path/to/project}
repo=${REPO_DIR:-/path/to/s2f_snp_eqtl_analysis_code}
manifest=${project}/scoring/manifests/master_scoring_jobs.tsv
submission=${project}/scoring/scoring_submission.tsv

if [ ! -s "${manifest}" ]; then
  echo "ERROR: missing scoring manifest: ${manifest}" >&2
  exit 1
fi

borzoi_n=$(awk -F '\t' 'NR>1 && $1=="borzoi" {n++} END {print n+0}' "${manifest}")
alphagenome_n=$(awk -F '\t' 'NR>1 && $1=="alphagenome" {n++} END {print n+0}' "${manifest}")

printf 'tool\tjob_id\tarray_tasks\n' > "${submission}.tmp"
dependencies=()
if [ "${borzoi_n}" -gt 0 ]; then
  job_id=$(sbatch --parsable --array="1-${borzoi_n}%80" "${repo}/benchmark_pipeline/slurm/run_borzoi_array.slurm")
  printf 'borzoi\t%s\t%s\n' "${job_id}" "${borzoi_n}" >> "${submission}.tmp"
  dependencies+=("${job_id}")
fi
if [ "${alphagenome_n}" -gt 0 ]; then
  job_id=$(sbatch --parsable --array="1-${alphagenome_n}%20" "${repo}/benchmark_pipeline/slurm/run_alphagenome_array.slurm")
  printf 'alphagenome\t%s\t%s\n' "${job_id}" "${alphagenome_n}" >> "${submission}.tmp"
  dependencies+=("${job_id}")
fi
if [ "${#dependencies[@]}" -gt 0 ]; then
  dependency=$(IFS=:; echo "${dependencies[*]}")
  collect_job=$(sbatch --parsable --dependency="afterok:${dependency}" "${repo}/benchmark_pipeline/slurm/run_collect_scores.slurm")
else
  collect_job=$(sbatch --parsable "${repo}/benchmark_pipeline/slurm/run_collect_scores.slurm")
fi
printf 'collector\t%s\t1\n' "${collect_job}" >> "${submission}.tmp"
model_first_job=$(sbatch --parsable --dependency="afterok:${collect_job}" "${repo}/model_first_native_ld/code/ch_inputs/launch_ch_model_first.sh")
printf 'model_first_setup\t%s\t1\n' "${model_first_job}" >> "${submission}.tmp"
benchmark_job=$(sbatch --parsable --dependency="afterok:${collect_job}" "${repo}/benchmark_pipeline/slurm/run_benchmark_results.slurm")
printf 'benchmark_results\t%s\t1\n' "${benchmark_job}" >> "${submission}.tmp"
mv "${submission}.tmp" "${submission}"
cat "${submission}"
