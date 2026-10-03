#!/bin/bash
#SBATCH --job-name=mfpost
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/model_first_post_%j.out
#SBATCH --error=logs/model_first_post_%j.err

set -euo pipefail

project=${PROJECT_DIR:-/path/to/project}
native_root=${project}/model_first_native_ld
followup=${MODEL_FIRST_DESIGN_DIR:-/path/to/model_first_eqtl_followup}

python -u "${native_root}/code/merge_native_ld_results.py" \
  --analysis-pairs "${followup}/results/analysis_pairs.tsv.gz"
python -u "${native_root}/code/analyze_native_ld_results.py" \
  --general-genes "${followup}/manifests/general_benchmark_analysis_weights.tsv" \
  --ad-genes "${followup}/manifests/ad_application_genes.tsv" \
  --replicates 1000 \
  --seed 20260820
python -u "${native_root}/code/fit_native_ld_adjusted.py"
python -u "${native_root}/code/summarize_native_ld_architecture.py"
python -u "${native_root}/code/make_native_ld_figure.py"
python -u "${native_root}/code/make_native_ld_distribution_figure.py"
