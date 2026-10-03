#!/bin/bash
# Run order of the matched FILER enrichment. FILER_PROJECT_DIR must contain scripts/ (this folder) and
# inputs/ (from the data package or build_filer_inputs.py).
set -euo pipefail
cd "${FILER_PROJECT_DIR:?set to the directory that holds scripts/ and inputs/}"
echo "start $(date)"
python3 scripts/01_prepare_and_match.py
echo "matched $(date)"
python3 scripts/02_query_filer_tabix.py --jobs 4
echo "annotated $(date)"
python3 scripts/03_run_enrichment.py
echo "enriched $(date)"
python3 scripts/04_validate_outputs.py
echo "CHAIN DONE $(date)"
