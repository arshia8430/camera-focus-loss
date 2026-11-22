#!/usr/bin/env bash
# Full pipeline. The order matters: exp1 writes the scene cache, exp2 writes the tuned learning rates.
# Usage: bash scripts/run_all.sh [results_dir]
set -e
OUT=${1:-results}
python sanity_checks.py --out "$OUT"
python exp1_rank_correlation.py --out "$OUT"
python exp2_convergence.py --out "$OUT"
python exp3_similarity_trajectory.py --out "$OUT"
python exp4_threshold_sensitivity.py --out "$OUT"
python exp5_gifs.py --out "$OUT"
