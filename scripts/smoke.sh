#!/usr/bin/env bash
# Plan one assembly with the seed-0 FAST checkpoint: features -> inference -> RedMax-validated search.
#
#   scripts/smoke.sh <assembly-dir> [output-root]
#
# Example: scripts/smoke.sh "$FAST_SOURCE_ROOT/training_assembly/00014" outputs
set -euo pipefail
cd "$(dirname "$0")/.."
assembly="${1:?usage: scripts/smoke.sh <assembly-dir> [output-root]}"
output="${2:-outputs}/smoke/$(basename "$assembly").jsonl"
export MPLBACKEND=Agg OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
"${PYTHON:-python}" -m fast.plan --assembly-dir "$assembly" --checkpoint checkpoints/fast_gt_seed0.pth \
    --normalization checkpoints/normalization.json --output "$output"
