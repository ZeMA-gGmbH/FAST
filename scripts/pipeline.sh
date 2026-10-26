#!/usr/bin/env bash
# Run one FAST pipeline stage locally and sequentially.
#
#   scripts/pipeline.sh <stage>
#
# Environment:
#   FAST_SOURCE_ROOT  ASAP assemblies: contains training_assembly/ and test_assembly/   (required)
#   FAST_DATA_ROOT    data root for labels, index, features, runs (required)
#   FAST_RUN_ROOT     planning-study run root   (default: $FAST_DATA_ROOT/runs/planning-study-v4)
#   FAST_REGISTRY     locked planning-study registry (stages study-*)
#   FAST_RELEASED_CHECKPOINT  ASAP's released GATv2 checkpoint (stages planner-prep, study-run)
#   FAST_FIRST, FAST_LAST     process range for study-run (default: all)
#
# Stages, in order: labels, index, features, screen, graphs, ranker, pointnet, planner-prep,
# study-setup, study-run, study-audit. Each stage refuses to overwrite earlier outputs.
# The thesis ran these jobs in parallel on a cluster; locally they run one after another.
set -euo pipefail

cd "$(dirname "$0")/.."
: "${FAST_SOURCE_ROOT:?set FAST_SOURCE_ROOT}" "${FAST_DATA_ROOT:?set FAST_DATA_ROOT}"
RUN_ROOT="${FAST_RUN_ROOT:-$FAST_DATA_ROOT/runs/planning-study-v4}"
PRIOR="$FAST_DATA_ROOT/runs/pointnet-v1/planner/direction-prior.json"
export MPLBACKEND=Agg PYTHONHASHSEED=0
PY="${PYTHON:-python}"

case "${1:-}" in
labels)
    while IFS=$'\t' read -r split assembly tier parts sha _; do
        "$PY" -m plan_sequence.prepare_sdf_cache --assembly-dir "$FAST_SOURCE_ROOT/$split/$assembly" \
            --cache-dir "$FAST_DATA_ROOT/cache/sdf/$sha" --expected-input-sha256 "$sha"
        for action in --labels --audit; do
            "$PY" -m plan_sequence.production_sim_labels --source-root "$FAST_SOURCE_ROOT" \
                --output-root "$FAST_DATA_ROOT" --asset-folder assets --split "$split" --assembly-id "$assembly" $action
        done
    done < configs/label_production_jobs.tsv
    "$PY" -m plan_sequence.production_sim_labels --source-root "$FAST_SOURCE_ROOT" \
        --output-root "$FAST_DATA_ROOT" --combined-audit ;;
index)
    "$PY" -m plan_sequence.build_dataset_index --data-root "$FAST_DATA_ROOT" ;;
features)
    "$PY" - "$FAST_DATA_ROOT/datasets/fast-labels-v1/assemblies.jsonl" <<'PYEOF' |
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    if row["included"]:
        print(row["source_split"], row["assembly_id"])
PYEOF
    while read -r split assembly; do
        "$PY" -m fast.cad_features --source-root "$FAST_SOURCE_ROOT" --data-root "$FAST_DATA_ROOT" \
            --extract --source-split "$split" --assembly-id "$assembly"
    done
    "$PY" -m fast.cad_features --source-root "$FAST_SOURCE_ROOT" --data-root "$FAST_DATA_ROOT" --audit ;;
screen)
    while IFS=$'\t' read -r model feature_set; do
        "$PY" -m fast.feature_screen --source-root "$FAST_SOURCE_ROOT" --data-root "$FAST_DATA_ROOT" \
            --screen --model "$model" --feature-set "$feature_set"
    done < configs/feature_screen_jobs.tsv
    "$PY" -m fast.feature_screen --source-root "$FAST_SOURCE_ROOT" --data-root "$FAST_DATA_ROOT" --audit ;;
graphs)
    common=(--source-root "$FAST_SOURCE_ROOT" --data-root "$FAST_DATA_ROOT")
    "$PY" -m fast.graph_materialize "${common[@]}" prepare
    jobs=$(($(wc -l < "$FAST_DATA_ROOT/features/graph-records-v2/development-jobs.tsv") - 1))
    for index in $(seq 0 $((jobs - 1))); do
        "$PY" -m fast.graph_materialize "${common[@]}" materialize --job-index "$index"
    done
    "$PY" -m fast.graph_materialize "${common[@]}" audit ;;
ranker)
    common=(--source-root "$FAST_SOURCE_ROOT" --data-root "$FAST_DATA_ROOT")
    "$PY" -m fast.ranker_train "${common[@]}" setup
    "$PY" -m fast.ranker_train "${common[@]}" gpu-pilot
    for process in $(seq 0 14); do
        "$PY" -m fast.ranker_train "${common[@]}" train --process "$process"
    done
    "$PY" -m fast.ranker_train "${common[@]}" audit
    "$PY" -m fast.ranker_validate "${common[@]}" validate
    "$PY" -m fast.ranker_validate "${common[@]}" audit ;;
pointnet)
    common=(--source-root "$FAST_SOURCE_ROOT" --data-root "$FAST_DATA_ROOT")
    "$PY" -m fast.pointnet_points "${common[@]}" prepare
    jobs=$(($(wc -l < "$FAST_DATA_ROOT/features/pointnet-v1/development-jobs.tsv") - 1))
    for index in $(seq 0 $((jobs - 1))); do
        "$PY" -m fast.pointnet_points "${common[@]}" materialize --job-index "$index"
    done
    "$PY" -m fast.pointnet_points "${common[@]}" audit
    "$PY" -m fast.pointnet_train "${common[@]}" setup
    "$PY" -m fast.pointnet_train "${common[@]}" gpu-pilot
    for process in $(seq 0 5); do
        "$PY" -m fast.pointnet_train "${common[@]}" train --process "$process"
    done
    "$PY" -m fast.pointnet_train "${common[@]}" audit
    "$PY" -m fast.pointnet_validate "${common[@]}" validate
    "$PY" -m fast.pointnet_validate "${common[@]}" audit ;;
planner-prep)
    : "${FAST_RELEASED_CHECKPOINT:?set FAST_RELEASED_CHECKPOINT}"
    "$PY" -m fast.planner_integration direction-prior --graph-root "$FAST_DATA_ROOT/features/graph-records-v2" --output "$PRIOR"
    "$PY" -m fast.planner_integration run --spec configs/pointnet.json --data-root "$FAST_DATA_ROOT" --asset-folder assets \
        --assembly-dir "$FAST_SOURCE_ROOT/training_assembly/00014" --output "$FAST_DATA_ROOT/runs/planner-smoke/trace.jsonl" \
        --method fast --seed 0 --released-checkpoint "$FAST_RELEASED_CHECKPOINT" --direction-prior "$PRIOR" --development
    "$PY" -m fast.planner_integration audit-smoke --trace "$FAST_DATA_ROOT/runs/planner-smoke/trace.jsonl" \
        --output "$FAST_DATA_ROOT/runs/planner-smoke/audit.json"
    "$PY" -m fast.planner_integration readiness --output "$FAST_DATA_ROOT/runs/planner-smoke/redmax-readiness.json"
    echo "Next: write a disclosure record and run 'python -m fast.planner_integration freeze-registry'; see docs/reproduction.md." ;;
study-setup|study-run|study-audit)
    : "${FAST_REGISTRY:?set FAST_REGISTRY}"
    common=(--spec configs/pointnet.json --data-root "$FAST_DATA_ROOT" --source-root "$FAST_SOURCE_ROOT"
            --run-root "$RUN_ROOT" --registry "$FAST_REGISTRY" --registry-sha256 "$(sha256sum "$FAST_REGISTRY" | cut -d' ' -f1)")
    case "$1" in
    study-setup) "$PY" -m fast.planning_study "${common[@]}" setup ;;
    study-audit) "$PY" -m fast.planning_study "${common[@]}" audit ${FAST_EXCLUDE:+$(printf -- '--exclude-assembly %s ' $FAST_EXCLUDE)} ;;
    study-run)
        : "${FAST_RELEASED_CHECKPOINT:?set FAST_RELEASED_CHECKPOINT}"
        last=$(($(wc -l < "$RUN_ROOT/jobs.tsv") - 2))
        for process in $(seq "${FAST_FIRST:-0}" "${FAST_LAST:-$last}"); do
            "$PY" -m fast.planning_study "${common[@]}" evaluate --asset-folder assets \
                --released-checkpoint "$FAST_RELEASED_CHECKPOINT" --direction-prior "$PRIOR" --process "$process"
        done ;;
    esac ;;
*)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
