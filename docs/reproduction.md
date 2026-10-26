# Reproducing the thesis results

This page lists, for each data table and figure of the thesis, the command
that computes its numbers. The figures themselves were drawn from these JSON
outputs; the drawing scripts are not part of this repository.

All commands run from the repository root. `$FAST_DATA_ROOT` is the data root,
`$FAST_SOURCE_ROOT` the folder with the ASAP `training_assembly/` and
`test_assembly/` folders, and `$RUN` the planning-study run root
(`$FAST_DATA_ROOT/runs/planning-study-v4` by default). Stages up to the planning
study are run by `scripts/pipeline.sh` (see the README).

## Pins of generated artifacts

Each specification in `configs/` pins its inputs by SHA-256. A stage refuses to
start when an input differs. After you regenerate an artifact, write its new
hash into the pin before the next stage:

| Pin | Value after |
|---|---|
| `ranker_models.json` → `inputs.feature_build_sha256` | `features` (`features/scalar-features-v3/manifest.json`) |
| `pointnet.json` → `inputs.ranker_graph_audit_sha256` | `graphs` (`features/graph-records-v2/audit.json`) |
| `pointnet.json` → `inputs.ranker_training_audit_sha256`, `ranker_validation_*`, `ranker_bootstrap_plan_sha256` | `ranker` |
| `pointnet.json` → `control.checkpoints` | `ranker` (selected Graph Transformer checkpoints) |
| config-to-config pins (`path` and `sha256`) | any edit of a config; recompute with `sha256sum` |

The released dataset index needs no change: its manifest hash `b57465c0…` is
the pinned value.

## Planning study

1. `scripts/pipeline.sh planner-prep` builds the direction prior from the
   training split, runs one development smoke on a training assembly, and
   records the RedMax binary (`readiness`).
2. Write a disclosure record, a JSON file such as
   `{"status": "passed", "deviations": []}`, that lists any deviation from the
   protocol. Then freeze the registry. Test assemblies cannot be opened before
   this step.

   ```bash
   python -m fast.planner_integration freeze-registry --spec configs/pointnet.json \
     --data-root "$FAST_DATA_ROOT" --direction-prior "$FAST_DATA_ROOT/runs/pointnet-v1/planner/direction-prior.json" \
     --released-checkpoint "$FAST_RELEASED_CHECKPOINT" \
     --smoke-audit "$FAST_DATA_ROOT/runs/planner-smoke/audit.json" --disclosure disclosure.json \
     --redmax-readiness "$FAST_DATA_ROOT/runs/planner-smoke/redmax-readiness.json" \
     --output "$FAST_DATA_ROOT/registries/planning-study-v4.json"
   ```

   The registry binds the clean Git commit, the RedMax binary, all checkpoints,
   and a runtime label (`FAST_IMAGE`, default `local`) that every later job
   must match. ASAP's released GATv2 checkpoint is
   `https://people.csail.mit.edu/yunsheng/ASAP/pretrained_model.pt`
   (SHA-256 `430a5125…`).
3. `FAST_REGISTRY=<registry> scripts/pipeline.sh study-setup`, then
   `study-run` (all 6,399 jobs, or a range with `FAST_FIRST` and `FAST_LAST`).
   A job that fails technically is retried once from the same row. An assembly
   that still fails is excluded from every method (assembly-complete pairing).
   In the thesis, 3 of 237 test assemblies were excluded.
4. `FAST_EXCLUDE="<id> <id> …" scripts/pipeline.sh study-audit` writes
   `$RUN/aggregate/summary.json`. It holds success counts per method, seed,
   tier, and budget, the effort accounting, and the paired tier-stratified
   bootstrap with Holm correction across the four budgets (seed `20260916`,
   10,000 replicates).

Then:

```bash
# demonstrated-solvability lower bound
python -m fast.solvability manifest --run-root "$RUN" --exclude-assembly <id> … --output solvability-manifest.json
python -m fast.solvability analyze --run-root "$RUN" --manifest solvability-manifest.json \
  --manifest-sha256 <sha> --output lower-bound.json
python -m fast.solvability audit --run-root "$RUN" --manifest solvability-manifest.json \
  --manifest-sha256 <sha> --output lower-bound-audit.json    # independent path; must be identical

# matched success, aggregate resource yield, feature-group synthesis
python -m fast.posthoc --data-root "$FAST_DATA_ROOT" --run-root "$RUN" --registry <registry> \
  --exclude-assembly <id> … --output-root "$FAST_DATA_ROOT/runs/posthoc-v1"

# MLP planning check (seed 0, complete-case assemblies)
python -m fast.mlp_check prepare --repository-root . --data-root "$FAST_DATA_ROOT" --run-root "$RUN" \
  --study-registry <registry> --exclude-assembly <id> … --output-dir "$FAST_DATA_ROOT/registries/mlp-check-v1"
python -m fast.mlp_check evaluate ... --process <i>       # one job per matrix row
python -m fast.mlp_check audit-traces ...
python -m fast.mlp_check analyze ...
```

## Component ablation and test prediction quality

These analyses use the frozen thesis checkpoints, so their inputs are pinned in
code (`fast/ablation_closure.py`) and in `configs/component_ablation.json` and
`configs/test_prediction_quality.json`. They also read the per-assembly label
files of the label search (`labels/fast-labels-v1/`), which the `labels` stage
writes and the released dataset does not contain. To run them on your own
artifacts, replace those hashes.

- `python -m fast.ablation_closure --data-root <closure> plan|transfer|verify` copies a
  validation-only closure of the inputs. The source is set by
  `FAST_SOURCE_DATA_ROOT`, `FAST_SOURCE_GEOMETRY_ROOT`, and optionally
  `FAST_SOURCE_HOST` (SSH).
- `python -m fast.ablation_audit --data-root <closure> pretraining` runs the
  reference gate. Then `ablation_offline` runs experiments A, B, and E, and
  `ablation_replay` runs experiment C.
- Retraining without action features (experiment D) uses
  `python -m fast.ablation_runner --data-root <closure> prepare|launch|status`
  in `environment/cu128`; set `FAST_COMPAT_PYTHON` to the `cu121` Python.
  `ablation_feature` compares D0, D1, and D2.
- Test prediction quality: `python -m fast.prediction_quality --data-root <root>`
  with `fixtures`, `specify`, `preflight`, `record-d-decision`, `transfer`,
  `materialize` (needs `PYTHONHASHSEED=0`), `evaluate`, and `final-audit`.

## Tables and figures

| Thesis | Source |
|---|---|
| Table 4.1, Figure 4.2 (splits, part counts) | dataset `profile.json`; `configs/dataset_index.json` |
| Table 4.2 (simulation labels) | dataset `profile.json` |
| Table 5.1 (feature groups) | `fast/cad_features.py`, `configs/features.json` |
| Tables 6.2, 6.3 (encoder comparison, paired differences) | `ranker`: `runs/ranker-training-v2/validation/summary.json`, `bootstrap/report.json` |
| Table 6.4 (calibration) | `ranker`: `runs/ranker-training-v2/calibration/` |
| Table 6.5 (prediction quality, validation) | component ablation, experiment E |
| Table 6.6 (ranking of FAST variants) | component ablation, experiments A and B |
| Table 6.7 (label replay) | component ablation, experiment C |
| Figure 6.1 (predictive value of feature groups) | `screen`; `posthoc` → `factor_value` |
| Figure 6.2 (FAST retrained without action features) | component ablation, experiment D |
| Table 7.1, Figure 7.1 (planning success, paired differences) | `study-audit` → `success`, `paired_inference` |
| Demonstrated-solvability lower bound (Chapter 7) | `fast.solvability` |
| Figure 7.2 (effort at matched success) | `posthoc` → `matched_success` |
| Table 7.2 (MLP encoder in the planner) | `fast.mlp_check analyze` |
| Table 7.3 (prediction quality, test) | `fast.prediction_quality` |
| Figure 7.3 (Raspberry Pi example) | not included: the demonstration assembly is not part of this release |

PointNet results (Chapter 6) come from the `pointnet` stage
(`runs/pointnet-v1/validation/summary.json`, `representation-decision.json`).
