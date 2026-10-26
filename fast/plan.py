"""Plan one assembly with a released checkpoint; every accepted removal is validated in RedMax.

This is the public entry point. It runs the same action-level search as the thesis study,
without the locked study registry, and writes one trace (JSON Lines plus a SHA-256 sidecar).
"""

import argparse
import importlib.util
import json
from pathlib import Path

from fast import action_planner, planner_integration as integration
from plan_sequence import production_sim_labels as production


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_METHODS = {"fast": "fast_checkpoint", integration.MLP_CHECK_METHOD: "mlp_checkpoint",
                      "cebulla_gcn": "cebulla_checkpoint"}


def _sha256(path):
    return production._sha256_file(path) if path else None


def plan(assembly_dir, output, method="fast", seed=0, checkpoint=None, normalization=None,
         direction_prior=None, released_checkpoint=None, asset_folder=None, wall_timeout=7200.0):
    assembly_dir = Path(assembly_dir).resolve()
    asset_folder = str(Path(asset_folder or ROOT / "assets").resolve())
    feature_config = json.loads((ROOT / "configs" / "features.json").read_text(encoding="utf-8"))
    features = integration.OnlineFeatureBuilder(assembly_dir, feature_config)
    inputs = {}
    if method in CHECKPOINT_METHODS:
        if not (checkpoint and normalization):
            raise ValueError("--checkpoint and --normalization are required for " + method)
        inputs[CHECKPOINT_METHODS[method]] = checkpoint  # no expected hash: weights-only release file
    if method == "heur_out_direction_prior":
        inputs["direction_rates"] = json.loads(Path(direction_prior).read_text(encoding="utf-8"))["rates"]
    if method == "asap_released_gatv2":
        inputs["released_checkpoint"] = released_checkpoint
    norm = json.loads(Path(normalization).read_text(encoding="utf-8")) if normalization else None
    ranker = integration.make_ranker(method, seed, assembly_dir, features, norm, asset_folder=asset_folder, **inputs)
    from assets.load import load_part_ids
    parts = tuple(load_part_ids(assembly_dir))
    oracle = action_planner.ASAPDirectionOracle(asset_folder, assembly_dir, removal_timeout=60.0,
                                                stability_timeout=900.0, save_sdf=True)
    actions, summary = action_planner.run_action_dfs(
        parts, action_planner.ASAPStablePoseProvider(assembly_dir), ranker, oracle,
        max_action_validations=8 * (len(parts) - 1), method=method, seed=seed,
        assembly_id=assembly_dir.name, wall_timeout_seconds=wall_timeout, test_rows_loaded=0,
    )
    simulator = importlib.util.find_spec("redmax_py")
    header = {
        "record_type": "header", "schema_version": action_planner.TRACE_SCHEMA_VERSION, "complete": False,
        "assembly_id": assembly_dir.name, "input_sha256": features.input_sha256, "method": method, "seed": seed,
        "dataset_scope": "public", "mandatory_redmax_validation": True,
        "checkpoint_sha256": _sha256(checkpoint), "normalization_sha256": _sha256(normalization),
        "direction_prior_sha256": _sha256(direction_prior), "released_checkpoint_sha256": _sha256(released_checkpoint),
        "redmax_binary_sha256": production._sha256_file(simulator.origin), "test_rows_loaded": 0,
    }
    digest = action_planner.write_trace(output, header, actions, summary)
    return {"trace": str(output), "trace_sha256": digest, "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", required=True, help="ASAP assembly folder with OBJ parts")
    parser.add_argument("--output", required=True, help="trace path; must not exist")
    parser.add_argument("--method", default="fast", choices=integration.METHODS + (integration.MLP_CHECK_METHOD,))
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--checkpoint", help="weights-only state dict (fast, mlp_check, cebulla_gcn)")
    parser.add_argument("--normalization", help="feature normalization JSON of the checkpoint")
    parser.add_argument("--direction-prior", help="direction-prior JSON (heur_out_direction_prior)")
    parser.add_argument("--released-checkpoint", help="ASAP's released GATv2 checkpoint (asap_released_gatv2)")
    parser.add_argument("--asset-folder", help="default: the repository's assets folder")
    parser.add_argument("--wall-timeout", type=float, default=7200.0, help="seconds (thesis: 7200)")
    args = parser.parse_args(argv)
    result = plan(args.assembly_dir, args.output, args.method, args.seed, args.checkpoint, args.normalization,
                  args.direction_prior, args.released_checkpoint, args.asset_folder, args.wall_timeout)
    summary = result["summary"]
    print(json.dumps({"trace": result["trace"], "trace_sha256": result["trace_sha256"],
                      "solution_action_validation": summary["solution_action_validation"],
                      "action_validation_count": summary["action_validation_count"],
                      "stop_reason": summary["stop_reason"],
                      "budgets": {key: value["success"] for key, value in summary["budgets"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
