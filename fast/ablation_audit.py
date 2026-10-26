"""Independent component-ablation audits: the A/B/C/E pretraining gate (before any D job) and the final audit."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from fast import ablation_closure as closure
from fast import ablation_common as common


SCHEMA_VERSION = "fast.ablation-audit.v1"
PRETRAINING_OUTPUTS = ("transfer-manifest.json", "transfer-verification.json", "reference-reproduction.json", "score-ablation.json",
                       "granularity-ablation.json", "prediction-quality.json", "replay-equivalence.json", "small-label-replay.json",
                       "replay/per-assembly.json")
VOLATILE_KEYS = {"created_at", "implementation", "inference_seconds", "environment", "reference_reproduction_sha256",
                 "replay_equivalence_sha256", "per_assembly_sha256"}
FOCUSED_TESTS = ("tests.test_ablation_closure", "tests.test_ablation_offline", "tests.test_ablation_replay")


def _strip(value):
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _git(*args):
    return subprocess.run(["git", *args], cwd=str(common.REPOSITORY_ROOT), text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=True).stdout


def _published(commit):
    return "origin/component-ablation" in _git("branch", "-r", "--contains", commit)


def _replay_outputs(data_root):
    """Re-run A/B/E and C from the same verified closure into a temporary run root."""
    from fast import ablation_offline, ablation_replay
    source = common.run_root(data_root)
    temporary = Path(tempfile.mkdtemp(prefix="ablation-audit-"))
    try:
        target = temporary / "runs" / common.RUN_ID
        target.mkdir(parents=True)
        (target / "closure").symlink_to(source / "closure")
        for name in ("transfer-manifest.json", "transfer-verification.json"):
            shutil.copy2(source / name, target / name)
        ablation_offline.run(temporary)
        ablation_replay.run(temporary)
        comparison = OrderedDict()
        for name in PRETRAINING_OUTPUTS[2:]:
            published, replayed = common.read_json(source / name), common.read_json(target / name)
            comparison[name] = _strip(published) == _strip(replayed)
        comparison["replay/per-assembly.json byte-identical"] = common.sha256_file(source / "replay/per-assembly.json") == common.sha256_file(target / "replay/per-assembly.json")
        return comparison
    finally:
        shutil.rmtree(str(temporary), ignore_errors=True)


def _focused_tests():
    suite = unittest.defaultTestLoader.loadTestsFromNames(FOCUSED_TESTS)
    with open(os.devnull, "w") as sink:
        result = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return OrderedDict([("modules", list(FOCUSED_TESTS)), ("tests_run", result.testsRun), ("failures", len(result.failures)),
                        ("errors", len(result.errors)), ("skipped", len(result.skipped)), ("passed", result.wasSuccessful())])


def _c_recompute(report, rows):
    mismatches = 0
    for method, budgets in report["success"].items():
        for budget, entry in budgets.items():
            for seed, value in entry["seeds"].items():
                selected = [row["success"][budget[1:]] for row in rows if row["method"] == method and row["seed"] == int(seed)]
                if len(selected) != report["assembly_count"] or float(np.mean(selected)) != value:
                    mismatches += 1
    return mismatches


def pretraining(data_root):
    root = common.run_root(data_root)
    for name in ("pretraining-audit.json", "pretraining-gate.json"):
        common.require(not (root / name).exists(), name + " already exists")
    implementation = common.implementation_identity(require_clean=True)
    spec_sha256 = common.sha256_file(common.REPOSITORY_ROOT / "configs" / "component_ablation.json")
    checks = OrderedDict()
    outputs = OrderedDict((name, common.read_json(root / name)) for name in PRETRAINING_OUTPUTS)
    hashes = OrderedDict((name, common.sha256_file(root / name)) for name in PRETRAINING_OUTPUTS)

    verification = outputs["transfer-verification.json"]
    checks["closure_verified"] = verification["status"] == "passed" and verification["transfer_manifest_sha256"] == hashes["transfer-manifest.json"]
    manifest = outputs["transfer-manifest.json"]
    expected = {(row["namespace"], row["path"]): row for row in manifest["files"]}
    found = closure._local_files(common.closure_root(data_root))
    checks["closure_rehashed_no_unlisted_or_missing"] = set(found) == set(expected) and all(
        common.sha256_file(found[key]) == row["sha256"] for key, row in expected.items())
    checks["closure_has_no_test_path"] = not any(part in common.FORBIDDEN_PATH_PARTS for namespace, path in expected for part in Path(path).parts)

    scientific = [name for name in PRETRAINING_OUTPUTS[2:] if name != "replay/per-assembly.json"]
    checks["all_outputs_zero_test_rows"] = all(outputs[name].get("test_rows_loaded") == 0 for name in PRETRAINING_OUTPUTS)
    checks["single_spec_hash"] = {outputs[name]["spec_sha256"] for name in scientific} == {spec_sha256}
    commits = sorted({outputs[name]["implementation"]["commit"] for name in scientific} | {manifest["implementation"]["commit"], verification["implementation"]["commit"]})
    checks["clean_published_implementation_commits"] = all(outputs[name]["implementation"]["clean"] for name in scientific) and all(_published(commit) for commit in commits)

    reference = outputs["reference-reproduction.json"]
    checks["reference_gate_passed"] = reference["status"] == "passed" and all(reference["rounded_5_checks"].values()) and all(reference["population_count_checks"].values())
    for name, key in (("score-ablation.json", "reference_reproduction_sha256"), ("granularity-ablation.json", "reference_reproduction_sha256"),
                      ("prediction-quality.json", "reference_reproduction_sha256"), ("small-label-replay.json", "replay_equivalence_sha256"),
                      ("small-label-replay.json", "per_assembly_sha256")):
        target = {"reference_reproduction_sha256": "reference-reproduction.json", "replay_equivalence_sha256": "replay-equivalence.json",
                  "per_assembly_sha256": "replay/per-assembly.json"}[key]
        checks["binding_{}_{}".format(name, key)] = outputs[name][key] == hashes[target]

    plan_sha = "a3f37a724ac16af0dc10080d501f4d73318f2dbab322f77c14afbeaa70a670b8"
    checks["bootstrap_plan_bound_and_stratified"] = all(
        outputs[name]["bootstrap_plan"]["sha256"] == plan_sha and outputs[name]["bootstrap_plan"]["every_tier_block_draws_only_its_tier"]
        and outputs[name]["bootstrap_plan"]["tier_draws_per_replicate"] == {"large": 11, "medium": 65, "small": 115}
        for name in ("score-ablation.json", "granularity-ablation.json", "prediction-quality.json", "small-label-replay.json"))

    score, granularity, quality = outputs["score-ablation.json"], outputs["granularity-ablation.json"], outputs["prediction-quality.json"]
    tier_counts = {row: tuple(score["variants"]["A0"][row][key] for key in ("assembly_count", "stored_context_count", "ranking_eligible_context_count"))
                   for row in ("all", "small", "medium", "large")}
    same = True
    for report in (score, granularity):
        for variant in report["variants"].values():
            for row, counts in tier_counts.items():
                same = same and tuple(variant[row][key] for key in ("assembly_count", "stored_context_count", "ranking_eligible_context_count")) == counts
    for target in ("R", "T", "Y"):
        for row in tier_counts:
            same = same and quality["seed_mean"][target][row]["assembly_count"] == tier_counts[row][0]
            same = same and len({quality["per_seed"][seed][target][row]["counts"]["unresolved"] for seed in ("0", "1", "2")}) == 1
    checks["dec032_tier_counts_agree_across_A_B_E"] = same
    checks["population_counts"] = tier_counts == {"all": (191, 2947, 1988), "small": (115, 2233, 1693), "medium": (65, 611, 242), "large": (11, 103, 53)}

    equivalence, replay_report = outputs["replay-equivalence.json"], outputs["small-label-replay.json"]
    checks["c_real_equivalence_passed"] = equivalence["status"] == "passed" and len(equivalence["traces"]) == 6 and all(row["passed"] and not row["mismatches"] for row in equivalence["traces"])
    coverage = replay_report["coverage"]
    checks["c_coverage"] = coverage["assemblies"] == 115 and coverage["totals"] == {"candidates": 6627, "nonterminal_states": 788, "state_pose_contexts": 2233}
    checks["c_small_context_count_matches_A"] = coverage["totals"]["state_pose_contexts"] == tier_counts["small"][1]
    checks["c_small_T_labels_match_E"] = quality["seed_mean"]["T"]["small"]["label_total"] == coverage["totals"]["candidates"]
    rows = outputs["replay/per-assembly.json"]["rows"]
    ids = set(manifest["small_validation_assembly_ids"])
    checks["c_matrix_complete"] = len(rows) == 115 * 9 * 3 and {row["assembly_id"] for row in rows} == ids and len(
        {(row["assembly_id"], row["method"], row["seed"]) for row in rows}) == len(rows)
    checks["c_aggregates_recomputed_from_per_assembly_rows"] = _c_recompute(replay_report, rows) == 0

    tests = _focused_tests()
    checks["focused_and_synthetic_fixture_tests_passed"] = tests["passed"]
    replay_comparison = _replay_outputs(data_root)
    checks["deterministic_replay_of_all_outputs"] = all(replay_comparison.values())

    status = "passed" if all(checks.values()) else "failed"
    audit = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "pretraining_audit"), ("run_id", common.RUN_ID), ("status", status),
        ("audited_at", common.utc_now()), ("implementation", implementation), ("spec_sha256", spec_sha256),
        ("output_sha256", hashes), ("checks", checks), ("focused_tests", tests), ("replay_comparison", replay_comparison),
        ("implementation_commits", commits), ("tier_counts", {row: list(values) for row, values in tier_counts.items()}),
        ("test_rows_loaded", 0),
    ])
    audit_sha256 = common.write_json_once(root / "pretraining-audit.json", audit)
    common.require(status == "passed", "A/B/C/E pretraining audit failed; D is blocked")
    gate = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "pretraining_gate"), ("run_id", common.RUN_ID), ("status", "passed"),
        ("recorded_at", common.utc_now()), ("implementation", implementation), ("spec_sha256", spec_sha256),
        ("bound_outputs", OrderedDict(list(hashes.items()) + [("pretraining-audit.json", audit_sha256)])),
        ("gates", OrderedDict([("input_audit", True), ("frozen_reference_reproduction", True), ("replay_coverage", True),
                               ("replay_trace_equivalence", True), ("synthetic_control_flow_fixtures", True), ("focused_audit", True)])),
        ("d_training_authorized", True),
        ("note", "operational order only; A/B/C/E outcomes do not change D variants, masks, seeds, protocol, metrics, or reporting"),
        ("test_rows_loaded", 0),
    ])
    gate_sha256 = common.write_json_once(root / "pretraining-gate.json", gate)
    return {"status": "passed", "pretraining-audit.json": audit_sha256, "pretraining-gate.json": gate_sha256, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("command", choices=("pretraining",))
    args = parser.parse_args(argv)
    print(common.canonical_json(pretraining(args.data_root)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
