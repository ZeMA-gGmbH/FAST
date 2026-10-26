"""Locked planning-study job preparation and single-assembly planner execution."""

import argparse
import csv
import importlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from fast import graph_contract, ranker_train, action_planner, complete_case
from fast.planner_integration import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, LEGACY_METHODS, METHODS, LegacyPartGenerator, frozen_artifacts, run_planner
from plan_sequence import production_sim_labels as production


SETUP_SCHEMA = "fast.study-setup.v1"
AUDIT_SCHEMA = "fast.study-audit.v1"
COMPLETE_CASE_AUDIT_SCHEMA = "fast.complete-case-audit.v1"


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _locked(registry_path, registry_sha256, require_current_implementation=True):
    registry = action_planner.validate_registry(registry_path, registry_sha256)
    simulator = importlib.import_module("redmax_py")
    simulator_path = Path(simulator.__file__).resolve()
    graph_contract._require(
        simulator_path.is_file()
        and (not require_current_implementation
             or registry.get("implementation") == ranker_train._repository_identity(require_clean=True))
        and registry.get("container") == os.environ.get("FAST_IMAGE", "local")
        and registry.get("redmax", {}).get("binary_sha256") == production._sha256_file(simulator_path),
        "planning-study runtime differs from registry",
    )
    return registry


def setup(spec_path, data_root, source_root, run_root, registry_path, registry_sha256):
    """Validate the lock before opening the frozen test assembly index."""
    registry = _locked(registry_path, registry_sha256)
    artifacts = frozen_artifacts(spec_path, data_root)
    data_root, source_root, run_root = Path(data_root).resolve(), Path(source_root).resolve(), Path(run_root).resolve()
    graph_contract._require(not run_root.exists(), "planning-study run root cannot be overwritten")
    ranker_spec = _read_json(artifacts["repository_root"] / artifacts["spec"]["inputs"]["ranker_spec"]["path"])
    dataset_root = data_root / "datasets" / "fast-labels-v1"
    manifest_path = dataset_root / "manifest.json"
    manifest = _read_json(manifest_path)
    graph_contract._require(production._sha256_file(manifest_path) == ranker_spec["inputs"]["dataset_manifest_sha256"],
                     "planning-study dataset manifest differs")
    assemblies_path = dataset_root / manifest["artifacts"]["assemblies"]["path"]
    expected = registry["dataset_protocol"]["artifact_hashes"]["assemblies"]
    graph_contract._require(production._sha256_file(assemblies_path) == expected["sha256"], "planning-study assembly index differs")
    with assemblies_path.open("r", encoding="utf-8") as stream:
        assemblies = [json.loads(line) for line in stream if line.strip()]
    assemblies = [row for row in assemblies if row["dataset_split"] == "test" and row["included"]]
    graph_contract._require(expected["row_count"] == 2132 and len(assemblies) == 237, "planning-study test population differs")
    graph_contract._require(all(row["source_split"] == "test_assembly" for row in assemblies), "planning-study source split differs")
    matrix = [
        ("action_validation", method, seed, row)
        for method in METHODS for seed in (0, 1, 2) for row in assemblies
    ] + [
        ("asap_legacy_evaluation", method, seed, row)
        for method in LEGACY_METHODS for seed in (0, 1, 2) for row in assemblies
    ]
    jobs = [
        {"process": process, "counter": counter, "method": method, "seed": seed,
         "assembly_id": row["assembly_id"], "tier": row["tier"], "part_count": row["part_count"],
         "input_sha256": row["input_sha256"]}
        for process, (counter, method, seed, row) in enumerate(matrix)
    ]
    run_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".planning-study-", dir=str(run_root.parent)) as temporary:
        staging = Path(temporary) / run_root.name
        staging.mkdir()
        job_path = staging / "jobs.tsv"
        fields = ("process", "counter", "method", "seed", "assembly_id", "tier", "part_count", "input_sha256")
        with job_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(jobs)
        report = {
            "schema_version": SETUP_SCHEMA, "status": "passed", "registry_sha256": registry_sha256,
            "job_table": job_path.name, "job_table_sha256": production._sha256_file(job_path),
            "test_assembly_count": len(assemblies), "method_count": len(METHODS), "seed_count": 3,
            "primary_job_count": len(METHODS) * 3 * len(assemblies),
            "legacy_job_count": len(LEGACY_METHODS) * 3 * len(assemblies),
            "job_count": len(jobs), "test_rows_loaded": len(assemblies),
            "aggregate_only": registry["aggregate_outputs"]["per_assembly_test_payload_export"] is False,
            "source_root": str(source_root),
        }
        from fast.graph_materialize import _write_json
        _write_json(staging / "setup.json", report)
        os.replace(str(staging), str(run_root))
    return report


def _job(run_root, process, registry_sha256):
    run_root = Path(run_root)
    setup_report = _read_json(run_root / "setup.json")
    graph_contract._require(setup_report.get("status") == "passed" and setup_report.get("registry_sha256") == registry_sha256,
                     "planning-study setup differs")
    job_path = run_root / setup_report["job_table"]
    graph_contract._require(production._sha256_file(job_path) == setup_report["job_table_sha256"], "planning-study job table differs")
    with job_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    graph_contract._require(0 <= process < len(rows) and int(rows[process]["process"]) == process, "planning-study process differs")
    return rows[process]


def evaluate(spec_path, data_root, source_root, asset_folder, run_root, registry_path, registry_sha256,
             released_checkpoint, direction_prior, process):
    _locked(registry_path, registry_sha256)
    row = _job(run_root, process, registry_sha256)
    assembly_dir = Path(source_root) / "test_assembly" / row["assembly_id"]
    graph_contract._require(production._hash_assembly(assembly_dir) == row["input_sha256"], "planning-study assembly input differs")
    trace_group = "primary" if row["counter"] == "action_validation" else "legacy"
    output = Path(run_root) / "traces" / trace_group / row["method"] / "seed-{}".format(row["seed"]) / (row["assembly_id"] + ".jsonl")
    if row["counter"] == "asap_legacy_evaluation":
        return run_legacy_planner(
            asset_folder, assembly_dir, output, row["method"], int(row["seed"]), released_checkpoint,
            registry_sha256, row["input_sha256"], registry_path,
        )
    result = run_planner(
        spec_path, data_root, asset_folder, assembly_dir, output, row["method"], int(row["seed"]),
        released_checkpoint=released_checkpoint, direction_prior_path=direction_prior,
        registry=registry_path, registry_sha256=registry_sha256, development=False,
    )
    graph_contract._require(result["summary"]["test_rows_loaded"] == 1, "planning-study trace test count differs")
    return result


def run_legacy_planner(asset_folder, assembly_dir, output, method, seed, released_checkpoint,
                       registry_sha256, input_sha256, registry_path):
    """Run the unchanged bundled ASAP counter separately from the primary metric."""
    from plan_sequence.planner.dfs import DFSSequencePlanner

    generator = LegacyPartGenerator(method, seed, asset_folder, assembly_dir, released_checkpoint)
    planner = DFSSequencePlanner(generator, num_proc=1, save_sdf=True)
    planner.seed(seed)
    started = time.perf_counter()
    tree = planner.plan(budget=400, max_grippers=2, max_poses=3, pose_reuse=0,
                        early_term=False, timeout=7200.0)
    wall_seconds = time.perf_counter() - started
    stats = planner.get_stats(tree)
    first_solution = stats["n_eval"] if stats["success"] else None
    header = {
        "record_type": "header", "schema_version": action_planner.TRACE_SCHEMA_VERSION,
        "complete": False, "assembly_id": Path(assembly_dir).name, "input_sha256": input_sha256,
        "method": method, "seed": seed, "dataset_scope": "locked_test",
        "registry_sha256": registry_sha256, "counter": "asap_legacy_evaluation",
        "redmax_binary_sha256": _read_json(registry_path)["redmax"]["binary_sha256"],
        "test_rows_loaded": 1,
    }
    summary = {
        "record_type": "summary", "schema_version": action_planner.TRACE_SCHEMA_VERSION,
        "complete": True, "assembly_id": Path(assembly_dir).name, "method": method, "seed": seed,
        "counter": "asap_legacy_evaluation", "budgets": {
            str(budget): {"absolute_legacy_evaluations": budget,
                          "success": first_solution is not None and first_solution <= budget}
            for budget in (50, 400)
        },
        "solution_legacy_evaluation": first_solution, "legacy_evaluation_count": planner.n_eval,
        "stop_reason": planner.stop_msg, "wall_seconds": wall_seconds, "wall_timeout_seconds": 7200.0,
        "bundled_six_direction_redmax": True, "mandatory_primary_action_validation": False,
        "unvalidated_two_part_leaf": True, "test_rows_loaded": 1,
    }
    graph_contract._require(planner.stop_msg != "exception", "legacy ASAP planner failed")
    digest = action_planner.write_trace(output, header, [], summary)
    return {"trace": str(output), "trace_sha256": digest, "summary": summary}


def _paired_statistics(rows, expected_assembly_count=237):
    assemblies = sorted({(row["tier"], row["assembly_id"]) for row in rows})
    values = {}
    for row in rows:
        for budget, success in row["success"].items():
            values.setdefault((row["method"], row["assembly_id"], budget), []).append(float(success))
    graph_contract._require(
        len(assemblies) == expected_assembly_count
        and all(len(seed_values) == 3 for seed_values in values.values()),
        "paired planning-study outcomes differ",
    )
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples = np.empty((BOOTSTRAP_REPLICATES, len(assemblies)), dtype=np.int64)
    offset = 0
    for tier in ("small", "medium", "large"):
        indices = np.asarray([index for index, row in enumerate(assemblies) if row[0] == tier])
        samples[:, offset:offset + len(indices)] = rng.choice(indices, (BOOTSTRAP_REPLICATES, len(indices)), replace=True)
        offset += len(indices)
    comparisons = {}
    for baseline in METHODS[:-1]:
        budget_rows = {}
        for budget in (1, 2, 4, 8):
            fast = np.asarray([np.mean(values[("fast", assembly_id, str(budget))]) for _, assembly_id in assemblies])
            control = np.asarray([np.mean(values[(baseline, assembly_id, str(budget))]) for _, assembly_id in assemblies])
            differences = fast - control
            estimate = float(differences.mean())
            replicates = differences[samples].mean(axis=1)
            null_replicates = (differences - estimate)[samples].mean(axis=1)
            p_value = float((np.count_nonzero(np.abs(null_replicates) >= abs(estimate)) + 1) / (BOOTSTRAP_REPLICATES + 1))
            budget_rows[str(budget)] = {
                "fast_minus_baseline": estimate,
                "percentile_2_5": float(np.percentile(replicates, 2.5)),
                "percentile_97_5": float(np.percentile(replicates, 97.5)),
                "two_sided_p_value": p_value,
            }
        previous = 0.0
        ordered = sorted(budget_rows, key=lambda budget: budget_rows[budget]["two_sided_p_value"])
        for index, budget in enumerate(ordered):
            adjusted = min(1.0, (len(ordered) - index) * budget_rows[budget]["two_sided_p_value"])
            previous = max(previous, adjusted)
            budget_rows[budget]["holm_adjusted_p_value"] = previous
            budget_rows[budget]["reject_at_0_05"] = previous <= 0.05
        comparisons[baseline] = budget_rows
    return {
        "experimental_unit": "assembly", "seed_aggregation": "mean within assembly",
        "bootstrap": "paired tier-stratified assembly resampling", "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "hypothesis_test": "two-sided centered paired tier-stratified bootstrap",
        "multiple_comparison_correction": "Holm across four budgets within each baseline contrast",
        "familywise_alpha": 0.05, "comparisons": comparisons,
    }


def audit(run_root, registry_path, registry_sha256, *, legacy_run_root=None,
          legacy_registry_sha256=None, primary_retry_run_root=None,
          primary_retry_registry_sha256=None, legacy_retry_run_root=None,
          legacy_retry_registry_sha256=None, output_root=None,
          allow_historical_implementation=False, attempt_ledger_sha256=None,
          complete_case_exclusions=None):
    registry = _locked(registry_path, registry_sha256, not allow_historical_implementation)
    run_root = Path(run_root)
    setup_report = _read_json(run_root / "setup.json")
    graph_contract._require(
        setup_report.get("primary_job_count") == 4266 and setup_report.get("legacy_job_count") == 2133
        and setup_report.get("job_count") == 6399 and setup_report.get("registry_sha256") == registry_sha256,
                     "planning-study setup cannot be audited")
    job_path = run_root / setup_report["job_table"]
    graph_contract._require(production._sha256_file(job_path) == setup_report["job_table_sha256"], "planning-study job table differs")
    with job_path.open("r", encoding="utf-8", newline="") as stream:
        jobs = list(csv.DictReader(stream, delimiter="\t"))
    excluded = frozenset(complete_case_exclusions or ())
    if excluded:
        jobs, retained_assemblies = complete_case.retained_jobs(
            jobs, excluded, METHODS, LEGACY_METHODS
        )
    else:
        retained_assemblies = {row["assembly_id"] for row in jobs}
    counts = {
        method: {
            str(seed): {
                tier: {str(budget): {"success": 0, "total": 0} for budget in (1, 2, 4, 8)}
                for tier in ("all", "small", "medium", "large")
            }
            for seed in (0, 1, 2)
        }
        for method in METHODS
    }
    accounting = {method: {"action_validations": 0, "physics_rollouts": 0, "wall_seconds": 0.0,
                            "removability_seconds": 0.0, "stability_seconds": 0.0,
                            "model_ranking_seconds": 0.0, "overhead_seconds": 0.0,
                            "wall_timeouts": 0, "censored_actions": 0} for method in METHODS}
    legacy = {
        method: {
            str(seed): {
                tier: {str(budget): {"success": 0, "total": 0} for budget in (50, 400)}
                for tier in ("all", "small", "medium", "large")
            }
            for seed in (0, 1, 2)
        }
        for method in LEGACY_METHODS
    }
    legacy_accounting = {method: {"legacy_evaluations": 0, "wall_seconds": 0.0, "timeouts": 0}
                         for method in LEGACY_METHODS}
    primary_outcomes = []
    valid_primary_trace_count = 0
    valid_legacy_trace_count = 0
    for job in jobs:
        trace_group = "primary" if job["counter"] == "action_validation" else "legacy"
        trace_root, trace_registry_sha256 = run_root, registry_sha256
        original = run_root / "traces" / trace_group / job["method"] / "seed-{}".format(job["seed"]) / (job["assembly_id"] + ".jsonl")
        if job["counter"] == "asap_legacy_evaluation" and legacy_run_root is not None:
            trace_root, trace_registry_sha256 = Path(legacy_run_root), legacy_registry_sha256
            recovery_trace = (trace_root / "traces" / trace_group / job["method"]
                              / "seed-{}".format(job["seed"]) / (job["assembly_id"] + ".jsonl"))
            if not recovery_trace.exists() and legacy_retry_run_root is not None:
                trace_root = Path(legacy_retry_run_root)
                trace_registry_sha256 = legacy_retry_registry_sha256
        elif job["counter"] == "action_validation" and not original.exists() and primary_retry_run_root is not None:
            trace_root, trace_registry_sha256 = Path(primary_retry_run_root), primary_retry_registry_sha256
        trace = trace_root / "traces" / trace_group / job["method"] / "seed-{}".format(job["seed"]) / (job["assembly_id"] + ".jsonl")
        digest_path = Path(str(trace) + ".sha256")
        graph_contract._require(trace.is_file() and digest_path.is_file(), "planning-study trace is missing")
        digest = digest_path.read_text(encoding="ascii").split()[0]
        graph_contract._require(digest == production._sha256_file(trace), "planning-study trace hash differs")
        with trace.open("r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        header, actions, summary = rows[0], rows[1:-1], rows[-1]
        if job["counter"] == "asap_legacy_evaluation":
            graph_contract._require(
                not actions and header.get("counter") == summary.get("counter") == "asap_legacy_evaluation"
                and header.get("dataset_scope") == "locked_test" and header.get("registry_sha256") == trace_registry_sha256
                and header.get("redmax_binary_sha256") == registry["redmax"]["binary_sha256"]
                and header.get("method") == job["method"] and header.get("seed") == int(job["seed"])
                and header.get("assembly_id") == job["assembly_id"] and header.get("input_sha256") == job["input_sha256"]
                and header.get("test_rows_loaded") == summary.get("test_rows_loaded") == 1
                and summary.get("complete") is True and summary.get("bundled_six_direction_redmax") is True
                and summary.get("mandatory_primary_action_validation") is False
                and summary.get("unvalidated_two_part_leaf") is True
                and set(summary.get("budgets", {})) == {"50", "400"}
                and all(summary["budgets"][str(budget)].get("absolute_legacy_evaluations") == budget
                        for budget in (50, 400)),
                "planning-study legacy trace contract differs",
            )
            for tier in ("all", job["tier"]):
                for budget in (50, 400):
                    cell = legacy[job["method"]][job["seed"]][tier][str(budget)]
                    cell["success"] += int(summary["budgets"][str(budget)]["success"])
                    cell["total"] += 1
            method_accounting = legacy_accounting[job["method"]]
            method_accounting["legacy_evaluations"] += summary["legacy_evaluation_count"]
            method_accounting["wall_seconds"] += summary["wall_seconds"]
            method_accounting["timeouts"] += int(summary["stop_reason"] == "timeout")
            valid_legacy_trace_count += 1
            continue
        graph_contract._require(
            header.get("dataset_scope") == "locked_test" and header.get("registry_sha256") == trace_registry_sha256
            and header.get("redmax_binary_sha256") == registry["redmax"]["binary_sha256"]
            and header.get("test_rows_loaded") == 1
            and header.get("method") == job["method"] and header.get("seed") == int(job["seed"])
            and header.get("assembly_id") == job["assembly_id"] and header.get("input_sha256") == job["input_sha256"]
            and summary.get("complete") is True and summary.get("test_rows_loaded") == 1
            and summary.get("mandatory_redmax_validation") is True and summary.get("unvalidated_leaf_shortcut") is False
            and summary.get("action_validation_count") == len(actions)
            and set(summary.get("budgets", {})) == {"1", "2", "4", "8"}
            and all(summary["budgets"][str(budget)].get("absolute_action_validations")
                    == budget * (int(job["part_count"]) - 1) for budget in (1, 2, 4, 8))
            and all(not row["accepted_transition"] or (row["R"]["label"] is True and row["T"]["label"] is True) for row in actions)
            and not any(row["R"]["status"] == "error" or (row["T"] and row["T"]["status"] == "error") for row in actions),
            "planning-study trace contract differs",
        )
        for tier in ("all", job["tier"]):
            for budget in (1, 2, 4, 8):
                cell = counts[job["method"]][job["seed"]][tier][str(budget)]
                cell["success"] += int(summary["budgets"][str(budget)]["success"])
                cell["total"] += 1
        primary_outcomes.append({
            "method": job["method"], "seed": int(job["seed"]), "assembly_id": job["assembly_id"],
            "tier": job["tier"],
            "success": {str(budget): bool(summary["budgets"][str(budget)]["success"]) for budget in (1, 2, 4, 8)},
        })
        method_accounting = accounting[job["method"]]
        method_accounting["action_validations"] += summary["action_validation_count"]
        method_accounting["physics_rollouts"] += summary["redmax_rollout_count"] + summary["stability_restart_count"]
        for field in ("wall_seconds", "removability_seconds", "stability_seconds", "model_ranking_seconds", "overhead_seconds"):
            method_accounting[field] += summary[field]
        method_accounting["wall_timeouts"] += int(summary["stop_reason"] == "wall_timeout")
        method_accounting["censored_actions"] += summary["censored_action_count"]
        valid_primary_trace_count += 1
    for method in METHODS:
        for seed in (0, 1, 2):
            for tier in ("all", "small", "medium", "large"):
                for budget in (1, 2, 4, 8):
                    cell = counts[method][str(seed)][tier][str(budget)]
                    cell["rate"] = cell["success"] / cell["total"]
    for method in LEGACY_METHODS:
        for seed in (0, 1, 2):
            for tier in ("all", "small", "medium", "large"):
                for budget in (50, 400):
                    cell = legacy[method][str(seed)][tier][str(budget)]
                    cell["rate"] = cell["success"] / cell["total"]
    report = {
        "schema_version": COMPLETE_CASE_AUDIT_SCHEMA if excluded else AUDIT_SCHEMA,
        "status": "passed", "registry_sha256": registry_sha256,
        "job_count": len(jobs), "test_assembly_count": len(retained_assemblies),
        "success": counts, "accounting": accounting,
        "legacy_success": legacy, "legacy_accounting": legacy_accounting,
        "paired_inference": _paired_statistics(primary_outcomes, len(retained_assemblies)),
        "claim_boundary": ("complete paired planner outcomes on {} of {} predefined test assemblies; "
                           "{} assemblies excluded after repeatable simulator execution failures; "
                           "no per-assembly test payload is exported".format(
                               len(retained_assemblies), setup_report["test_assembly_count"], len(excluded))
                           if excluded else
                           "locked aggregate planner outcomes; no per-assembly test payload is exported"),
        "test_rows_loaded": len(retained_assemblies), "per_assembly_test_payload_export": False,
        "valid_primary_trace_count": valid_primary_trace_count,
        "valid_legacy_trace_count": valid_legacy_trace_count,
    }
    if excluded:
        report["complete_case"] = {
            "rule": "assembly-wide exclusion after a failed retry", "predefined_test_assembly_count": setup_report["test_assembly_count"],
            "excluded_assembly_count": len(excluded), "analyzed_assembly_count": len(retained_assemblies),
            "primary_job_count": sum(row["counter"] == "action_validation" for row in jobs),
            "legacy_job_count": sum(row["counter"] == "asap_legacy_evaluation" for row in jobs), "job_count": len(jobs),
            "exclusion_unit": "assembly", "outcome_imputation": False,
            "excluded_attempt_classification": "simulator_execution_failure",
        }
    if legacy_run_root is not None:
        report["recovery"] = {
            "parent_registry_sha256": registry_sha256,
            "legacy_registry_sha256": legacy_registry_sha256,
            "legacy_retry_registry_sha256": legacy_retry_registry_sha256,
            "primary_retry_registry_sha256": primary_retry_registry_sha256,
            "attempt_ledger_sha256": attempt_ledger_sha256,
        }
    output = Path(output_root or run_root) / "aggregate" / "summary.json"
    graph_contract._require(not output.exists(), "planning-study aggregate cannot be overwritten")
    from fast.graph_materialize import _write_json
    _write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("setup")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--asset-folder", required=True)
    evaluate_parser.add_argument("--released-checkpoint", required=True)
    evaluate_parser.add_argument("--direction-prior", required=True)
    evaluate_parser.add_argument("--process", type=int, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--exclude-assembly", action="append", default=[],
                              help="assembly excluded from every method after a failed retry (repeatable)")
    args = parser.parse_args(argv)
    if args.command == "setup":
        result = setup(args.spec, args.data_root, args.source_root, args.run_root, args.registry, args.registry_sha256)
    elif args.command == "evaluate":
        result = evaluate(args.spec, args.data_root, args.source_root, args.asset_folder, args.run_root,
                          args.registry, args.registry_sha256, args.released_checkpoint,
                          args.direction_prior, args.process)
    else:
        result = audit(args.run_root, args.registry, args.registry_sha256,
                       complete_case_exclusions=args.exclude_assembly)
    print(production._canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
