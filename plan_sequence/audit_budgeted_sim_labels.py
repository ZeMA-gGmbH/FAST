"""Audit deterministic budgeted mixed-search FAST labels."""

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from plan_sequence import audit_reachable_sim_labels as candidate_audit
from plan_sequence import dfs_search
from plan_sequence import generate_budgeted_sim_labels as budgeted
from plan_sequence import generate_reachable_sim_labels as reachable
from plan_sequence import generate_sim_labels as labels
from plan_sequence import mixed_search
from plan_sequence import prepare_pose_cache as pose_cache_tools
from plan_sequence import prepare_sdf_cache as sdf_cache_tools


SCHEMA_VERSION = "fast.budgeted-label-audit.v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _existing_report(path, spec_sha256, schema_version=SCHEMA_VERSION):
    if not path.exists():
        return None
    records = reachable._read_jsonl(path)
    if len(records) != 1 or records[0].get("schema_version") != schema_version:
        raise RuntimeError("existing audit is invalid: {}".format(path))
    if records[0].get("spec_sha256") != spec_sha256:
        raise FileExistsError("existing audit belongs to another specification: {}".format(path))
    result = dict(records[0])
    result["idempotent"] = True
    return result


def audit(manifest_path, source_root, output_root):
    prepared = budgeted._validate_manifest(manifest_path, source_root, output_root)
    return audit_prepared(prepared, SCHEMA_VERSION)


def audit_prepared(prepared, audit_schema_version):
    if not prepared["summary_path"].is_file():
        raise FileNotFoundError("worker summary not found: {}".format(prepared["summary_path"]))
    records = reachable._read_jsonl(prepared["summary_path"])
    _require(len(records) == 1, "worker summary must contain one record")
    completion = records[0]
    _require(
        completion.get("schema_version") == reachable.SCHEMA_VERSION
        and completion.get("record_type") == "summary"
        and completion.get("complete") is True,
        "worker summary is invalid",
    )
    worker_job_hash = completion.get("job_hash")
    _require(
        completion.get("manifest_sha256") == prepared["manifest_sha256"]
        and completion.get("input", {}).get("sha256") == prepared["input_sha256"]
        and completion.get("input", {}).get("source_directory_unchanged") is True,
        "worker summary provenance is invalid",
    )
    provenance = completion.get("provenance", {})
    runtime = provenance.get("runtime")
    source = provenance.get("source")
    _require(isinstance(runtime, dict) and isinstance(source, dict), "worker provenance is missing")
    candidate_generator_sha256 = labels._sha256_file(Path(reachable.__file__))
    scheduler_sha256 = prepared.get(
        "scheduler_sha256", labels._sha256_file(Path(budgeted.__file__))
    )
    _require(
        provenance.get("candidate_generator_sha256") == candidate_generator_sha256
        and provenance.get("scheduler_sha256") == scheduler_sha256,
        "worker implementation provenance is invalid",
    )
    current_source = labels._source_provenance(Path(__file__).resolve().parents[1])
    _require(
        current_source.get("sha256") == source.get("sha256"),
        "worker core source differs from the audit checkout",
    )

    cache_manifest = sdf_cache_tools.validate_cache(
        prepared["cache_dir"],
        expected_input_sha256=prepared["input_sha256"],
        expected_parts=prepared["part_ids"],
        expected_simulator_sha256=runtime.get("simulator", {}).get("sha256"),
        expected_sim_string_sha256=labels._sha256_file(
            Path(__file__).resolve().parents[1] / "plan_sequence" / "sim_string.py"
        ),
    )
    cache_sha256 = labels._sha256_file(prepared["cache_dir"] / "manifest.json")
    prepared["cache_manifest_sha256"] = cache_sha256
    prepared["scheduler_sha256"] = scheduler_sha256
    prepared["expected_source"] = source
    prepared["expected_runtime"] = runtime
    prepared["expected_generator_sha256"] = candidate_generator_sha256
    _require(
        completion.get("sdf_cache", {}).get("manifest_sha256") == cache_sha256
        and completion.get("sdf_cache", {}).get("cache_key") == cache_manifest["cache_key"]
        and completion.get("sdf_cache", {}).get("files_unchanged") is True,
        "worker SDF cache provenance is invalid",
    )
    _require(
        worker_job_hash == budgeted._worker_job_hash(prepared, cache_sha256, source, runtime),
        "worker job hash is invalid",
    )

    timings = completion.get("timings_s")
    timing_fields = {
        "preflight_wall_s",
        "one_time_setup_wall_s",
        "copy_inputs_wall_s",
        "copy_sdf_cache_wall_s",
        "traversal_wall_s",
        "stable_pose_search_wall_s",
        "candidate_dataset_totals",
        "total_wall_s",
    }
    _require(
        isinstance(timings, dict)
        and set(timings) == timing_fields
        and all(
            not isinstance(timings[field], bool)
            and isinstance(timings[field], (int, float))
            and math.isfinite(timings[field])
            and timings[field] >= 0
            for field in timing_fields - {"candidate_dataset_totals"}
        ),
        "worker timings are invalid",
    )

    root = tuple(prepared["part_ids"])
    # State/lane selection is independent of heur-out ordering. The worker's
    # implementation hash pins the ranker, while its recorded action avoids
    # re-running floating-point mesh ranking on a potentially different host.
    state_ranker = lambda state: state
    pose_generator = pose_cache_tools._generator_identity(
        Path(__file__).resolve().parents[1], runtime
    )
    discovered = {root}
    attempted = {}
    spine_state = root
    balanced_step = 0
    dfs_path = [root]
    states = []
    terminals = []
    transitions = []
    pose_files = []
    candidate_files = []
    replayed_schedule = []
    accounting = {
        "candidate_count": 0,
        "candidate_actions": 0,
        "removability_batches": 0,
        "removability_rollouts": 0,
        "stability_queries": 0,
        "stability_rollouts": 0,
        "oracle_rollouts_total": 0,
    }
    label_counts = {name: labels._label_counts() for name in ("R", "T", "Y")}
    candidate_timing_s = {
        "removability_wall_s": 0.0,
        "stability_wall_s": 0.0,
        "candidate_wall_s": 0.0,
    }
    stable_pose_search_wall_s = 0.0
    censored_action_count = 0
    censored_stability_count = 0
    observed_ranks = {}

    schedule = completion.get("schedule")
    _require(isinstance(schedule, list), "worker schedule is missing")
    _require(
        len(schedule) <= prepared["parameters"]["state_part_budget"],
        "state-part attempt budget exceeded",
    )
    for attempt_index, observed_step in enumerate(schedule):
        candidate_time_budget_s = prepared["parameters"].get(
            "candidate_time_budget_s"
        )
        _require(
            candidate_time_budget_s is None
            or candidate_timing_s["candidate_wall_s"] < candidate_time_budget_s,
            "schedule continued after candidate-time budget",
        )
        if prepared["search_mode"] == "adaptive":
            selected = mixed_search.select_next_adaptive(
                root,
                discovered,
                attempted,
                spine_state,
                balanced_step,
                state_ranker,
                prepared["parameters"]["seed"],
            )
        elif prepared["search_mode"] == "mixed":
            selected = mixed_search.select_next(
                root, discovered, attempted, spine_state, balanced_step, state_ranker
            )
        else:
            selected = dfs_search.select_next(dfs_path, attempted, state_ranker)
        _require(selected is not None, "schedule continued after frontier exhaustion")
        if prepared["search_mode"] == "adaptive":
            (
                lane,
                parts,
                selected_part,
                _,
                sampling_policy,
                selection_probability,
            ) = selected
        else:
            lane, parts, selected_part, _ = selected
            sampling_policy = "heur-out"
            selection_probability = 1.0
        _require(isinstance(observed_step, dict), "schedule step must be an object")
        _require(
            observed_step.get("attempt_index") == attempt_index
            and observed_step.get("lane") == lane
            and observed_step.get("state_id") == labels._state_id(parts)
            and observed_step.get("parts") == list(parts)
            and observed_step.get("depth") == len(root) - len(parts),
            "schedule state/lane replay differs at attempt {}".format(attempt_index),
        )
        part_move = observed_step.get("part_move")
        heuristic_rank = observed_step.get("heuristic_rank")
        _require(
            part_move in parts and part_move not in attempted.get(parts, ()),
            "schedule selected an invalid or repeated part at attempt {}".format(
                attempt_index
            ),
        )
        _require(
            type(heuristic_rank) is int and 0 <= heuristic_rank < len(parts),
            "schedule heuristic rank is invalid at attempt {}".format(attempt_index),
        )
        ranks_for_state = observed_ranks.setdefault(parts, {})
        _require(
            heuristic_rank not in ranks_for_state
            or ranks_for_state[heuristic_rank] == part_move,
            "schedule assigns one heuristic rank to multiple parts",
        )
        ranks_for_state[heuristic_rank] = part_move
        if prepared["search_mode"] == "adaptive" and lane == "breadth":
            _require(
                part_move == selected_part,
                "seeded breadth selection differs at attempt {}".format(attempt_index),
            )
        if prepared["search_mode"] in {"mixed", "adaptive"} and lane != "spine":
            balanced_step += 1
        first_visit = parts not in attempted
        if first_visit:
            attempted[parts] = []
            states.append({"state_id": labels._state_id(parts), "parts": list(parts)})

        pose_path = reachable._state_dir(prepared, parts) / "poses.json"
        _require(pose_path.is_file(), "state pose checkpoint is missing: {}".format(pose_path))
        pose_identity = reachable._pose_identity(prepared, parts, pose_generator)
        pose_checkpoint = reachable._validate_pose_record(pose_path, pose_identity)
        pose_sha256 = labels._sha256_file(pose_path)
        if first_visit:
            stable_pose_search_wall_s += pose_checkpoint["stable_pose_search_wall_s"]
            pose_files.append(
                {
                    "state_id": labels._state_id(parts),
                    "path": pose_path.relative_to(prepared["labels_root"]).as_posix(),
                    "sha256": pose_sha256,
                }
            )

        candidate_before = accounting["candidate_count"]
        feasible = False
        evaluated_pose_count = 0
        part_attempt_complete = True
        pose_records = list(pose_checkpoint["poses"])
        if prepared.get("seed_pose_order"):
            pose_records.sort(
                key=lambda pose: mixed_search.seeded_tie(
                    prepared["parameters"]["seed"],
                    (prepared["assembly_dir"].name,) + parts,
                    "{}:{}".format(part_move, pose["pose_id"]),
                )
            )
        time_budget_hit = False
        for pose_position, pose in enumerate(pose_records):
            if accounting["candidate_count"] >= prepared["parameters"]["candidate_budget"]:
                part_attempt_complete = False
                break
            path, candidate_records = candidate_audit._validate_candidate(
                prepared, parts, pose, pose_sha256, part_move, worker_job_hash
            )
            candidate_summary = candidate_records[-1]
            accounting["candidate_count"] += 1
            evaluated_pose_count += 1
            for field in accounting:
                if field != "candidate_count":
                    accounting[field] += candidate_summary["accounting"].get(field, 0)
            reachable._add_counts(label_counts, candidate_summary["label_counts"])
            for field in candidate_timing_s:
                candidate_timing_s[field] += candidate_summary["timings_s"][field]
            feasible = feasible or any(
                action.get("Y") is True for action in candidate_records[2:8]
            )
            censored_action_count += sum(
                action.get("Y") is None for action in candidate_records[2:8]
            )
            stability = candidate_records[1]
            censored_stability_count += int(
                stability.get("queried") is True and stability.get("T") is None
            )
            candidate_files.append(
                {
                    "state_id": labels._state_id(parts),
                    "pose_id": pose["pose_id"],
                    "part_move": part_move,
                    "path": path.relative_to(prepared["labels_root"]).as_posix(),
                    "sha256": labels._sha256_file(path),
                }
            )
            candidate_time_budget_s = prepared["parameters"].get(
                "candidate_time_budget_s"
            )
            time_budget_hit = (
                candidate_time_budget_s is not None
                and candidate_timing_s["candidate_wall_s"]
                >= candidate_time_budget_s
            )
            if prepared.get(
                "stop_on_first_feasible_pose",
                prepared["search_mode"] == "dfs",
            ) and feasible:
                break
            if time_budget_hit:
                if pose_position + 1 < len(pose_records):
                    part_attempt_complete = False
                break

        if part_attempt_complete:
            attempted[parts].append(part_move)
        child = tuple(part for part in parts if part != part_move)
        if feasible:
            transitions.append(
                {
                    "parent_state_id": labels._state_id(parts),
                    "child_state_id": labels._state_id(child),
                    "part_move": part_move,
                }
            )
            if child not in discovered:
                discovered.add(child)
                if len(child) == 1:
                    terminals.append({"state_id": labels._state_id(child), "parts": list(child)})
            if prepared["search_mode"] in {"mixed", "adaptive"} and lane == "spine":
                spine_state = child if len(child) > 1 else None
            elif prepared["search_mode"] == "dfs" and len(child) > 1:
                dfs_path.append(child)
        elif lane == "spine" and len(attempted[parts]) == len(parts):
            spine_state = None

        expected_step = {
            "attempt_index": attempt_index,
            "lane": lane,
            "state_id": labels._state_id(parts),
            "parts": list(parts),
            "depth": len(root) - len(parts),
            "part_move": part_move,
            "heuristic_rank": heuristic_rank,
            **(
                {
                    "sampling_policy": sampling_policy,
                    "selection_probability": selection_probability,
                    "pose_order": [pose["pose_id"] for pose in pose_records],
                    "candidate_time_after_s": candidate_timing_s[
                        "candidate_wall_s"
                    ],
                }
                if prepared["search_mode"] == "adaptive"
                else {}
            ),
            "pose_count": len(pose_checkpoint["poses"]),
            "evaluated_pose_count": evaluated_pose_count,
            "part_attempt_complete": part_attempt_complete,
            "candidate_evaluations_before": candidate_before,
            "candidate_evaluations_after": accounting["candidate_count"],
            "feasible": feasible,
            "child_state_id": labels._state_id(child) if feasible else None,
        }
        _require(observed_step == expected_step, "schedule replay differs at attempt {}".format(attempt_index))
        replayed_schedule.append(expected_step)

    budget = completion.get("budget")
    if prepared["search_mode"] == "dfs" and terminals:
        expected_stop = "solution_found"
    elif (
        prepared["parameters"].get("candidate_time_budget_s") is not None
        and candidate_timing_s["candidate_wall_s"]
        >= prepared["parameters"]["candidate_time_budget_s"]
    ):
        expected_stop = "candidate_time_budget_reached"
    elif (
        len(schedule) == prepared["parameters"]["state_part_budget"]
        or accounting["candidate_count"] == prepared["parameters"]["candidate_budget"]
    ):
        expected_stop = "budget_reached"
    else:
        expected_stop = "frontier_exhausted"
    if expected_stop == "frontier_exhausted":
        if prepared["search_mode"] == "adaptive":
            remaining = mixed_search.select_next_adaptive(
                root,
                discovered,
                attempted,
                spine_state,
                balanced_step,
                state_ranker,
                prepared["parameters"]["seed"],
            )
        elif prepared["search_mode"] == "mixed":
            remaining = mixed_search.select_next(
                root, discovered, attempted, spine_state, balanced_step, state_ranker
            )
        else:
            remaining = dfs_search.select_next(dfs_path, attempted, state_ranker)
        _require(remaining is None, "worker stopped before budget with candidates remaining")
    expected_budget = {
        "state_part_attempt_budget": prepared["parameters"]["state_part_budget"],
        "candidate_evaluation_budget": prepared["parameters"]["candidate_budget"],
        "stop_reason": expected_stop,
    }
    if prepared["parameters"].get("candidate_time_budget_s") is not None:
        expected_budget["candidate_time_budget_s"] = prepared["parameters"][
            "candidate_time_budget_s"
        ]
    _require(budget == expected_budget, "worker budget record is invalid")
    _require(
        accounting["candidate_count"] <= prepared["parameters"]["candidate_budget"],
        "candidate evaluation budget exceeded",
    )

    accounting["stability_queries_avoided"] = (
        accounting["candidate_count"] - accounting["stability_queries"]
    )
    accounting["state_count"] = len(discovered)
    accounting["evaluated_state_count"] = len(states)
    accounting["terminal_state_count"] = len(terminals)
    accounting["transition_edge_count"] = len(transitions)
    accounting["state_part_attempt_count"] = len(schedule)
    coverage = budgeted._coverage(root, discovered, attempted, transitions, replayed_schedule)

    expected_state_dirs = {state["state_id"] for state in states}
    actual_state_dirs = {
        path.name for path in prepared["labels_root"].glob("state-*") if path.is_dir()
    }
    _require(
        expected_state_dirs <= actual_state_dirs,
        "one or more referenced state checkpoints are missing",
    )
    orphan_state_dirs = sorted(actual_state_dirs - expected_state_dirs)
    _require(completion.get("accounting") == accounting, "worker aggregate accounting differs")
    _require(completion.get("label_counts") == label_counts, "worker label counts differ")
    _require(completion.get("coverage") == coverage, "worker coverage summary differs")
    _require(completion.get("states") == states, "worker evaluated-state list differs")
    _require(completion.get("terminal_states") == terminals, "worker terminal-state list differs")
    _require(completion.get("transitions") == transitions, "worker transition list differs")
    _require(completion.get("state_pose_files") == pose_files, "worker pose file list differs")
    _require(completion.get("candidate_files") == candidate_files, "worker candidate file list differs")
    _require(
        timings.get("stable_pose_search_wall_s") == stable_pose_search_wall_s
        and timings.get("candidate_dataset_totals") == candidate_timing_s,
        "worker dataset timing totals differ",
    )
    current_run = completion.get("current_run")
    _require(
        isinstance(current_run, dict)
        and current_run.get("state_pose_checkpoints_created", 0)
        + current_run.get("state_pose_checkpoints_reused", 0)
        == len(states)
        and current_run.get("candidate_checkpoints_created", 0)
        + current_run.get("candidate_checkpoints_reused", 0)
        == accounting["candidate_count"],
        "worker resume accounting differs",
    )

    spec = {
        "schema_version": audit_schema_version,
        "manifest_sha256": prepared["manifest_sha256"],
        "completion_sha256": labels._sha256_file(prepared["summary_path"]),
        "candidate_sha256": [entry["sha256"] for entry in candidate_files],
        "pose_sha256": [entry["sha256"] for entry in pose_files],
        "auditor_sha256": labels._sha256_file(Path(__file__)),
    }
    spec_sha256 = labels._sha256_bytes(labels._canonical_json(spec).encode("utf-8"))
    existing = _existing_report(
        prepared["audit_output"], spec_sha256, audit_schema_version
    )
    if existing is not None:
        return existing
    if prepared.get("audit_policy") == "adaptive":
        policy_checks = {
            "state_part_budget_respected": (
                len(schedule) <= prepared["parameters"]["state_part_budget"]
            ),
            "candidate_budget_respected": (
                accounting["candidate_count"]
                <= prepared["parameters"]["candidate_budget"]
            ),
            "time_stop_valid": (
                expected_stop != "candidate_time_budget_reached"
                or candidate_timing_s["candidate_wall_s"]
                >= prepared["parameters"]["candidate_time_budget_s"]
            ),
        }
    else:
        policy_checks = {
            "complete_sequence_found": coverage["complete_sequence_found"],
            "candidate_budget_respected": (
                accounting["candidate_count"]
                <= prepared["parameters"]["candidate_budget"]
            ),
            "no_censored_actions": censored_action_count == 0,
            "no_censored_queried_stability": censored_stability_count == 0,
        }
    if prepared["search_mode"] in {"mixed", "adaptive"} and prepared.get(
        "audit_policy"
    ) != "adaptive":
        policy_checks.update(
            {
                "breadth_sampled": coverage["lane_counts"]["breadth"] > 0,
                "depth_sampled": coverage["lane_counts"]["depth"] > 0,
                "non_top_rank_sampled": (
                    coverage["heuristic_rank_counts"]["non_top"] > 0
                ),
            }
        )
    elif prepared["search_mode"] == "dfs":
        policy_checks["dfs_only"] = (
            coverage["lane_counts"]["dfs"] == len(schedule)
        )
    policy_checks["no_orphan_state_checkpoints"] = not orphan_state_dirs
    report = {
        "schema_version": audit_schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": spec_sha256,
        "status": "passed" if all(policy_checks.values()) else "review_required",
        "integrity_status": "passed",
        "phase": prepared["manifest"]["phase"],
        "assembly_id": prepared["assembly_dir"].name,
        "part_count": len(prepared["part_ids"]),
        "search_mode": prepared["search_mode"],
        "manifest_sha256": prepared["manifest_sha256"],
        "completion_sha256": spec["completion_sha256"],
        "reachability_scope": "budgeted oracle-observed Y=true transitions; null actions are censored",
        "budget": budget,
        "observed": accounting,
        "coverage": coverage,
        "policy_checks": policy_checks,
        "label_counts": label_counts,
        "censoring": {
            "action_count": censored_action_count,
            "queried_stability_count": censored_stability_count,
        },
        "state_pose_file_count": len(pose_files),
        "candidate_file_count": len(candidate_files),
        "orphan_state_checkpoints": orphan_state_dirs,
        "terminal_states": terminals,
    }
    if prepared.get("complexity") is not None:
        expected_complexity = dict(
            prepared["complexity"], sdf_cache_bytes=cache_manifest["bytes"]
        )
        _require(
            completion.get("complexity") == expected_complexity,
            "worker complexity record differs",
        )
        report["complexity"] = expected_complexity
        report["outcomes"] = {
            "complete_sequence_found": coverage["complete_sequence_found"],
            "censored_action_count": censored_action_count,
            "censored_queried_stability_count": censored_stability_count,
        }
    try:
        labels._write_atomic(prepared["audit_output"], [report])
    except FileExistsError:
        return _existing_report(
            prepared["audit_output"], spec_sha256, audit_schema_version
        )
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report = audit(args.manifest, args.source_root, args.output_root)
    print(labels._canonical_json(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
