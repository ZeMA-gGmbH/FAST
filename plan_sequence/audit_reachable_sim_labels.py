"""Audit reachable-state labels and their equivalence with the cached-pose root reference."""

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from plan_sequence import generate_reachable_sim_labels as reachable
from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_pose_cache as pose_cache_tools
from plan_sequence import prepare_sdf_cache as sdf_cache_tools


SCHEMA_VERSION = "fast.reachable-label-audit.v1"
POSE_PROBABILITY_ABS_TOLERANCE = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_probability(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=POSE_PROBABILITY_ABS_TOLERANCE,
    )


def _count(counts, value):
    counts["true" if value is True else "false" if value is False else "null"] += 1


def _read_candidate(path):
    records = reachable._read_jsonl(path)
    _require(len(records) == 9, "candidate must contain 9 records: {}".format(path))
    _require(
        [record.get("record_type") for record in records]
        == ["header", "stability"] + ["action"] * 6 + ["summary"],
        "candidate record order is invalid: {}".format(path),
    )
    _require(
        all(record.get("schema_version") == reachable.SCHEMA_VERSION for record in records),
        "candidate schema is invalid: {}".format(path),
    )
    job_hash = records[0].get("job_hash")
    _require(
        all(record.get("job_hash") == job_hash for record in records),
        "candidate job hashes disagree: {}".format(path),
    )
    _require(
        records[-1].get("complete") is True
        and records[-1].get("record_count") == len(records),
        "candidate is incomplete: {}".format(path),
    )
    return records


def _semantic_mismatches(records, reference, location):
    mismatches = []
    header, stability = records[0], records[1]
    reference_header, reference_stability = reference[0], reference[1]
    for field in ("part_move", "pose_id", "pose_index", "pose", "parts", "parts_after"):
        if header.get(field) != reference_header.get(field):
            mismatches.append("{}: root {} differs from the cached-pose root reference".format(location, field))
    if not _same_probability(
        header.get("pose_probability"), reference_header.get("pose_probability")
    ):
        mismatches.append("{}: root pose probability differs from the cached-pose root reference".format(location))
    for field in ("queried", "T", "T_status"):
        if stability.get(field) != reference_stability.get(field):
            mismatches.append("{}: root stability {} differs from the cached-pose root reference".format(location, field))
    for action, reference_action in zip(records[2:8], reference[2:8]):
        if action.get("direction_id") != reference_action.get("direction_id"):
            mismatches.append("{}: root action order differs from the cached-pose root reference".format(location))
            continue
        for field in ("direction", "R", "R_status", "T", "T_status", "Y", "Y_status"):
            if action.get(field) != reference_action.get(field):
                mismatches.append(
                    "{} {}: root {} differs from the cached-pose root reference".format(
                        location, action.get("direction_id"), field
                    )
                )
    return mismatches


def _validate_candidate(
    prepared,
    parts,
    pose,
    pose_sha256,
    part_move,
    worker_job_hash,
):
    path = reachable._candidate_path(prepared, parts, pose["index"], part_move)
    if not path.is_file():
        raise FileNotFoundError("candidate checkpoint not found: {}".format(path))
    records = _read_candidate(path)
    header, stability, summary = records[0], records[1], records[-1]
    expected_hash = reachable._candidate_hash(
        worker_job_hash, pose_sha256, parts, pose["index"], part_move
    )
    _require(header.get("job_hash") == expected_hash, "candidate hash is invalid: {}".format(path))
    _require(
        header.get("worker_job_hash") == worker_job_hash,
        "worker hash is invalid: {}".format(path),
    )
    _require(
        header.get("state_id") == labels._state_id(parts)
        and header.get("parts") == list(parts)
        and header.get("part_move") == part_move
        and header.get("parts_after") == [part for part in parts if part != part_move],
        "candidate state identity is invalid: {}".format(path),
    )
    _require(
        header.get("pose_id") == pose["pose_id"]
        and header.get("pose_index") == pose["index"]
        and header.get("pose") == pose["pose"]
        and header.get("pose_probability") == pose["probability"],
        "candidate pose identity is invalid: {}".format(path),
    )
    _require(
        header.get("manifest", {}).get("sha256") == prepared["manifest_sha256"]
        and header.get("input", {}).get("sha256") == prepared["input_sha256"],
        "candidate input provenance is invalid: {}".format(path),
    )
    _require(
        header.get("sdf_cache", {}).get("manifest_sha256")
        == prepared["cache_manifest_sha256"]
        and header.get("sdf_cache", {}).get("files_unchanged") is True,
        "candidate SDF cache provenance is invalid: {}".format(path),
    )
    _require(
        header.get("state_pose_checkpoint", {}).get("manifest_sha256") == pose_sha256,
        "candidate pose checkpoint hash is invalid: {}".format(path),
    )
    _require(
        header.get("provenance", {}).get("source") == prepared["expected_source"]
        and header.get("provenance", {}).get("runtime") == prepared["expected_runtime"]
        and header.get("provenance", {}).get("generator_sha256")
        == prepared["expected_generator_sha256"],
        "candidate worker provenance is invalid: {}".format(path),
    )
    actions = records[2:8]
    _require(
        [(action.get("direction_id"), tuple(action.get("direction", []))) for action in actions]
        == list(labels.A6),
        "candidate directions are invalid: {}".format(path),
    )
    has_removal = any(action.get("R") is True for action in actions)
    _require(
        stability.get("queried") is has_removal,
        "conditional stability policy is invalid: {}".format(path),
    )
    if has_removal:
        _require(
            stability.get("T_status") != "not_evaluated",
            "queried stability is not evaluated: {}".format(path),
        )
    else:
        _require(
            stability.get("T") is None
            and stability.get("T_status") == "not_evaluated"
            and stability.get("oracle_rollout_count") == 0,
            "skipped stability is invalid: {}".format(path),
        )
    for action in actions:
        _require(
            action.get("stability_ref") == stability.get("stability_id")
            and action.get("T_queried") is has_removal
            and action.get("T") == stability.get("T")
            and action.get("T_status") == stability.get("T_status"),
            "candidate stability reference is invalid: {}".format(path),
        )
        expected_y, expected_status = labels._combined_label(
            action.get("R"),
            action.get("T"),
            action.get("R_status"),
            action.get("T_status"),
        )
        _require(
            action.get("Y") is expected_y and action.get("Y_status") == expected_status,
            "candidate R/T/Y algebra is invalid: {}".format(path),
        )
    expected_counts = {name: labels._label_counts() for name in ("R", "T", "Y")}
    _count(expected_counts["T"], stability.get("T"))
    for action in actions:
        _count(expected_counts["R"], action.get("R"))
        _count(expected_counts["Y"], action.get("Y"))
    expected_accounting = {
        "candidate_actions": len(labels.A6),
        "removability_batches": 1,
        "removability_rollouts": sum(
            action.get("oracle_rollout_count", 0) for action in actions
        ),
        "stability_queries": 1 if has_removal else 0,
        "stability_rollouts": stability.get("oracle_rollout_count", 0),
    }
    expected_accounting["oracle_rollouts_total"] = (
        expected_accounting["removability_rollouts"]
        + expected_accounting["stability_rollouts"]
    )
    _require(
        summary.get("accounting") == expected_accounting,
        "candidate accounting is invalid: {}".format(path),
    )
    _require(
        summary.get("label_counts") == expected_counts,
        "candidate label counts are invalid: {}".format(path),
    )
    timings = summary.get("timings_s")
    _require(
        isinstance(timings, dict)
        and set(timings)
        == {"removability_wall_s", "stability_wall_s", "candidate_wall_s"}
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
            for value in timings.values()
        ),
        "candidate timings are invalid: {}".format(path),
    )
    return path, records


def _existing_report(path, spec_sha256):
    if not path.exists():
        return None
    records = reachable._read_jsonl(path)
    if len(records) != 1 or records[0].get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("existing audit is invalid: {}".format(path))
    if records[0].get("spec_sha256") != spec_sha256:
        raise FileExistsError("existing audit belongs to another specification: {}".format(path))
    result = dict(records[0])
    result["idempotent"] = True
    return result


def audit(manifest_path, source_root, output_root, *, check_root_reference=True):
    prepared = reachable._validate_manifest(manifest_path, source_root, output_root)
    if not prepared["summary_path"].is_file():
        raise FileNotFoundError("worker summary not found: {}".format(prepared["summary_path"]))
    completion_records = reachable._read_jsonl(prepared["summary_path"])
    _require(len(completion_records) == 1, "worker summary must contain one record")
    completion = completion_records[0]
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
    runtime = completion.get("provenance", {}).get("runtime")
    source = completion.get("provenance", {}).get("source")
    _require(isinstance(runtime, dict) and isinstance(source, dict), "worker provenance is missing")
    cache_manifest = sdf_cache_tools.validate_cache(
        prepared["cache_dir"],
        expected_input_sha256=prepared["input_sha256"],
        expected_parts=prepared["part_ids"],
        expected_simulator_sha256=runtime.get("simulator", {}).get("sha256"),
        expected_sim_string_sha256=labels._sha256_file(
            Path(__file__).resolve().parents[1] / "plan_sequence" / "sim_string.py"
        ),
    )
    cache_manifest_sha256 = labels._sha256_file(prepared["cache_dir"] / "manifest.json")
    prepared["cache_manifest_sha256"] = cache_manifest_sha256
    _require(
        completion.get("sdf_cache", {}).get("manifest_sha256") == cache_manifest_sha256
        and completion.get("sdf_cache", {}).get("cache_key") == cache_manifest["cache_key"]
        and completion.get("sdf_cache", {}).get("files_unchanged") is True,
        "worker SDF cache provenance is invalid",
    )
    _require(
        worker_job_hash
        == reachable._worker_job_hash(prepared, cache_manifest_sha256, source, runtime),
        "worker job hash is invalid",
    )
    prepared["expected_source"] = source
    prepared["expected_runtime"] = runtime
    prepared["expected_generator_sha256"] = labels._sha256_file(
        Path(__file__).resolve().parent / "generate_reachable_sim_labels.py"
    )
    _require(
        completion.get("provenance", {}).get("generator_sha256")
        == prepared["expected_generator_sha256"],
        "worker generator provenance is invalid",
    )
    current_source = labels._source_provenance(Path(__file__).resolve().parents[1])
    _require(
        current_source.get("sha256") == source.get("sha256"),
        "worker core source differs from the audit checkout",
    )
    timings = completion.get("timings_s")
    _require(
        isinstance(timings, dict)
        and {
            "preflight_wall_s",
            "one_time_setup_wall_s",
            "copy_inputs_wall_s",
            "copy_sdf_cache_wall_s",
            "traversal_wall_s",
            "stable_pose_search_wall_s",
            "candidate_dataset_totals",
            "total_wall_s",
        }
        == set(timings)
        and all(
            not isinstance(timings[field], bool)
            and isinstance(timings[field], (int, float))
            and math.isfinite(timings[field])
            and timings[field] >= 0
            for field in set(timings) - {"candidate_dataset_totals"}
        ),
        "worker timings are invalid",
    )
    pose_generator = pose_cache_tools._generator_identity(
        Path(__file__).resolve().parents[1], runtime
    )

    queue = deque([tuple(prepared["part_ids"])])
    discovered = {tuple(prepared["part_ids"])}
    evaluated = []
    terminals = []
    edges = []
    candidate_files = []
    pose_files = []
    root_files = []
    root_pose_count = None
    reference_files = []
    mismatches = []
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

    while queue:
        parts = queue.popleft()
        state_id = labels._state_id(parts)
        if len(parts) == 1:
            terminals.append({"state_id": state_id, "parts": list(parts)})
            continue
        evaluated.append({"state_id": state_id, "parts": list(parts)})
        pose_path = reachable._state_dir(prepared, parts) / "poses.json"
        if not pose_path.is_file():
            raise FileNotFoundError("state pose checkpoint not found: {}".format(pose_path))
        pose_identity = reachable._pose_identity(prepared, parts, pose_generator)
        pose_checkpoint = reachable._validate_pose_record(pose_path, pose_identity)
        if parts == tuple(prepared["part_ids"]):
            root_pose_count = len(pose_checkpoint["poses"])
        stable_pose_search_wall_s += pose_checkpoint["stable_pose_search_wall_s"]
        pose_sha256 = labels._sha256_file(pose_path)
        pose_files.append(
            {
                "state_id": state_id,
                "path": pose_path.relative_to(prepared["labels_root"]).as_posix(),
                "sha256": pose_sha256,
            }
        )
        feasible_parts = set()
        for pose in pose_checkpoint["poses"]:
            for part_move in parts:
                path, records = _validate_candidate(
                    prepared,
                    parts,
                    pose,
                    pose_sha256,
                    part_move,
                    worker_job_hash,
                )
                summary = records[-1]
                accounting["candidate_count"] += 1
                for field in accounting:
                    if field != "candidate_count":
                        accounting[field] += summary["accounting"].get(field, 0)
                for name in label_counts:
                    for value in label_counts[name]:
                        label_counts[name][value] += summary["label_counts"][name][value]
                for field in candidate_timing_s:
                    candidate_timing_s[field] += summary["timings_s"][field]
                entry = {
                    "state_id": state_id,
                    "pose_id": pose["pose_id"],
                    "part_move": part_move,
                    "path": path.relative_to(prepared["labels_root"]).as_posix(),
                    "sha256": labels._sha256_file(path),
                }
                candidate_files.append(entry)
                if any(action.get("Y") is True for action in records[2:8]):
                    feasible_parts.add(part_move)
                if parts == tuple(prepared["part_ids"]):
                    root_files.append(entry)

                if parts == tuple(prepared["part_ids"]) and check_root_reference:
                    reference_path = (
                        prepared["reference_root"]
                        / "pose-{:03d}".format(pose["index"])
                        / "part-{}.jsonl".format(part_move)
                    )
                    if not reference_path.is_file():
                        raise FileNotFoundError(
                            "cached-pose root reference not found: {}".format(reference_path)
                        )
                    reference = reachable._read_jsonl(reference_path)
                    _require(len(reference) == 9, "cached-pose root reference is invalid")
                    _require(
                        [record.get("record_type") for record in reference]
                        == ["header", "stability"] + ["action"] * 6 + ["summary"],
                        "cached-pose root reference order is invalid",
                    )
                    location = "{}/{}".format(pose["pose_id"], part_move)
                    mismatches.extend(_semantic_mismatches(records, reference, location))
                    reference_files.append(
                        {"path": str(reference_path), "sha256": labels._sha256_file(reference_path)}
                    )

        for part_move in parts:
            if part_move not in feasible_parts:
                continue
            child = tuple(part for part in parts if part != part_move)
            edges.append(
                {
                    "parent_state_id": state_id,
                    "child_state_id": labels._state_id(child),
                    "part_move": part_move,
                }
            )
            if child not in discovered:
                discovered.add(child)
                _require(
                    len(discovered) <= prepared["parameters"]["max_states"],
                    "reachable-state ceiling exceeded",
                )
                queue.append(child)

    expected_state_dirs = {state["state_id"] for state in evaluated}
    actual_state_dirs = {
        path.name
        for path in prepared["labels_root"].glob("state-*")
        if path.is_dir()
    }
    _require(actual_state_dirs == expected_state_dirs, "state checkpoint coverage is not exact")

    accounting["stability_queries_avoided"] = (
        accounting["candidate_count"] - accounting["stability_queries"]
    )
    accounting["state_count"] = len(discovered)
    accounting["evaluated_state_count"] = len(evaluated)
    accounting["terminal_state_count"] = len(terminals)
    accounting["transition_edge_count"] = len(edges)
    _require(completion.get("accounting") == accounting, "worker aggregate accounting differs")
    _require(completion.get("label_counts") == label_counts, "worker label counts differ")
    _require(completion.get("states") == evaluated, "worker evaluated-state list differs")
    _require(completion.get("terminal_states") == terminals, "worker terminal-state list differs")
    _require(completion.get("transitions") == edges, "worker transition list differs")
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
        == len(evaluated)
        and current_run.get("candidate_checkpoints_created", 0)
        + current_run.get("candidate_checkpoints_reused", 0)
        == accounting["candidate_count"],
        "worker resume accounting differs",
    )
    _require(root_pose_count is not None, "root pose checkpoint was not audited")
    expected_root_candidates = root_pose_count * len(prepared["part_ids"])
    expected_root_actions = expected_root_candidates * len(labels.A6)
    _require(
        len(root_files) == expected_root_candidates,
        "root candidate coverage is incomplete for the recorded poses",
    )
    _require(
        len(root_files) <= prepared["manifest"]["expected"]["root_candidate_count"],
        "root candidate coverage exceeds the configured maximum",
    )
    _require(
        len(root_files) * len(labels.A6) == expected_root_actions,
        "root action coverage is incomplete for the recorded poses",
    )
    _require(
        expected_root_actions
        <= prepared["manifest"]["expected"]["root_action_count"],
        "root action coverage exceeds the configured maximum",
    )

    spec = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": prepared["manifest_sha256"],
        "completion_sha256": labels._sha256_file(prepared["summary_path"]),
        "candidate_sha256": [entry["sha256"] for entry in candidate_files],
        "pose_sha256": [entry["sha256"] for entry in pose_files],
        "reference_sha256": [entry["sha256"] for entry in reference_files],
        "auditor_sha256": labels._sha256_file(Path(__file__)),
    }
    spec_sha256 = labels._sha256_bytes(labels._canonical_json(spec).encode("utf-8"))
    existing = _existing_report(prepared["audit_output"], spec_sha256)
    if existing is not None:
        return existing
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": spec_sha256,
        "status": "passed" if not mismatches else "review_required",
        "integrity_status": "passed",
        "phase": prepared["manifest"]["phase"],
        "manifest_sha256": prepared["manifest_sha256"],
        "completion_sha256": spec["completion_sha256"],
        "reachability_scope": "oracle-observed Y=true transitions; null actions are censored",
        "pose_comparison": {
            "matrix": "exact",
            "probability_abs_tolerance": POSE_PROBABILITY_ABS_TOLERANCE,
            "probability_rel_tolerance": 0.0,
        },
        "observed": accounting,
        "label_counts": label_counts,
        "root_reference": {
            "candidate_count": len(root_files),
            "action_count": len(root_files) * len(labels.A6),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        },
        "state_pose_file_count": len(pose_files),
        "candidate_file_count": len(candidate_files),
        "terminal_states": terminals,
    }
    if not check_root_reference:
        report["root_reference"]["checked"] = False
    try:
        labels._write_atomic(prepared["audit_output"], [report])
    except FileExistsError:
        return _existing_report(prepared["audit_output"], spec_sha256)
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
