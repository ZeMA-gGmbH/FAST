"""Generate exhaustive oracle-reachable FAST labels for one assembly.

One process owns the assembly, performs deterministic breadth-first traversal,
and atomically checkpoints every state pose set and candidate.  A restarted
process reconstructs the frontier from validated checkpoints.
"""

import argparse
import json
import math
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from plan_sequence import generate_conditional_sim_labels as conditional
from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_pose_cache as pose_cache_tools
from plan_sequence import prepare_sdf_cache as sdf_cache_tools
from plan_sequence import run_sim_label_manifest as manifest_tools


SCHEMA_VERSION = "fast.reachable-sim-labels.v1"
POSE_SCHEMA_VERSION = "fast.reachable-state-poses.v1"
MANIFEST_SCHEMA_VERSION = "fast.reachable-label-manifest.v1"
_ROOT_FIELDS = {
    "schema_version",
    "label_schema_version",
    "phase",
    "protocol",
    "parameters",
    "assembly",
    "cache",
    "root_reference",
    "output_root",
    "audit_output",
    "expected",
}
_ASSEMBLY_FIELDS = {"path", "part_count", "part_ids", "input_sha256"}
_PARAMETER_FIELDS = {
    "max_poses",
    "pose_probability",
    "removal_timeout_s",
    "stability_timeout_s",
    "save_sdf",
    "max_states",
}
_EXPECTED_FIELDS = {
    "root_candidate_count",
    "root_action_count",
    "theoretical_max_nonempty_states",
}
_PROTOCOL = {
    "state_scope": "oracle_reachable_remaining_part_states",
    "traversal": "deterministic_breadth_first",
    "action_space": "A6_world",
    "direction_frame": "world",
    "gravity": [0, 0, -1],
    "max_grippers": 2,
    "max_holding_parts": 1,
    "base_part": None,
    "support": "ground_plane",
    "support_normal": [0, 0, 1],
    "stability_oracle": "ASAP-greedy",
    "removability_coverage": "all_six_directions",
    "stability_policy": "query_once_if_any_direction_is_true",
    "state_expansion": "any_pose_direction_Y_true",
    "terminal_state": "one_part_remaining",
    "candidate_isolation": "one_process_per_assembly",
    "stable_pose_policy": "compute_once_per_state_and_checkpoint",
}


def _sha256(value, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("{} must be lowercase SHA-256".format(field))
    return value


def _validate_manifest(manifest_path, source_root, output_root):
    manifest_path = Path(manifest_path).resolve()
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("manifest not found: {}".format(manifest_path))
    if not source_root.is_dir():
        raise FileNotFoundError("source root not found: {}".format(source_root))
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("output root is not a directory: {}".format(output_root))
    if manifest_tools._is_within(source_root, output_root) or manifest_tools._is_within(
        output_root, source_root
    ):
        raise ValueError("source and output roots must not contain one another")

    manifest = manifest_tools._read_manifest(manifest_path)
    manifest_tools._require_fields(manifest, _ROOT_FIELDS, "manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported reachable-state manifest schema")
    if manifest.get("label_schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest label schema does not match the worker")
    manifest_tools._component(manifest.get("phase"), "phase")
    if manifest.get("protocol") != _PROTOCOL:
        raise ValueError("manifest protocol does not match the worker")

    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    manifest_tools._require_fields(parameters, _PARAMETER_FIELDS, "parameters")
    max_poses = parameters.get("max_poses")
    if type(max_poses) is not int or not 1 <= max_poses <= 3:
        raise ValueError("parameters.max_poses must be an integer from 1 to 3")
    pose_probability = manifest_tools._positive_number(
        parameters.get("pose_probability"), "parameters.pose_probability"
    )
    if pose_probability > 1:
        raise ValueError("parameters.pose_probability must be at most 1")
    removal_timeout = manifest_tools._positive_number(
        parameters.get("removal_timeout_s"), "parameters.removal_timeout_s"
    )
    stability_timeout = manifest_tools._positive_number(
        parameters.get("stability_timeout_s"), "parameters.stability_timeout_s"
    )
    if parameters.get("save_sdf") is not True:
        raise ValueError("parameters.save_sdf must be true")
    max_states = parameters.get("max_states")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("parameters.max_states must be a positive integer")

    assembly = manifest.get("assembly")
    if not isinstance(assembly, dict):
        raise ValueError("assembly must be an object")
    manifest_tools._require_fields(assembly, _ASSEMBLY_FIELDS, "assembly")
    assembly_relative = manifest_tools._relative_path(assembly.get("path"), "assembly.path")
    assembly_dir = manifest_tools._resolve_relative(
        source_root, assembly_relative, "assembly.path"
    )
    part_ids = assembly.get("part_ids")
    part_count = assembly.get("part_count")
    if (
        type(part_count) is not int
        or part_count < 2
        or not isinstance(part_ids, list)
        or len(part_ids) != part_count
        or part_ids != sorted(part_ids)
        or len(set(part_ids)) != len(part_ids)
        or any(not isinstance(part, str) or not part for part in part_ids)
    ):
        raise ValueError("assembly part IDs are invalid")
    input_sha256 = _sha256(assembly.get("input_sha256"), "assembly.input_sha256")
    if not assembly_dir.is_dir():
        raise FileNotFoundError("assembly not found: {}".format(assembly_dir))
    if not (assembly_dir / "contact_graph.json").is_file():
        raise FileNotFoundError("contact graph not found: {}".format(assembly_dir))
    actual_parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    if actual_parts != part_ids:
        raise ValueError("assembly part IDs do not match its OBJ files")
    if labels._hash_assembly(assembly_dir)[0] != input_sha256:
        raise ValueError("assembly input SHA-256 does not match")

    cache_relative = manifest_tools._relative_path(manifest.get("cache"), "cache")
    reference_relative = manifest_tools._relative_path(
        manifest.get("root_reference"), "root_reference"
    )
    labels_relative = manifest_tools._relative_path(
        manifest.get("output_root"), "output_root"
    )
    audit_relative = manifest_tools._relative_path(
        manifest.get("audit_output"), "audit_output", ".json"
    )
    cache_dir = manifest_tools._resolve_relative(output_root, cache_relative, "cache")
    reference_root = manifest_tools._resolve_relative(
        output_root, reference_relative, "root_reference"
    )
    labels_root = manifest_tools._resolve_relative(
        output_root, labels_relative, "output_root"
    )
    audit_output = manifest_tools._resolve_relative(
        output_root, audit_relative, "audit_output"
    )
    for protected, name in ((cache_dir, "cache"), (reference_root, "root_reference")):
        if manifest_tools._is_within(protected, labels_root) or manifest_tools._is_within(
            labels_root, protected
        ):
            raise ValueError("output_root must not overlap {}".format(name))
    if manifest_tools._is_within(labels_root, audit_output):
        raise ValueError("audit_output must not be inside output_root")

    theoretical_states = 2 ** part_count - 1
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("expected must be an object")
    manifest_tools._require_fields(expected, _EXPECTED_FIELDS, "expected")
    if expected.get("root_candidate_count") != max_poses * part_count:
        raise ValueError("expected.root_candidate_count is invalid")
    if expected.get("root_action_count") != max_poses * part_count * len(labels.A6):
        raise ValueError("expected.root_action_count is invalid")
    if expected.get("theoretical_max_nonempty_states") != theoretical_states:
        raise ValueError("expected theoretical state count is invalid")
    if max_states != theoretical_states:
        raise ValueError("parameters.max_states must equal the nonempty subset ceiling")

    return {
        "manifest": manifest,
        "manifest_sha256": labels._sha256_bytes(
            labels._canonical_json(manifest).encode("utf-8")
        ),
        "assembly_dir": assembly_dir,
        "part_ids": part_ids,
        "input_sha256": input_sha256,
        "parameters": {
            "max_poses": max_poses,
            "pose_probability": pose_probability,
            "removal_timeout": removal_timeout,
            "stability_timeout": stability_timeout,
            "save_sdf": True,
            "max_states": max_states,
        },
        "cache_dir": cache_dir,
        "reference_root": reference_root,
        "labels_root": labels_root,
        "summary_path": labels_root / "summary.json",
        "audit_output": audit_output,
    }


def _read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank line at {}:{}".format(path, line_number))
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("invalid record at {}:{}".format(path, line_number))
            records.append(value)
    return records


def _existing_summary(path, job_hash):
    if not path.exists():
        return None
    records = _read_jsonl(path)
    if (
        len(records) != 1
        or records[0].get("schema_version") != SCHEMA_VERSION
        or records[0].get("record_type") != "summary"
        or records[0].get("complete") is not True
    ):
        raise RuntimeError("existing worker summary is incomplete or corrupt: {}".format(path))
    if records[0].get("job_hash") != job_hash:
        raise FileExistsError("existing worker summary belongs to another job: {}".format(path))
    result = dict(records[0])
    result["idempotent"] = True
    return result


def _state_dir(prepared, parts):
    return prepared["labels_root"] / labels._state_id(parts)


def _pose_identity(prepared, parts, generator):
    return {
        "schema_version": POSE_SCHEMA_VERSION,
        "manifest_sha256": prepared["manifest_sha256"],
        "input_sha256": prepared["input_sha256"],
        "state_id": labels._state_id(parts),
        "part_ids": list(parts),
        "parameters": {
            "max_poses": prepared["parameters"]["max_poses"],
            "probability_threshold": prepared["parameters"]["pose_probability"],
        },
        "generator": generator,
    }


def _validate_pose_record(path, identity):
    records = _read_jsonl(path)
    if len(records) != 1:
        raise ValueError("state pose checkpoint must contain one record: {}".format(path))
    record = records[0]
    spec_sha256 = labels._sha256_bytes(labels._canonical_json(identity).encode("utf-8"))
    if (
        record.get("schema_version") != POSE_SCHEMA_VERSION
        or record.get("complete") is not True
        or record.get("identity") != identity
        or record.get("spec_sha256") != spec_sha256
    ):
        raise ValueError("state pose checkpoint identity is invalid: {}".format(path))
    poses = record.get("poses")
    max_poses = identity["parameters"]["max_poses"]
    if not isinstance(poses, list) or not 1 <= len(poses) <= max_poses:
        raise ValueError("state pose checkpoint pose count is invalid: {}".format(path))
    for index, pose in enumerate(poses):
        if not isinstance(pose, dict) or pose.get("index") != index:
            raise ValueError("state pose checkpoint order is invalid: {}".format(path))
        if pose.get("pose_id") != "pose-{:03d}".format(index):
            raise ValueError("state pose checkpoint ID is invalid: {}".format(path))
        if pose.get("source") == "identity_fallback":
            if pose.get("pose") is not None or pose.get("probability") is not None or len(poses) != 1:
                raise ValueError("state pose fallback is invalid: {}".format(path))
        elif pose.get("source") == "stable_pose":
            if not pose_cache_tools._valid_matrix(pose.get("pose")):
                raise ValueError("state pose matrix is invalid: {}".format(path))
            probability = pose.get("probability")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("state pose probability is invalid: {}".format(path))
        else:
            raise ValueError("state pose source is invalid: {}".format(path))
    if record.get("pose_count") != len(poses):
        raise ValueError("state pose checkpoint count is invalid: {}".format(path))
    runtime_s = record.get("stable_pose_search_wall_s")
    if (
        isinstance(runtime_s, bool)
        or not isinstance(runtime_s, (int, float))
        or not math.isfinite(runtime_s)
        or runtime_s < 0
    ):
        raise ValueError("state pose checkpoint timing is invalid: {}".format(path))
    return record


def _state_poses(prepared, scratch_assembly, parts, generator, pose_factory=None):
    path = _state_dir(prepared, parts) / "poses.json"
    identity = _pose_identity(prepared, parts, generator)
    if path.exists():
        return _validate_pose_record(path, identity), True

    started = time.perf_counter()
    factory = pose_factory or labels._compute_stable_poses
    poses = factory(
        scratch_assembly,
        list(parts),
        prepared["parameters"]["max_poses"],
        prepared["parameters"]["pose_probability"],
    )
    if not poses:
        poses = [(None, None)]
    if len(poses) > prepared["parameters"]["max_poses"]:
        raise ValueError("stable-pose search returned more poses than requested")
    pose_records = [
        {
            "index": index,
            "pose_id": "pose-{:03d}".format(index),
            "pose": labels._json_value(pose),
            "probability": None if probability is None else float(probability),
            "source": "stable_pose" if pose is not None else "identity_fallback",
        }
        for index, (pose, probability) in enumerate(poses)
    ]
    record = {
        "schema_version": POSE_SCHEMA_VERSION,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": labels._sha256_bytes(
            labels._canonical_json(identity).encode("utf-8")
        ),
        "identity": identity,
        "assembly_id": prepared["assembly_dir"].name,
        "state_id": labels._state_id(parts),
        "part_ids": list(parts),
        "pose_count": len(pose_records),
        "poses": pose_records,
        "stable_pose_search_wall_s": time.perf_counter() - started,
    }
    try:
        labels._write_atomic(path, [record])
    except FileExistsError:
        return _validate_pose_record(path, identity), True
    return _validate_pose_record(path, identity), False


def _candidate_path(prepared, parts, pose_index, part_move):
    return (
        _state_dir(prepared, parts)
        / "pose-{:03d}".format(pose_index)
        / "part-{}.jsonl".format(part_move)
    )


def _candidate_hash(worker_job_hash, pose_checkpoint_sha256, parts, pose_index, part_move):
    spec = {
        "schema_version": SCHEMA_VERSION,
        "worker_job_hash": worker_job_hash,
        "state_id": labels._state_id(parts),
        "parts": list(parts),
        "pose_checkpoint_sha256": pose_checkpoint_sha256,
        "pose_index": pose_index,
        "part_move": part_move,
    }
    return labels._sha256_bytes(labels._canonical_json(spec).encode("utf-8"))


def _verify_scratch_sdf(scratch_assembly, cache_manifest):
    expected = [entry["path"] for entry in cache_manifest["files"]]
    actual = sorted(path.name for path in scratch_assembly.glob("*.sdf"))
    if actual != expected:
        raise RuntimeError("worker produced an unexpected SDF file set")
    if any(
        labels._sha256_file(scratch_assembly / entry["path"]) != entry["sha256"]
        for entry in cache_manifest["files"]
    ):
        raise RuntimeError("RedMax changed one or more cached SDF files")


def _run_candidate(
    prepared,
    parts,
    pose_record,
    pose_checkpoint_sha256,
    part_move,
    worker_job_hash,
    cache_manifest,
    cache_manifest_sha256,
    source,
    runtime,
    removal_oracle,
    stability_oracle,
    scratch_assembly,
    candidate_started=None,
):
    pose_index = pose_record["index"]
    output_path = _candidate_path(prepared, parts, pose_index, part_move)
    job_hash = _candidate_hash(
        worker_job_hash, pose_checkpoint_sha256, parts, pose_index, part_move
    )
    existing = labels._existing_complete(output_path, job_hash)
    if existing is not None:
        return _read_jsonl(output_path), True

    started = time.perf_counter() if candidate_started is None else candidate_started
    pose = pose_record["pose"]
    pose_id = pose_record["pose_id"]
    parts_after = [part for part in parts if part != part_move]
    state_id = labels._state_id(parts)
    child_state_id = labels._state_id(parts_after)
    stability_id = "{}:{}:{}".format(pose_id, child_state_id, part_move)

    removal_started = time.perf_counter()
    try:
        batch = removal_oracle.evaluate_removability(list(parts), part_move, pose)
        removal = labels._normalise_result(batch, default_rollouts=0)
        directions = labels._normalise_directions(batch)
    except Exception as exc:
        removal = labels._error_result(exc, time.perf_counter() - removal_started)
        directions = [(name, vector, dict(removal)) for name, vector in labels.A6]
    removal_wall_s = time.perf_counter() - removal_started

    stability_queried = any(result["label"] is True for _, _, result in directions)
    if stability_queried:
        stability_started = time.perf_counter()
        try:
            stability = labels._normalise_result(
                stability_oracle.evaluate_stability(parts_after, pose)
            )
        except Exception as exc:
            stability = labels._error_result(
                exc, time.perf_counter() - stability_started
            )
        stability_wall_s = time.perf_counter() - stability_started
    else:
        stability = conditional._not_evaluated_stability()
        stability_wall_s = 0.0

    _verify_scratch_sdf(scratch_assembly, cache_manifest)
    common = {
        "schema_version": SCHEMA_VERSION,
        "job_hash": job_hash,
        "worker_job_hash": worker_job_hash,
        "assembly_id": prepared["assembly_dir"].name,
        "state_id": state_id,
        "root_state_id": labels._state_id(prepared["part_ids"]),
        "child_state_id": child_state_id,
        "pose_id": pose_id,
        "pose_index": pose_index,
        "pose_probability": pose_record["probability"],
        "pose": pose,
        "pose_source": pose_record["source"],
        "part_move": part_move,
    }
    protocol = dict(prepared["manifest"]["protocol"])
    protocol.update(
        {
            "max_poses": prepared["parameters"]["max_poses"],
            "pose_probability_threshold": prepared["parameters"]["pose_probability"],
            "removal_timeout_s": prepared["parameters"]["removal_timeout"],
            "removal_timeout_scope": "per_direction",
            "stability_timeout_s": prepared["parameters"]["stability_timeout"],
            "stability_timeout_scope": "per_candidate_greedy_search",
            "save_sdf_in_scratch": True,
        }
    )
    records = [
        dict(
            common,
            record_type="header",
            created_at=datetime.now(timezone.utc).isoformat(),
            parts=list(parts),
            parts_after=parts_after,
            input={"sha256": prepared["input_sha256"]},
            protocol=protocol,
            manifest={
                "schema_version": prepared["manifest"]["schema_version"],
                "sha256": prepared["manifest_sha256"],
                "phase": prepared["manifest"]["phase"],
            },
            sdf_cache={
                "cache_key": cache_manifest["cache_key"],
                "manifest_sha256": cache_manifest_sha256,
                "file_count": cache_manifest["file_count"],
                "bytes": cache_manifest["bytes"],
                "files_unchanged": True,
            },
            state_pose_checkpoint={
                "schema_version": POSE_SCHEMA_VERSION,
                "manifest_sha256": pose_checkpoint_sha256,
            },
            provenance={
                "source": source,
                "runtime": runtime,
                "generator_sha256": labels._sha256_file(Path(__file__)),
                **(
                    {"scheduler_sha256": prepared["scheduler_sha256"]}
                    if prepared.get("scheduler_sha256")
                    else {}
                ),
            },
            host={
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "visible_cpu_count": os.cpu_count(),
            },
        ),
        dict(
            common,
            record_type="stability",
            stability_id=stability_id,
            queried=stability_queried,
            parts=parts_after,
            T=stability["label"],
            T_status=stability["status"],
            reason=stability["reason"],
            parts_fix=stability.get("parts_fix"),
            forward_steps=stability.get("forward_steps"),
            rollouts=stability.get("rollouts"),
            oracle_rollout_count=stability["rollout_count"],
            runtime_s=stability["runtime_s"],
            error=stability["error"],
        ),
    ]
    label_counts = {name: labels._label_counts() for name in ("R", "T", "Y")}
    labels._count_label(label_counts["T"], stability["label"])
    removal_rollouts = 0
    for direction_id, vector, removable in directions:
        y_label, y_status = labels._combined_label(
            removable["label"],
            stability["label"],
            removable["status"],
            stability["status"],
        )
        labels._count_label(label_counts["R"], removable["label"])
        labels._count_label(label_counts["Y"], y_label)
        removal_rollouts += removable["rollout_count"]
        records.append(
            dict(
                common,
                record_type="action",
                direction_id=direction_id,
                direction=list(vector),
                direction_frame="world",
                stability_ref=stability_id,
                T_queried=stability_queried,
                R=removable["label"],
                R_status=removable["status"],
                R_reason=removable["reason"],
                T=stability["label"],
                T_status=stability["status"],
                Y=y_label,
                Y_status=y_status,
                oracle_rollout_count=removable["rollout_count"],
                runtime_s=removable["runtime_s"],
                forward_steps=removable.get("forward_steps"),
                path_steps=removable.get("path_steps"),
                error=removable["error"],
            )
        )
    accounting = {
        "candidate_actions": len(labels.A6),
        "removability_batches": 1,
        "removability_rollouts": removal_rollouts,
        "stability_queries": 1 if stability_queried else 0,
        "stability_rollouts": stability["rollout_count"],
        "oracle_rollouts_total": removal_rollouts + stability["rollout_count"],
    }
    records.append(
        dict(
            common,
            record_type="summary",
            complete=True,
            accounting=accounting,
            label_counts=label_counts,
            removability=removal,
            timings_s={
                "removability_wall_s": removal_wall_s,
                "stability_wall_s": stability_wall_s,
                "candidate_wall_s": time.perf_counter() - started,
            },
            record_count=len(records) + 1,
        )
    )
    try:
        labels._write_atomic(output_path, records)
    except FileExistsError:
        labels._existing_complete(output_path, job_hash)
        return _read_jsonl(output_path), True
    return records, False


def _add_counts(target, source):
    for name in target:
        for value in target[name]:
            target[name][value] += source[name][value]


def _worker_job_hash(prepared, cache_manifest_sha256, source, runtime):
    spec = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": prepared["manifest_sha256"],
        "input_sha256": prepared["input_sha256"],
        "cache_manifest_sha256": cache_manifest_sha256,
        "source_sha256": source["sha256"],
        "source_git_commit": source["git_commit"],
        "generator_sha256": labels._sha256_file(Path(__file__)),
        "simulator_sha256": runtime["simulator"]["sha256"],
    }
    return labels._sha256_bytes(labels._canonical_json(spec).encode("utf-8"))


def run(prepared, *, asset_folder=None, oracle=None, pose_factory=None):
    """Traverse and checkpoint one manifest assembly."""
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    assembly_dir = prepared["assembly_dir"]
    asset_folder = Path(asset_folder).resolve() if asset_folder else repo_root / "assets"
    if not asset_folder.is_dir():
        raise FileNotFoundError("ASAP asset folder not found: {}".format(asset_folder))
    labels._validate_output_target(assembly_dir.parent, prepared["labels_root"])
    labels._validate_output_target(assembly_dir.parent, prepared["cache_dir"])

    preflight_started = time.perf_counter()
    runtime = labels._runtime_provenance()
    source = labels._source_provenance(repo_root)
    cache_manifest = sdf_cache_tools.validate_cache(
        prepared["cache_dir"],
        expected_input_sha256=prepared["input_sha256"],
        expected_parts=prepared["part_ids"],
        expected_simulator_sha256=runtime["simulator"]["sha256"],
        expected_sim_string_sha256=labels._sha256_file(
            repo_root / "plan_sequence" / "sim_string.py"
        ),
    )
    cache_manifest_sha256 = labels._sha256_file(prepared["cache_dir"] / "manifest.json")
    worker_job_hash = _worker_job_hash(
        prepared, cache_manifest_sha256, source, runtime
    )
    existing = _existing_summary(prepared["summary_path"], worker_job_hash)
    if existing is not None:
        return existing
    preflight_wall_s = time.perf_counter() - preflight_started

    one_time_setup_started = time.perf_counter()
    source_tree_before, source_tree_files = sdf_cache_tools._directory_snapshot(
        assembly_dir
    )
    pose_generator = pose_cache_tools._generator_identity(repo_root, runtime)
    current_run = {
        "state_pose_checkpoints_created": 0,
        "state_pose_checkpoints_reused": 0,
        "candidate_checkpoints_created": 0,
        "candidate_checkpoints_reused": 0,
    }
    copy_inputs_wall_s = 0.0
    copy_sdf_wall_s = 0.0
    traversal_wall_s = 0.0
    with tempfile.TemporaryDirectory(prefix="fast-reachable-{}-".format(assembly_dir.name)) as scratch:
        scratch_assembly = Path(scratch) / assembly_dir.name
        scratch_assembly.mkdir()
        copy_started = time.perf_counter()
        for source_path in labels._assembly_input_paths(assembly_dir):
            shutil.copy2(str(source_path), str(scratch_assembly / source_path.name))
        copy_inputs_wall_s = time.perf_counter() - copy_started
        if labels._hash_assembly(scratch_assembly)[0] != prepared["input_sha256"]:
            raise RuntimeError("scratch input hash differs from source")
        copy_started = time.perf_counter()
        for entry in cache_manifest["files"]:
            shutil.copy2(
                str(prepared["cache_dir"] / entry["path"]),
                str(scratch_assembly / entry["path"]),
            )
        copy_sdf_wall_s = time.perf_counter() - copy_started
        _verify_scratch_sdf(scratch_assembly, cache_manifest)

        if oracle is None:
            removal_oracle = labels._load_oracle(
                asset_folder,
                scratch_assembly,
                prepared["parameters"]["removal_timeout"],
                prepared["parameters"]["save_sdf"],
            )
            stability_oracle = labels._load_oracle(
                asset_folder,
                scratch_assembly,
                prepared["parameters"]["stability_timeout"],
                prepared["parameters"]["save_sdf"],
            )
        else:
            removal_oracle = stability_oracle = oracle

        one_time_setup_wall_s = time.perf_counter() - one_time_setup_started
        traversal_started = time.perf_counter()
        root = tuple(prepared["part_ids"])
        queue = deque([root])
        discovered = {root}
        evaluated_states = []
        terminal_states = []
        edges = []
        candidate_files = []
        pose_files = []
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

        while queue:
            parts = queue.popleft()
            if len(parts) == 1:
                terminal_states.append(
                    {"state_id": labels._state_id(parts), "parts": list(parts)}
                )
                continue
            evaluated_states.append(
                {"state_id": labels._state_id(parts), "parts": list(parts)}
            )
            pose_checkpoint, pose_reused = _state_poses(
                prepared, scratch_assembly, parts, pose_generator, pose_factory
            )
            current_run[
                "state_pose_checkpoints_reused" if pose_reused else "state_pose_checkpoints_created"
            ] += 1
            pose_path = _state_dir(prepared, parts) / "poses.json"
            pose_checkpoint_sha256 = labels._sha256_file(pose_path)
            pose_files.append(
                {
                    "state_id": labels._state_id(parts),
                    "path": pose_path.relative_to(prepared["labels_root"]).as_posix(),
                    "sha256": pose_checkpoint_sha256,
                }
            )
            feasible_parts = set()
            for pose_record in pose_checkpoint["poses"]:
                for part_move in parts:
                    records, reused = _run_candidate(
                        prepared,
                        parts,
                        pose_record,
                        pose_checkpoint_sha256,
                        part_move,
                        worker_job_hash,
                        cache_manifest,
                        cache_manifest_sha256,
                        source,
                        runtime,
                        removal_oracle,
                        stability_oracle,
                        scratch_assembly,
                    )
                    current_run[
                        "candidate_checkpoints_reused" if reused else "candidate_checkpoints_created"
                    ] += 1
                    candidate_path = _candidate_path(
                        prepared, parts, pose_record["index"], part_move
                    )
                    candidate_files.append(
                        {
                            "state_id": labels._state_id(parts),
                            "pose_id": pose_record["pose_id"],
                            "part_move": part_move,
                            "path": candidate_path.relative_to(prepared["labels_root"]).as_posix(),
                            "sha256": labels._sha256_file(candidate_path),
                        }
                    )
                    summary = records[-1]
                    accounting["candidate_count"] += 1
                    for field in accounting:
                        if field != "candidate_count":
                            accounting[field] += summary["accounting"].get(field, 0)
                    _add_counts(label_counts, summary["label_counts"])
                    for field in candidate_timing_s:
                        candidate_timing_s[field] += summary["timings_s"][field]
                    if any(action.get("Y") is True for action in records[2:8]):
                        feasible_parts.add(part_move)

            for part_move in parts:
                if part_move not in feasible_parts:
                    continue
                child = tuple(part for part in parts if part != part_move)
                edges.append(
                    {
                        "parent_state_id": labels._state_id(parts),
                        "child_state_id": labels._state_id(child),
                        "part_move": part_move,
                    }
                )
                if child not in discovered:
                    discovered.add(child)
                    if len(discovered) > prepared["parameters"]["max_states"]:
                        raise RuntimeError("reachable-state ceiling exceeded")
                    queue.append(child)

        _verify_scratch_sdf(scratch_assembly, cache_manifest)
        if labels._hash_assembly(scratch_assembly)[0] != prepared["input_sha256"]:
            raise RuntimeError("scratch inputs changed during traversal")
        traversal_wall_s = time.perf_counter() - traversal_started

    source_tree_after, _ = sdf_cache_tools._directory_snapshot(assembly_dir)
    if source_tree_after != source_tree_before:
        raise RuntimeError("authoritative assembly changed during traversal")

    accounting["stability_queries_avoided"] = (
        accounting["candidate_count"] - accounting["stability_queries"]
    )
    accounting["state_count"] = len(discovered)
    accounting["evaluated_state_count"] = len(evaluated_states)
    accounting["terminal_state_count"] = len(terminal_states)
    accounting["transition_edge_count"] = len(edges)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "summary",
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_hash": worker_job_hash,
        "phase": prepared["manifest"]["phase"],
        "assembly_id": assembly_dir.name,
        "root_state_id": labels._state_id(prepared["part_ids"]),
        "manifest_sha256": prepared["manifest_sha256"],
        "input": {
            "sha256": prepared["input_sha256"],
            "source_directory_snapshot_sha256": source_tree_before,
            "source_directory_files": source_tree_files,
            "source_directory_unchanged": True,
        },
        "sdf_cache": {
            "cache_key": cache_manifest["cache_key"],
            "manifest_sha256": cache_manifest_sha256,
            "file_count": cache_manifest["file_count"],
            "bytes": cache_manifest["bytes"],
            "files_unchanged": True,
        },
        "provenance": {
            "source": source,
            "runtime": runtime,
            "generator_sha256": labels._sha256_file(Path(__file__)),
        },
        "accounting": accounting,
        "label_counts": label_counts,
        "states": evaluated_states,
        "terminal_states": terminal_states,
        "transitions": edges,
        "state_pose_files": pose_files,
        "candidate_files": candidate_files,
        "current_run": current_run,
        "timings_s": {
            "preflight_wall_s": preflight_wall_s,
            "one_time_setup_wall_s": one_time_setup_wall_s,
            "copy_inputs_wall_s": copy_inputs_wall_s,
            "copy_sdf_cache_wall_s": copy_sdf_wall_s,
            "traversal_wall_s": traversal_wall_s,
            "stable_pose_search_wall_s": sum(
                record["stable_pose_search_wall_s"]
                for record in (
                    _read_jsonl(_state_dir(prepared, state["parts"]) / "poses.json")[0]
                    for state in evaluated_states
                )
            ),
            "candidate_dataset_totals": candidate_timing_s,
            "total_wall_s": time.perf_counter() - started,
        },
    }
    try:
        labels._write_atomic(prepared["summary_path"], [summary])
    except FileExistsError:
        return _existing_summary(prepared["summary_path"], worker_job_hash)
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--asset-folder")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    prepared = _validate_manifest(args.manifest, args.source_root, args.output_root)
    if args.validate_only:
        print(
            "validated {} for {}".format(
                prepared["manifest"]["phase"], prepared["assembly_dir"].name
            ),
            flush=True,
        )
        return 0
    result = run(prepared, asset_folder=args.asset_folder)
    print(labels._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
