"""Generate one isolated, removability-first FAST label candidate."""

import argparse
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_pose_cache as pose_cache_tools
from plan_sequence import prepare_sdf_cache as sdf_cache_tools
from plan_sequence import run_sim_label_manifest as manifest_tools


SCHEMA_VERSION = "fast.conditional-sim-labels.v1"
SCHEMA_VERSION_V2 = "fast.conditional-sim-labels.v2"
MANIFEST_SCHEMA_VERSION = "fast.conditional-label-manifest.v1"
MANIFEST_SCHEMA_VERSION_V2 = "fast.conditional-label-manifest.v2"
_ROOT_FIELDS = {
    "schema_version",
    "label_schema_version",
    "phase",
    "protocol",
    "parameters",
    "assembly",
    "cache",
    "baseline",
    "output_root",
    "audit_output",
    "expected",
}
_ROOT_FIELDS_V2 = _ROOT_FIELDS | {"pose_cache", "reference_labels"}
_ASSEMBLY_FIELDS = {"path", "part_count", "part_ids", "input_sha256"}
_PARAMETER_FIELDS = {
    "max_poses",
    "pose_probability",
    "removal_timeout_s",
    "stability_timeout_s",
    "save_sdf",
}
_PROTOCOL = {
    "state_scope": "root",
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
    "candidate_isolation": "one_process_per_pose_part",
}
_PROTOCOL_V2 = dict(_PROTOCOL, stable_pose_policy="validated_shared_cache")


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
    manifest_schema = manifest.get("schema_version")
    if manifest_schema == MANIFEST_SCHEMA_VERSION:
        root_fields = _ROOT_FIELDS
        label_schema = SCHEMA_VERSION
        protocol = _PROTOCOL
    elif manifest_schema == MANIFEST_SCHEMA_VERSION_V2:
        root_fields = _ROOT_FIELDS_V2
        label_schema = SCHEMA_VERSION_V2
        protocol = _PROTOCOL_V2
    else:
        raise ValueError("unsupported conditional manifest schema")
    manifest_tools._require_fields(manifest, root_fields, "manifest")
    if manifest.get("label_schema_version") != label_schema:
        raise ValueError("manifest label schema does not match the worker")
    manifest_tools._component(manifest.get("phase"), "phase")
    if manifest.get("protocol") != protocol:
        raise ValueError("manifest protocol does not match its schema")

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
        or any(not isinstance(part, str) or not part for part in part_ids)
        or len(set(part_ids)) != len(part_ids)
        or sorted(part_ids) != part_ids
    ):
        raise ValueError("assembly part IDs are invalid")
    expected_input = assembly.get("input_sha256")
    if (
        not isinstance(expected_input, str)
        or len(expected_input) != 64
        or any(character not in "0123456789abcdef" for character in expected_input)
    ):
        raise ValueError("assembly.input_sha256 must be lowercase SHA-256")
    if not assembly_dir.is_dir():
        raise FileNotFoundError("assembly not found: {}".format(assembly_dir))
    if not (assembly_dir / "contact_graph.json").is_file():
        raise FileNotFoundError("contact graph not found: {}".format(assembly_dir))
    actual_parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    if actual_parts != part_ids:
        raise ValueError("assembly part IDs do not match its OBJ files")
    if labels._hash_assembly(assembly_dir)[0] != expected_input:
        raise ValueError("assembly input SHA-256 does not match")

    cache_relative = manifest_tools._relative_path(manifest.get("cache"), "cache")
    pose_cache_relative = None
    reference_labels_relative = None
    if manifest_schema == MANIFEST_SCHEMA_VERSION_V2:
        pose_cache_relative = manifest_tools._relative_path(
            manifest.get("pose_cache"), "pose_cache"
        )
        reference_labels_relative = manifest_tools._relative_path(
            manifest.get("reference_labels"), "reference_labels"
        )
    baseline_relative = manifest_tools._relative_path(
        manifest.get("baseline"), "baseline", ".jsonl"
    )
    labels_relative = manifest_tools._relative_path(
        manifest.get("output_root"), "output_root"
    )
    audit_relative = manifest_tools._relative_path(
        manifest.get("audit_output"), "audit_output", ".json"
    )
    cache_dir = manifest_tools._resolve_relative(output_root, cache_relative, "cache")
    pose_cache_dir = (
        manifest_tools._resolve_relative(
            output_root, pose_cache_relative, "pose_cache"
        )
        if pose_cache_relative is not None
        else None
    )
    reference_labels_root = (
        manifest_tools._resolve_relative(
            output_root, reference_labels_relative, "reference_labels"
        )
        if reference_labels_relative is not None
        else None
    )
    baseline_path = manifest_tools._resolve_relative(
        output_root, baseline_relative, "baseline"
    )
    labels_root = manifest_tools._resolve_relative(
        output_root, labels_relative, "output_root"
    )
    audit_output = manifest_tools._resolve_relative(
        output_root, audit_relative, "audit_output"
    )

    candidate_count = max_poses * part_count
    expected = manifest.get("expected")
    if not isinstance(expected, dict) or expected.get("candidate_count") != candidate_count:
        raise ValueError("expected.candidate_count must equal max_poses * part_count")
    if expected.get("action_count") != candidate_count * len(labels.A6):
        raise ValueError("expected.action_count must equal candidate_count * 6")

    manifest_sha256 = labels._sha256_bytes(
        labels._canonical_json(manifest).encode("utf-8")
    )
    candidates = []
    for index in range(candidate_count):
        pose_index, part_index = divmod(index, part_count)
        part_move = part_ids[part_index]
        candidates.append(
            {
                "index": index,
                "pose_index": pose_index,
                "part_move": part_move,
                "output_path": labels_root
                / "pose-{:03d}".format(pose_index)
                / "part-{}.jsonl".format(part_move),
            }
        )
    return {
        "manifest": manifest,
        "manifest_schema_version": manifest_schema,
        "label_schema_version": label_schema,
        "manifest_sha256": manifest_sha256,
        "assembly_dir": assembly_dir,
        "part_ids": part_ids,
        "input_sha256": expected_input,
        "parameters": {
            "max_poses": max_poses,
            "pose_probability": pose_probability,
            "removal_timeout": removal_timeout,
            "stability_timeout": stability_timeout,
            "save_sdf": True,
        },
        "cache_dir": cache_dir,
        "pose_cache_dir": pose_cache_dir,
        "reference_labels_root": reference_labels_root,
        "baseline_path": baseline_path,
        "labels_root": labels_root,
        "audit_output": audit_output,
        "candidates": candidates,
    }


def _not_evaluated_stability():
    return {
        "label": None,
        "status": "not_evaluated",
        "reason": "no_proven_removable_direction",
        "rollout_count": 0,
        "runtime_s": 0.0,
        "error": None,
        "parts_fix": None,
        "forward_steps": 0,
        "path_steps": None,
        "rollouts": [],
    }


def run_candidate(
    prepared,
    index,
    *,
    asset_folder=None,
    oracle=None,
):
    """Run one manifest candidate; each call belongs in its own process."""
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    if type(index) is not int or not 0 <= index < len(prepared["candidates"]):
        raise IndexError("candidate index out of range: {}".format(index))
    candidate = prepared["candidates"][index]
    assembly_dir = prepared["assembly_dir"]
    output_path = candidate["output_path"]
    asset_folder = Path(asset_folder).resolve() if asset_folder else repo_root / "assets"
    if not asset_folder.is_dir():
        raise FileNotFoundError("ASAP asset folder not found: {}".format(asset_folder))
    labels._validate_output_target(assembly_dir.parent, output_path)
    labels._validate_output_target(assembly_dir.parent, prepared["cache_dir"])
    if prepared["pose_cache_dir"] is not None:
        labels._validate_output_target(
            assembly_dir.parent, prepared["pose_cache_dir"]
        )

    parts = prepared["part_ids"]
    parameters = prepared["parameters"]
    runtime = labels._runtime_provenance()
    source = labels._source_provenance(repo_root)
    cache_manifest = sdf_cache_tools.validate_cache(
        prepared["cache_dir"],
        expected_input_sha256=prepared["input_sha256"],
        expected_parts=parts,
        expected_simulator_sha256=runtime["simulator"]["sha256"],
        expected_sim_string_sha256=labels._sha256_file(
            repo_root / "plan_sequence" / "sim_string.py"
        ),
    )
    cache_manifest_sha256 = labels._sha256_file(
        prepared["cache_dir"] / "manifest.json"
    )
    pose_cache_manifest = None
    pose_cache_manifest_sha256 = None
    cached_poses = None
    if prepared["pose_cache_dir"] is not None:
        pose_cache_manifest = pose_cache_tools.validate_cache(
            prepared["pose_cache_dir"],
            expected_input_sha256=prepared["input_sha256"],
            expected_parts=parts,
            expected_max_poses=parameters["max_poses"],
            expected_probability_threshold=parameters["pose_probability"],
            expected_generator=pose_cache_tools._generator_identity(repo_root, runtime),
        )
        if pose_cache_manifest["pose_count"] < parameters["max_poses"]:
            raise ValueError(
                "stable-pose cache contains fewer poses than the candidate manifest"
            )
        pose_cache_manifest_sha256 = labels._sha256_file(
            prepared["pose_cache_dir"] / "manifest.json"
        )
        cached_poses = [
            (entry["pose"], entry["probability"])
            for entry in pose_cache_manifest["poses"]
        ]
    generator_sha256 = labels._sha256_file(Path(__file__))
    job_spec = {
        "schema_version": prepared["label_schema_version"],
        "manifest_sha256": prepared["manifest_sha256"],
        "candidate_index": index,
        "pose_index": candidate["pose_index"],
        "part_move": candidate["part_move"],
        "input_sha256": prepared["input_sha256"],
        "cache_manifest_sha256": cache_manifest_sha256,
        "source_sha256": source["sha256"],
        "source_git_commit": source["git_commit"],
        "generator_sha256": generator_sha256,
        "simulator_sha256": runtime["simulator"]["sha256"],
    }
    if pose_cache_manifest_sha256 is not None:
        job_spec["pose_cache_manifest_sha256"] = pose_cache_manifest_sha256
    job_hash = labels._sha256_bytes(labels._canonical_json(job_spec).encode("utf-8"))
    existing = labels._existing_complete(output_path, job_hash)
    if existing is not None:
        return existing

    source_tree_before, source_tree_files = sdf_cache_tools._directory_snapshot(
        assembly_dir
    )
    timings = {}
    with tempfile.TemporaryDirectory(
        prefix="fast-conditional-{}-{}-".format(assembly_dir.name, index)
    ) as scratch:
        scratch_assembly = Path(scratch) / assembly_dir.name
        scratch_assembly.mkdir()
        copy_started = time.perf_counter()
        for source_path in labels._assembly_input_paths(assembly_dir):
            shutil.copy2(str(source_path), str(scratch_assembly / source_path.name))
        timings["copy_inputs_wall_s"] = time.perf_counter() - copy_started
        if labels._hash_assembly(scratch_assembly)[0] != prepared["input_sha256"]:
            raise RuntimeError("scratch input hash differs from source")

        cache_copy_started = time.perf_counter()
        for entry in cache_manifest["files"]:
            shutil.copy2(
                str(prepared["cache_dir"] / entry["path"]),
                str(scratch_assembly / entry["path"]),
            )
        timings["copy_sdf_cache_wall_s"] = time.perf_counter() - cache_copy_started

        if cached_poses is None:
            pose_started = time.perf_counter()
            poses = labels._compute_stable_poses(
                scratch_assembly,
                parts,
                parameters["max_poses"],
                parameters["pose_probability"],
            )
            timings["stable_pose_search_wall_s"] = time.perf_counter() - pose_started
            if not poses:
                poses = [(None, None)]
        else:
            poses = cached_poses
            timings["stable_pose_search_wall_s"] = 0.0
        if candidate["pose_index"] >= len(poses):
            raise ValueError(
                "pose index {} unavailable; search returned {} pose(s)".format(
                    candidate["pose_index"], len(poses)
                )
            )
        pose, probability = poses[candidate["pose_index"]]
        pose_id = "pose-{:03d}".format(candidate["pose_index"])
        pose_json = labels._json_value(pose)
        pose_source = "stable_pose" if pose is not None else "identity_fallback"
        part_move = candidate["part_move"]
        parts_after = [part for part in parts if part != part_move]
        root_state_id = labels._state_id(parts)
        child_state_id = labels._state_id(parts_after)
        stability_id = "{}:{}:{}".format(pose_id, child_state_id, part_move)

        if oracle is None:
            removal_oracle = labels._load_oracle(
                asset_folder,
                scratch_assembly,
                parameters["removal_timeout"],
                parameters["save_sdf"],
            )
            stability_oracle = None
        else:
            removal_oracle = stability_oracle = oracle

        removal_started = time.perf_counter()
        try:
            removal_batch = removal_oracle.evaluate_removability(parts, part_move, pose)
            removal = labels._normalise_result(removal_batch, default_rollouts=0)
            directions = labels._normalise_directions(removal_batch)
        except Exception as exc:
            removal = labels._error_result(exc, time.perf_counter() - removal_started)
            directions = [
                (direction_id, vector, dict(removal))
                for direction_id, vector in labels.A6
            ]
        timings["removability_wall_s"] = time.perf_counter() - removal_started

        stability_queried = any(result["label"] is True for _, _, result in directions)
        if stability_queried:
            if stability_oracle is None:
                stability_oracle = labels._load_oracle(
                    asset_folder,
                    scratch_assembly,
                    parameters["stability_timeout"],
                    parameters["save_sdf"],
                )
            stability_started = time.perf_counter()
            try:
                stability = labels._normalise_result(
                    stability_oracle.evaluate_stability(parts_after, pose)
                )
            except Exception as exc:
                stability = labels._error_result(
                    exc, time.perf_counter() - stability_started
                )
            timings["stability_wall_s"] = time.perf_counter() - stability_started
        else:
            stability = _not_evaluated_stability()
            timings["stability_wall_s"] = 0.0

        sdf_files = sorted(scratch_assembly.glob("*.sdf"))
        expected_sdf_names = [entry["path"] for entry in cache_manifest["files"]]
        if [path.name for path in sdf_files] != expected_sdf_names:
            raise RuntimeError("candidate produced an unexpected SDF file set")
        if any(
            labels._sha256_file(scratch_assembly / entry["path"])
            != entry["sha256"]
            for entry in cache_manifest["files"]
        ):
            raise RuntimeError("RedMax changed one or more cached SDF files")

    source_tree_after, _ = sdf_cache_tools._directory_snapshot(assembly_dir)
    if source_tree_after != source_tree_before:
        raise RuntimeError("authoritative assembly changed during candidate generation")

    protocol = dict(prepared["manifest"]["protocol"])
    protocol.update(
        {
            "max_poses": parameters["max_poses"],
            "pose_probability_threshold": parameters["pose_probability"],
            "removal_timeout_s": parameters["removal_timeout"],
            "removal_timeout_scope": "per_direction",
            "stability_timeout_s": parameters["stability_timeout"],
            "stability_timeout_scope": "per_candidate_greedy_search",
            "save_sdf_in_scratch": True,
        }
    )
    common = {
        "schema_version": prepared["label_schema_version"],
        "job_hash": job_hash,
        "assembly_id": assembly_dir.name,
        "candidate_index": index,
        "root_state_id": root_state_id,
        "child_state_id": child_state_id,
        "pose_id": pose_id,
        "pose_index": candidate["pose_index"],
        "pose_probability": None if probability is None else float(probability),
        "pose": pose_json,
        "pose_source": pose_source,
        "part_move": part_move,
    }
    header = dict(
            common,
            record_type="header",
            created_at=datetime.now(timezone.utc).isoformat(),
            parts=parts,
            parts_after=parts_after,
            input={
                "sha256": prepared["input_sha256"],
                "source_directory_snapshot_sha256": source_tree_before,
                "source_directory_files": source_tree_files,
                "source_directory_unchanged": True,
            },
            protocol=protocol,
            manifest={
                "schema_version": prepared["manifest_schema_version"],
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
            provenance={
                "source": source,
                "runtime": runtime,
                "generator_sha256": generator_sha256,
            },
            host={
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "visible_cpu_count": os.cpu_count(),
            },
        )
    if pose_cache_manifest is not None:
        header["stable_pose_cache"] = {
            "cache_key": pose_cache_manifest["cache_key"],
            "manifest_sha256": pose_cache_manifest_sha256,
            "pose_count": pose_cache_manifest["pose_count"],
        }
    records = [header]
    records.append(
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
        )
    )

    label_counts = {
        "R": labels._label_counts(),
        "T": labels._label_counts(),
        "Y": labels._label_counts(),
    }
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

    timings["total_wall_s"] = time.perf_counter() - started
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
            timings_s=timings,
            record_count=len(records) + 1,
        )
    )
    try:
        labels._write_atomic(output_path, records)
    except FileExistsError:
        existing = labels._existing_complete(output_path, job_hash)
        if existing is not None:
            return existing
        raise
    return records[-1]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--asset-folder")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=int)
    mode.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    prepared = _validate_manifest(args.manifest, args.source_root, args.output_root)
    if args.validate_only:
        print(
            labels._canonical_json(
                {
                    "status": "validated",
                    "manifest_sha256": prepared["manifest_sha256"],
                    "phase": prepared["manifest"]["phase"],
                    "candidate_count": len(prepared["candidates"]),
                }
            ),
            flush=True,
        )
        return 0
    result = run_candidate(
        prepared,
        args.index,
        asset_folder=args.asset_folder,
    )
    print(labels._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
