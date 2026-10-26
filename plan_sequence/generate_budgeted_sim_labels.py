"""Generate budgeted, breadth/depth-balanced FAST labels for one assembly."""

import argparse
import math
import multiprocessing
import os
import platform
import queue
import shutil
import socket
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

from assets.load import load_assembly
from plan_sequence import dfs_search
from plan_sequence import generate_reachable_sim_labels as reachable
from plan_sequence import generate_sim_labels as labels
from plan_sequence import mixed_search
from plan_sequence import prepare_pose_cache as pose_cache_tools
from plan_sequence import prepare_sdf_cache as sdf_cache_tools
from plan_sequence import run_sim_label_manifest as manifest_tools


MANIFEST_SCHEMA_VERSION = "fast.budgeted-label-manifest.v1"
_ROOT_FIELDS = {
    "schema_version",
    "label_schema_version",
    "phase",
    "protocol",
    "parameters",
    "assembly",
    "cache",
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
    "seed",
}
_EXPECTED_FIELDS = {"state_part_attempt_budget", "candidate_evaluation_budget"}
_PROTOCOL = {
    "state_scope": "budgeted_oracle_reachable_remaining_part_states",
    "traversal": "greedy_spine_then_alternating_breadth_depth",
    "candidate_ranking": "heur-out",
    "candidate_tie_break": "seeded_sha256",
    "breadth_state_priority": "shallowest_then_least_expanded",
    "depth_state_priority": "deepest_then_least_expanded",
    "breadth_candidate_order": "upper_middle_lower_rank_strata",
    "depth_candidate_order": "heur-out",
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
    "budget_atomicity": "all_poses_for_selected_state_part",
}


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
        raise ValueError("unsupported budgeted-label manifest schema")
    if manifest.get("label_schema_version") != reachable.SCHEMA_VERSION:
        raise ValueError("manifest label schema does not match the candidate worker")
    manifest_tools._component(manifest.get("phase"), "phase")
    if manifest.get("protocol") != _PROTOCOL:
        raise ValueError("manifest protocol does not match the worker")

    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    manifest_tools._require_fields(parameters, _PARAMETER_FIELDS, "parameters")
    if parameters.get("max_poses") != 3:
        raise ValueError("parameters.max_poses must be 3 for the approved budget")
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
    seed = parameters.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("parameters.seed must be a non-negative integer")

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
        or not 7 <= part_count <= 20
        or not isinstance(part_ids, list)
        or len(part_ids) != part_count
        or part_ids != sorted(part_ids)
        or len(set(part_ids)) != len(part_ids)
        or any(not isinstance(part, str) or not part for part in part_ids)
    ):
        raise ValueError("medium assembly part IDs are invalid")
    input_sha256 = reachable._sha256(
        assembly.get("input_sha256"), "assembly.input_sha256"
    )
    if not assembly_dir.is_dir():
        raise FileNotFoundError("assembly not found: {}".format(assembly_dir))
    if not (assembly_dir / "contact_graph.json").is_file():
        raise FileNotFoundError("contact graph not found: {}".format(assembly_dir))
    actual_parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    if actual_parts != part_ids:
        raise ValueError("assembly part IDs do not match its OBJ files")
    if labels._hash_assembly(assembly_dir)[0] != input_sha256:
        raise ValueError("assembly input SHA-256 does not match")

    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("expected must be an object")
    manifest_tools._require_fields(expected, _EXPECTED_FIELDS, "expected")
    state_part_budget = 4 * (part_count - 1)
    candidate_budget = 12 * (part_count - 1)
    if expected.get("state_part_attempt_budget") != state_part_budget:
        raise ValueError("expected.state_part_attempt_budget must equal 4*(part_count-1)")
    if expected.get("candidate_evaluation_budget") != candidate_budget:
        raise ValueError("expected.candidate_evaluation_budget must equal 12*(part_count-1)")

    cache_relative = manifest_tools._relative_path(manifest.get("cache"), "cache")
    labels_relative = manifest_tools._relative_path(
        manifest.get("output_root"), "output_root"
    )
    audit_relative = manifest_tools._relative_path(
        manifest.get("audit_output"), "audit_output", ".json"
    )
    cache_dir = manifest_tools._resolve_relative(output_root, cache_relative, "cache")
    labels_root = manifest_tools._resolve_relative(
        output_root, labels_relative, "output_root"
    )
    audit_output = manifest_tools._resolve_relative(
        output_root, audit_relative, "audit_output"
    )
    if manifest_tools._is_within(cache_dir, labels_root) or manifest_tools._is_within(
        labels_root, cache_dir
    ):
        raise ValueError("output_root must not overlap cache")
    if manifest_tools._is_within(labels_root, audit_output):
        raise ValueError("audit_output must not be inside output_root")

    return {
        "manifest": manifest,
        "manifest_sha256": labels._sha256_bytes(
            labels._canonical_json(manifest).encode("utf-8")
        ),
        "assembly_dir": assembly_dir,
        "part_ids": part_ids,
        "input_sha256": input_sha256,
        "parameters": {
            "max_poses": 3,
            "pose_probability": pose_probability,
            "removal_timeout": removal_timeout,
            "stability_timeout": stability_timeout,
            "save_sdf": True,
            "seed": seed,
            "state_part_budget": state_part_budget,
            "candidate_budget": candidate_budget,
        },
        "cache_dir": cache_dir,
        "search_mode": "mixed",
        "labels_root": labels_root,
        "summary_path": labels_root / "summary.json",
        "audit_output": audit_output,
    }


def _heur_out_ranker(assembly_dir, seed):
    assembly = load_assembly(assembly_dir)
    meshes = {part: value["mesh"] for part, value in assembly.items()}

    def rank(parts):
        center = trimesh.Scene([meshes[part] for part in parts]).centroid
        distances = {
            part: float(np.linalg.norm(meshes[part].center_mass - center)) for part in parts
        }
        if any(not math.isfinite(value) for value in distances.values()):
            raise ValueError("heur-out produced a non-finite distance")
        return sorted(
            parts,
            key=lambda part: (
                -distances[part],
                mixed_search.seeded_tie(seed, parts, part),
                part,
            ),
        )

    return rank


def _worker_job_hash(prepared, cache_sha256, source, runtime):
    spec = {
        "schema_version": reachable.SCHEMA_VERSION,
        "manifest_sha256": prepared["manifest_sha256"],
        "input_sha256": prepared["input_sha256"],
        "cache_manifest_sha256": cache_sha256,
        "source_sha256": source["sha256"],
        "source_git_commit": source["git_commit"],
        "candidate_generator_sha256": labels._sha256_file(Path(reachable.__file__)),
        "scheduler_sha256": prepared["scheduler_sha256"],
        "simulator_sha256": runtime["simulator"]["sha256"],
    }
    return labels._sha256_bytes(labels._canonical_json(spec).encode("utf-8"))


def _existing_summary(path, job_hash):
    if not path.exists():
        return None
    records = reachable._read_jsonl(path)
    if (
        len(records) != 1
        or records[0].get("schema_version") != reachable.SCHEMA_VERSION
        or records[0].get("record_type") != "summary"
        or records[0].get("complete") is not True
    ):
        raise RuntimeError("existing worker summary is incomplete or corrupt: {}".format(path))
    if records[0].get("job_hash") != job_hash:
        raise FileExistsError("existing worker summary belongs to another job: {}".format(path))
    result = dict(records[0])
    result["idempotent"] = True
    return result


def _isolated_candidate_worker(result_queue, arguments):
    """Run one RedMax candidate in a disposable process."""
    try:
        candidate_started = time.perf_counter()
        prepared = arguments[0]
        asset_folder = arguments[-2]
        scratch_assembly = arguments[-1]
        removal_oracle = labels._load_oracle(
            asset_folder,
            scratch_assembly,
            prepared["parameters"]["removal_timeout"],
            True,
        )
        stability_oracle = labels._load_oracle(
            asset_folder,
            scratch_assembly,
            prepared["parameters"]["stability_timeout"],
            True,
        )
        reachable._run_candidate(
            *arguments[:-2],
            removal_oracle,
            stability_oracle,
            scratch_assembly,
            candidate_started,
        )
        result_queue.put({"ok": True})
    except BaseException:
        result_queue.put({"ok": False, "error": traceback.format_exc()})


def _run_candidate_isolated(
    prepared,
    parts,
    pose_record,
    pose_sha256,
    part_move,
    worker_job_hash,
    cache_manifest,
    cache_sha256,
    source,
    runtime,
    asset_folder,
    scratch_assembly,
):
    """Reuse a checkpoint or create it in a fresh spawned process."""
    path = reachable._candidate_path(
        prepared, parts, pose_record["index"], part_move
    )
    job_hash = reachable._candidate_hash(
        worker_job_hash, pose_sha256, parts, pose_record["index"], part_move
    )
    if labels._existing_complete(path, job_hash) is not None:
        return reachable._read_jsonl(path), True

    arguments = (
        prepared,
        parts,
        pose_record,
        pose_sha256,
        part_move,
        worker_job_hash,
        cache_manifest,
        cache_sha256,
        source,
        runtime,
        Path(asset_folder),
        Path(scratch_assembly),
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_isolated_candidate_worker, args=(result_queue, arguments)
    )
    process.start()
    process.join()
    try:
        result = result_queue.get(timeout=5)
    except queue.Empty as exc:
        raise RuntimeError(
            "isolated candidate exited without a result (exit code {})".format(
                process.exitcode
            )
        ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()
    if not result["ok"]:
        raise RuntimeError("isolated candidate failed:\n{}".format(result["error"]))
    if process.exitcode != 0:
        raise RuntimeError(
            "isolated candidate exited with code {}".format(process.exitcode)
        )
    labels._existing_complete(path, job_hash)
    return reachable._read_jsonl(path), False


def _coverage(root, discovered, attempted, transitions, schedule):
    states_per_depth = {}
    attempts_per_depth = {}
    lane_counts = {"spine": 0, "breadth": 0, "depth": 0, "dfs": 0}
    rank_counts = {"top": 0, "non_top": 0}
    for state in discovered:
        depth = len(root) - len(state)
        states_per_depth[str(depth)] = states_per_depth.get(str(depth), 0) + 1
    for step in schedule:
        depth = str(step["depth"])
        attempts_per_depth[depth] = attempts_per_depth.get(depth, 0) + 1
        lane_counts[step["lane"]] += 1
        rank_counts["top" if step["heuristic_rank"] == 0 else "non_top"] += 1
    child_counts = {}
    for edge in transitions:
        parent = edge["parent_state_id"]
        child_counts[parent] = child_counts.get(parent, 0) + 1
    state_coverage = []
    for state in sorted(discovered, key=lambda value: (len(root) - len(value), value)):
        tried = list(attempted.get(state, ()))
        state_coverage.append(
            {
                "state_id": labels._state_id(state),
                "depth": len(root) - len(state),
                "part_count": len(state),
                "attempted_parts": tried,
                "outgoing_part_coverage": len(tried) / float(len(state)),
                "feasible_child_count": child_counts.get(labels._state_id(state), 0),
            }
        )
    evaluated_count = sum(1 for state in discovered if attempted.get(state))
    return {
        "max_depth": max(len(root) - len(state) for state in discovered),
        "complete_sequence_found": any(len(state) == 1 for state in discovered),
        "states_per_depth": states_per_depth,
        "attempts_per_depth": attempts_per_depth,
        "lane_counts": lane_counts,
        "heuristic_rank_counts": rank_counts,
        "observed_branching_factor_mean": (
            len(transitions) / float(evaluated_count) if evaluated_count else 0.0
        ),
        "states": state_coverage,
    }


def run(prepared, *, asset_folder=None, oracle=None, pose_factory=None, ranker=None):
    """Run one resumable, assembly-owned mixed-search label job."""
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
    cache_sha256 = labels._sha256_file(prepared["cache_dir"] / "manifest.json")
    prepared.setdefault("scheduler_sha256", labels._sha256_file(Path(__file__)))
    worker_job_hash = _worker_job_hash(prepared, cache_sha256, source, runtime)
    existing = _existing_summary(prepared["summary_path"], worker_job_hash)
    if existing is not None:
        return existing
    preflight_wall_s = time.perf_counter() - preflight_started

    source_before, source_files = sdf_cache_tools._directory_snapshot(assembly_dir)
    pose_generator = pose_cache_tools._generator_identity(repo_root, runtime)
    current_run = {
        "state_pose_checkpoints_created": 0,
        "state_pose_checkpoints_reused": 0,
        "candidate_checkpoints_created": 0,
        "candidate_checkpoints_reused": 0,
    }
    copy_inputs_wall_s = 0.0
    copy_sdf_wall_s = 0.0
    with tempfile.TemporaryDirectory(prefix="fast-budgeted-{}-".format(assembly_dir.name)) as scratch:
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
        reachable._verify_scratch_sdf(scratch_assembly, cache_manifest)

        isolate_candidates = (
            prepared.get("candidate_isolation") == "one_process_per_candidate"
            and oracle is None
        )
        if oracle is None and not isolate_candidates:
            removal_oracle = labels._load_oracle(
                asset_folder,
                scratch_assembly,
                prepared["parameters"]["removal_timeout"],
                True,
            )
            stability_oracle = labels._load_oracle(
                asset_folder,
                scratch_assembly,
                prepared["parameters"]["stability_timeout"],
                True,
            )
        elif oracle is not None:
            removal_oracle = stability_oracle = oracle
        else:
            removal_oracle = stability_oracle = None
        ranker = ranker or _heur_out_ranker(scratch_assembly, prepared["parameters"]["seed"])
        one_time_setup_wall_s = time.perf_counter() - started - preflight_wall_s

        traversal_started = time.perf_counter()
        root = tuple(prepared["part_ids"])
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
        schedule = []
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

        stop_reason = None
        while (
            len(schedule) < prepared["parameters"]["state_part_budget"]
            and accounting["candidate_count"] < prepared["parameters"]["candidate_budget"]
        ):
            if prepared["search_mode"] == "adaptive":
                selected = mixed_search.select_next_adaptive(
                    root,
                    discovered,
                    attempted,
                    spine_state,
                    balanced_step,
                    ranker,
                    prepared["parameters"]["seed"],
                )
            elif prepared["search_mode"] == "mixed":
                selected = mixed_search.select_next(
                    root, discovered, attempted, spine_state, balanced_step, ranker
                )
            else:
                selected = dfs_search.select_next(dfs_path, attempted, ranker)
            if selected is None:
                stop_reason = "frontier_exhausted"
                break
            if prepared["search_mode"] == "adaptive":
                (
                    lane,
                    parts,
                    part_move,
                    heuristic_rank,
                    sampling_policy,
                    selection_probability,
                ) = selected
            else:
                lane, parts, part_move, heuristic_rank = selected
                sampling_policy = "heur-out"
                selection_probability = 1.0
            if prepared["search_mode"] in {"mixed", "adaptive"} and lane != "spine":
                balanced_step += 1

            first_visit = parts not in attempted
            if first_visit:
                attempted[parts] = []
                states.append({"state_id": labels._state_id(parts), "parts": list(parts)})
            pose_checkpoint, pose_reused = reachable._state_poses(
                prepared, scratch_assembly, parts, pose_generator, pose_factory
            )
            if first_visit:
                current_run[
                    "state_pose_checkpoints_reused"
                    if pose_reused
                    else "state_pose_checkpoints_created"
                ] += 1
                pose_path = reachable._state_dir(prepared, parts) / "poses.json"
                pose_files.append(
                    {
                        "state_id": labels._state_id(parts),
                        "path": pose_path.relative_to(prepared["labels_root"]).as_posix(),
                        "sha256": labels._sha256_file(pose_path),
                    }
                )
            pose_path = reachable._state_dir(prepared, parts) / "poses.json"
            pose_sha256 = labels._sha256_file(pose_path)
            candidate_before = accounting["candidate_count"]
            feasible = False
            evaluated_pose_count = 0
            part_attempt_complete = True
            pose_records = list(pose_checkpoint["poses"])
            if prepared.get("seed_pose_order"):
                pose_records.sort(
                    key=lambda pose: mixed_search.seeded_tie(
                        prepared["parameters"]["seed"],
                        (assembly_dir.name,) + parts,
                        "{}:{}".format(part_move, pose["pose_id"]),
                    )
                )
            time_budget_hit = False
            for pose_position, pose_record in enumerate(pose_records):
                if accounting["candidate_count"] >= prepared["parameters"]["candidate_budget"]:
                    part_attempt_complete = False
                    break
                if isolate_candidates:
                    records, reused = _run_candidate_isolated(
                        prepared,
                        parts,
                        pose_record,
                        pose_sha256,
                        part_move,
                        worker_job_hash,
                        cache_manifest,
                        cache_sha256,
                        source,
                        runtime,
                        asset_folder,
                        scratch_assembly,
                    )
                else:
                    records, reused = reachable._run_candidate(
                        prepared,
                        parts,
                        pose_record,
                        pose_sha256,
                        part_move,
                        worker_job_hash,
                        cache_manifest,
                        cache_sha256,
                        source,
                        runtime,
                        removal_oracle,
                        stability_oracle,
                        scratch_assembly,
                    )
                current_run[
                    "candidate_checkpoints_reused"
                    if reused
                    else "candidate_checkpoints_created"
                ] += 1
                candidate_path = reachable._candidate_path(
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
                evaluated_pose_count += 1
                for field in accounting:
                    if field != "candidate_count":
                        accounting[field] += summary["accounting"].get(field, 0)
                reachable._add_counts(label_counts, summary["label_counts"])
                for field in candidate_timing_s:
                    candidate_timing_s[field] += summary["timings_s"][field]
                feasible = feasible or any(action.get("Y") is True for action in records[2:8])
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
                transition = {
                    "parent_state_id": labels._state_id(parts),
                    "child_state_id": labels._state_id(child),
                    "part_move": part_move,
                }
                transitions.append(transition)
                if child not in discovered:
                    discovered.add(child)
                    if len(child) == 1:
                        terminals.append(
                            {"state_id": labels._state_id(child), "parts": list(child)}
                        )
                if prepared["search_mode"] in {"mixed", "adaptive"} and lane == "spine":
                    spine_state = child if len(child) > 1 else None
                elif prepared["search_mode"] == "dfs" and len(child) > 1:
                    dfs_path.append(child)
            elif lane == "spine" and len(attempted[parts]) == len(parts):
                spine_state = None

            schedule.append(
                {
                    "attempt_index": len(schedule),
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
            )
            if prepared["search_mode"] == "dfs" and feasible and len(child) == 1:
                stop_reason = "solution_found"
                break
            if time_budget_hit:
                stop_reason = "candidate_time_budget_reached"
                break
            if not part_attempt_complete:
                stop_reason = "budget_reached"
                break

        if stop_reason is None:
            stop_reason = "budget_reached"

        if accounting["candidate_count"] > prepared["parameters"]["candidate_budget"]:
            raise RuntimeError("candidate evaluation budget exceeded")
        reachable._verify_scratch_sdf(scratch_assembly, cache_manifest)
        if labels._hash_assembly(scratch_assembly)[0] != prepared["input_sha256"]:
            raise RuntimeError("scratch inputs changed during traversal")
        traversal_wall_s = time.perf_counter() - traversal_started

    source_after, _ = sdf_cache_tools._directory_snapshot(assembly_dir)
    if source_after != source_before:
        raise RuntimeError("authoritative assembly changed during traversal")

    accounting["stability_queries_avoided"] = (
        accounting["candidate_count"] - accounting["stability_queries"]
    )
    accounting["state_count"] = len(discovered)
    accounting["evaluated_state_count"] = len(states)
    accounting["terminal_state_count"] = len(terminals)
    accounting["transition_edge_count"] = len(transitions)
    accounting["state_part_attempt_count"] = len(schedule)
    coverage = _coverage(root, discovered, attempted, transitions, schedule)
    summary = {
        "schema_version": reachable.SCHEMA_VERSION,
        "record_type": "summary",
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_hash": worker_job_hash,
        "phase": prepared["manifest"]["phase"],
        "assembly_id": assembly_dir.name,
        "root_state_id": labels._state_id(root),
        "manifest_sha256": prepared["manifest_sha256"],
        "input": {
            "sha256": prepared["input_sha256"],
            "source_directory_snapshot_sha256": source_before,
            "source_directory_files": source_files,
            "source_directory_unchanged": True,
        },
        "sdf_cache": {
            "cache_key": cache_manifest["cache_key"],
            "manifest_sha256": cache_sha256,
            "file_count": cache_manifest["file_count"],
            "bytes": cache_manifest["bytes"],
            "files_unchanged": True,
        },
        "provenance": {
            "source": source,
            "runtime": runtime,
            "candidate_generator_sha256": labels._sha256_file(Path(reachable.__file__)),
            "scheduler_sha256": prepared["scheduler_sha256"],
        },
        "budget": {
            "state_part_attempt_budget": prepared["parameters"]["state_part_budget"],
            "candidate_evaluation_budget": prepared["parameters"]["candidate_budget"],
            **(
                {
                    "candidate_time_budget_s": prepared["parameters"][
                        "candidate_time_budget_s"
                    ]
                }
                if prepared["parameters"].get("candidate_time_budget_s") is not None
                else {}
            ),
            "stop_reason": stop_reason,
        },
        "accounting": accounting,
        "label_counts": label_counts,
        "coverage": coverage,
        "schedule": schedule,
        "states": states,
        "terminal_states": terminals,
        "transitions": transitions,
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
                reachable._read_jsonl(
                    reachable._state_dir(prepared, state["parts"]) / "poses.json"
                )[0]["stable_pose_search_wall_s"]
                for state in states
            ),
            "candidate_dataset_totals": candidate_timing_s,
            "total_wall_s": time.perf_counter() - started,
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "visible_cpu_count": os.cpu_count(),
        },
    }
    if prepared.get("complexity") is not None:
        summary["complexity"] = dict(
            prepared["complexity"], sdf_cache_bytes=cache_manifest["bytes"]
        )
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
