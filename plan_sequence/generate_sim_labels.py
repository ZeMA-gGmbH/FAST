"""Generate root-state ASAP simulation labels for one assembly.

The same command is intentionally usable as a local process or as one
batch job.  Authoritative inputs are copied to job-local scratch before
any ASAP/RedMax object is constructed, because RedMax may create SDF caches
beside mesh files.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "fast.sim-labels.v1"
A6 = (
    ("+Z", (0, 0, 1)),
    ("-Z", (0, 0, -1)),
    ("+X", (1, 0, 0)),
    ("-X", (-1, 0, 0)),
    ("+Y", (0, 1, 0)),
    ("-Y", (0, -1, 0)),
)
_RECORD_TYPES = {"header", "stability", "action", "summary"}


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_files(root, paths):
    entries = []
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256_file(path)
        entries.append({"path": relative, "sha256": file_hash, "size": path.stat().st_size})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), entries


def _assembly_input_paths(assembly_dir):
    """Return only authoritative ASAP inputs; never reuse simulator caches."""
    paths = sorted(path for path in assembly_dir.glob("*.obj") if path.is_file())
    for name in ("contact_graph.json", "config.json"):
        path = assembly_dir / name
        if path.is_file():
            paths.append(path)
    if not any(path.suffix.lower() == ".obj" for path in paths):
        raise ValueError("assembly directory contains no OBJ parts: {}".format(assembly_dir))
    return paths


def _hash_assembly(assembly_dir):
    return _hash_files(assembly_dir, _assembly_input_paths(assembly_dir))


def _source_provenance(repo_root):
    relative_paths = (
        Path("plan_sequence/generate_sim_labels.py"),
        Path("plan_sequence/asap_label_oracle.py"),
        Path("plan_sequence/physics_planner.py"),
        Path("plan_sequence/feasibility_check.py"),
        Path("plan_sequence/sim_string.py"),
        Path("plan_sequence/stable_pose.py"),
        Path("assets/load.py"),
        Path("assets/transform.py"),
    )
    paths = [repo_root / path for path in relative_paths if (repo_root / path).is_file()]
    source_hash, files = _hash_files(repo_root, paths)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        commit = result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"sha256": source_hash, "files": files, "git_commit": commit}


def _distribution_version(name):
    try:
        from importlib import metadata
    except ImportError:
        try:
            import importlib_metadata as metadata
        except ImportError:
            return None
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _runtime_provenance():
    simulator = {"module": "redmax_py", "available": False, "origin": None, "sha256": None}
    try:
        spec = importlib.util.find_spec("redmax_py")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None:
        simulator["available"] = True
        simulator["origin"] = spec.origin
        if spec.origin and Path(spec.origin).is_file():
            simulator["sha256"] = _sha256_file(Path(spec.origin))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "numpy": _distribution_version("numpy"),
            "scipy": _distribution_version("scipy"),
            "trimesh": _distribution_version("trimesh"),
        },
        "simulator": simulator,
    }


def _state_id(parts):
    payload = _canonical_json(sorted(parts)).encode("utf-8")
    return "state-{}".format(_sha256_bytes(payload)[:16])


def _compute_stable_poses(assembly_dir, parts, max_poses, probability_threshold):
    """Use ASAP's stable-pose implementation without importing RedMax."""
    from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses

    mesh = get_combined_mesh(str(assembly_dir), parts)
    poses, probabilities = get_stable_poses(
        mesh,
        prob_th=probability_threshold,
        max_num=max_poses,
        return_prob=True,
    )
    return list(zip(poses, probabilities))


def _load_oracle(asset_folder, assembly_dir, timeout, save_sdf):
    """Import the RedMax-backed adapter only when a real job starts."""
    from plan_sequence.asap_label_oracle import ASAPLabelOracle

    return ASAPLabelOracle(
        str(asset_folder),
        str(assembly_dir),
        timeout=timeout,
        save_sdf=save_sdf,
    )


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


def _error_result(exc, runtime_s):
    return {
        "label": None,
        "status": "error",
        "reason": "simulator_error",
        "rollout_count": 0,
        "runtime_s": runtime_s,
        "forward_steps": None,
        "path_steps": None,
        "parts_fix": None,
        "rollouts": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _normalise_result(result, default_rollouts=1):
    if not isinstance(result, dict):
        raise TypeError("oracle result must be a dictionary")
    label = result.get("label")
    if label is not None and not isinstance(label, bool):
        raise TypeError("oracle label must be bool or None")
    status = result.get("status")
    if status is None:
        status = "success" if label is True else "failure" if label is False else "error"
    if status in {"timeout", "error"} and label is not None:
        raise ValueError("timeout/error oracle results must have label=None")
    rollout_count = int(result.get("rollout_count", default_rollouts))
    if rollout_count < 0:
        raise ValueError("oracle rollout_count cannot be negative")
    return {
        "label": label,
        "status": str(status),
        "reason": _json_value(result.get("reason")),
        "rollout_count": rollout_count,
        "runtime_s": float(result.get("runtime_s", 0.0)),
        "error": _json_value(result.get("error")),
        "parts_fix": _json_value(result.get("parts_fix")),
        "forward_steps": _json_value(result.get("forward_steps")),
        "path_steps": _json_value(result.get("path_steps", result.get("path_length"))),
        "rollouts": _json_value(result.get("rollouts")),
    }


def _direction_key(value):
    try:
        return tuple(int(round(float(component))) for component in value)
    except (TypeError, ValueError):
        return None


def _normalise_directions(batch_result):
    by_vector = {}
    if isinstance(batch_result, dict):
        raw_directions = batch_result.get("directions", [])
    else:
        raw_directions = []
    for raw in raw_directions:
        if not isinstance(raw, dict):
            continue
        vector = _direction_key(raw.get("action", raw.get("direction")))
        if vector is not None and vector not in by_vector:
            try:
                by_vector[vector] = _normalise_result(raw)
            except (TypeError, ValueError) as exc:
                by_vector[vector] = _error_result(exc, 0.0)

    results = []
    for direction_id, vector in A6:
        result = by_vector.get(vector)
        if result is None:
            result = {
                "label": None,
                "status": "error",
                "rollout_count": 0,
                "runtime_s": 0.0,
                "reason": "missing_direction_result",
                "error": {
                    "type": "MissingDirectionResult",
                    "message": "oracle did not return {}".format(direction_id),
                },
                "parts_fix": None,
                "forward_steps": None,
                "path_steps": None,
                "rollouts": None,
            }
        results.append((direction_id, vector, result))
    return results


def _label_counts():
    return {"true": 0, "false": 0, "null": 0}


def _count_label(counts, label):
    counts["true" if label is True else "false" if label is False else "null"] += 1


def _combined_label(removable, stable, removable_status, stable_status):
    if removable is False:
        return False, removable_status
    if stable is False:
        return False, stable_status
    if removable is None:
        return None, removable_status
    if stable is None:
        return None, stable_status
    return bool(stable), "success" if stable else "failure"


def _validate_output_target(source_split_dir, output_path):
    try:
        common = os.path.commonpath((str(source_split_dir), str(output_path)))
    except ValueError:
        return
    if os.path.normcase(common) == os.path.normcase(str(source_split_dir)):
        raise ValueError("output must not be inside the authoritative source split")


def _existing_complete(output_path, expected_job_hash):
    if not output_path.exists():
        return None
    first = None
    last = None
    count = 0
    seen_summary = False
    try:
        with output_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError("blank line at {}".format(line_number))
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("record_type") not in _RECORD_TYPES:
                    raise ValueError("invalid record at {}".format(line_number))
                if record.get("job_hash") != expected_job_hash:
                    raise FileExistsError(
                        "existing output belongs to a different job: {}".format(output_path)
                    )
                if seen_summary:
                    raise ValueError("records found after summary")
                seen_summary = record["record_type"] == "summary"
                if first is None:
                    first = record
                last = record
                count += 1
    except FileExistsError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("existing output is corrupt: {} ({})".format(output_path, exc)) from exc

    if (
        first is None
        or first.get("record_type") != "header"
        or last.get("record_type") != "summary"
        or last.get("complete") is not True
        or last.get("record_count") != count
    ):
        raise RuntimeError("existing output is incomplete or corrupt: {}".format(output_path))
    result = dict(last)
    result["idempotent"] = True
    return result


def _write_atomic(output_path, records):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}-".format(output_path.name),
            suffix=".tmp",
            dir=str(output_path.parent),
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for record in records:
                stream.write(_canonical_json(record))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without overwriting a shard produced by another job.
        os.link(str(temporary_path), str(output_path))
        temporary_path.unlink()
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def run_job(
    assembly_dir,
    output_path,
    *,
    asset_folder=None,
    max_poses=3,
    pose_probability=0.9,
    timeout=60.0,
    save_sdf=True,
    oracle=None,
    expected_input_sha256=None,
    manifest_context=None,
):
    """Generate one complete JSONL shard and return its summary record."""
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    assembly_dir = Path(assembly_dir).resolve()
    output_path = Path(output_path).resolve()
    asset_folder = Path(asset_folder).resolve() if asset_folder else repo_root / "assets"

    if not assembly_dir.is_dir():
        raise FileNotFoundError("assembly directory not found: {}".format(assembly_dir))
    if not asset_folder.is_dir():
        raise FileNotFoundError("ASAP asset folder not found: {}".format(asset_folder))
    if not 1 <= int(max_poses) <= 3:
        raise ValueError("max_poses must be between 1 and 3")
    if not 0.0 < float(pose_probability) <= 1.0:
        raise ValueError("pose_probability must be in (0, 1]")
    if timeout is not None and float(timeout) <= 0:
        raise ValueError("timeout must be positive")
    _validate_output_target(assembly_dir.parent, output_path)

    parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    if len(parts) < 2:
        raise ValueError("assembly must contain at least two OBJ parts")

    input_hash, input_files = _hash_assembly(assembly_dir)
    if expected_input_sha256 is not None and input_hash != expected_input_sha256:
        raise ValueError(
            "assembly input SHA-256 does not match expected value: {}".format(assembly_dir)
        )
    source = _source_provenance(repo_root)
    runtime = _runtime_provenance()
    assembly_id = assembly_dir.name
    oracle_backend = (
        "plan_sequence.asap_label_oracle.ASAPLabelOracle"
        if oracle is None
        else "{}.{}".format(type(oracle).__module__, type(oracle).__qualname__)
    )
    protocol = {
        "state_scope": "root",
        "action_space": "A6_world",
        "directions": [
            {"id": direction_id, "vector": list(vector)} for direction_id, vector in A6
        ],
        "gravity": [0, 0, -1],
        "support": {"mode": "ground_plane", "normal": [0, 0, 1], "z": 0},
        "max_grippers": 2,
        "base_part": None,
        "max_holding_parts": 1,
        "stability_oracle": "ASAP-greedy",
        "oracle_backend": oracle_backend,
        "full_stability_coverage": True,
        "max_poses": int(max_poses),
        "pose_probability_threshold": float(pose_probability),
        "removal_timeout_s": 60.0 if timeout is None else float(timeout),
        "removal_timeout_scope": "per_direction",
        "stability_timeout_s": None if timeout is None else float(timeout),
        "stability_timeout_scope": "per_candidate_greedy_search",
        "save_sdf_in_scratch": bool(save_sdf),
    }
    if manifest_context is not None:
        if not isinstance(manifest_context, dict):
            raise TypeError("manifest_context must be a dictionary or None")
        manifest_context = _json_value(manifest_context)
        _canonical_json(manifest_context)
    job_spec = {
        "schema_version": SCHEMA_VERSION,
        "assembly_id": assembly_id,
        "input_sha256": input_hash,
        "source_sha256": source["sha256"],
        "runtime": {
            "python": runtime["python"],
            "platform": runtime["platform"],
            "dependencies": runtime["dependencies"],
            "simulator_available": runtime["simulator"]["available"],
            "simulator_sha256": runtime["simulator"]["sha256"],
        },
        "parts": parts,
        "protocol": protocol,
    }
    if manifest_context is not None:
        job_spec["manifest"] = manifest_context
    job_hash = _sha256_bytes(_canonical_json(job_spec).encode("utf-8"))
    existing = _existing_complete(output_path, job_hash)
    if existing is not None:
        return existing

    records = []
    accounting = {
        "candidate_actions": 0,
        "removability_batches": 0,
        "removability_rollouts": 0,
        "stability_queries": 0,
        "stability_rollouts": 0,
        "oracle_rollouts_total": 0,
    }
    label_counts = {"R": _label_counts(), "T": _label_counts(), "Y": _label_counts()}
    root_state_id = _state_id(parts)

    with tempfile.TemporaryDirectory(prefix="fast-label-{}-".format(assembly_id)) as scratch:
        scratch_assembly = Path(scratch) / assembly_id
        scratch_assembly.mkdir()
        for source_path in _assembly_input_paths(assembly_dir):
            shutil.copy2(str(source_path), str(scratch_assembly / source_path.name))
        scratch_hash, _ = _hash_assembly(scratch_assembly)
        if scratch_hash != input_hash:
            raise RuntimeError(
                "assembly inputs changed while creating the scratch snapshot: {}".format(
                    assembly_dir
                )
            )
        poses = _compute_stable_poses(
            scratch_assembly, parts, int(max_poses), float(pose_probability)
        )
        if not poses:
            # Match SequencePlanner: an empty stable-pose search uses identity via pose=None.
            poses = [(None, None)]

        active_oracle = (
            oracle
            if oracle is not None
            else _load_oracle(asset_folder, scratch_assembly, timeout, bool(save_sdf))
        )
        records.append(
            {
                "record_type": "header",
                "schema_version": SCHEMA_VERSION,
                "job_hash": job_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "assembly_id": assembly_id,
                "split": assembly_dir.parent.name,
                "root_state_id": root_state_id,
                "parts": parts,
                "input": {
                    "path": str(assembly_dir),
                    "sha256": input_hash,
                    "files": input_files,
                    "config_present": (assembly_dir / "config.json").is_file(),
                    "transform_source": (
                        "config_final_state"
                        if (assembly_dir / "config.json").is_file()
                        else "baked_obj_identity"
                    ),
                    "part_order": "lexicographic_asap_v1",
                },
                "protocol": protocol,
                "manifest": manifest_context,
                "provenance": {
                    "source": source,
                    "runtime": runtime,
                },
            }
        )

        for pose_index, (pose, probability) in enumerate(poses):
            pose_id = "pose-{:03d}".format(pose_index)
            pose_json = _json_value(pose)
            pose_source = "stable_pose" if pose is not None else "identity_fallback"
            for part_move in parts:
                parts_after = [part for part in parts if part != part_move]
                child_state_id = _state_id(parts_after)
                stability_id = "{}:{}:{}".format(pose_id, child_state_id, part_move)

                stability_started = time.perf_counter()
                try:
                    stability = _normalise_result(
                        active_oracle.evaluate_stability(parts_after, pose)
                    )
                except Exception as exc:  # simulator failures are censored, not negatives
                    stability = _error_result(exc, time.perf_counter() - stability_started)
                accounting["stability_queries"] += 1
                accounting["stability_rollouts"] += stability["rollout_count"]
                _count_label(label_counts["T"], stability["label"])
                records.append(
                    {
                        "record_type": "stability",
                        "schema_version": SCHEMA_VERSION,
                        "job_hash": job_hash,
                        "assembly_id": assembly_id,
                        "root_state_id": root_state_id,
                        "state_id": child_state_id,
                        "stability_id": stability_id,
                        "pose_id": pose_id,
                        "pose_probability": None if probability is None else float(probability),
                        "pose": pose_json,
                        "pose_source": pose_source,
                        "removed_part": part_move,
                        "parts": parts_after,
                        "T": stability["label"],
                        "T_status": stability["status"],
                        "reason": stability["reason"],
                        "parts_fix": stability.get("parts_fix"),
                        "forward_steps": stability.get("forward_steps"),
                        "rollouts": stability.get("rollouts"),
                        "oracle_rollout_count": stability["rollout_count"],
                        "runtime_s": stability["runtime_s"],
                        "error": stability["error"],
                    }
                )

                removability_started = time.perf_counter()
                try:
                    batch = active_oracle.evaluate_removability(parts, part_move, pose)
                    directions = _normalise_directions(batch)
                except Exception as exc:  # one failed batch censors all six directions
                    error = _error_result(exc, time.perf_counter() - removability_started)
                    directions = [
                        (direction_id, vector, dict(error)) for direction_id, vector in A6
                    ]
                accounting["removability_batches"] += 1

                for direction_id, vector, removable in directions:
                    r_label = removable["label"]
                    t_label = stability["label"]
                    y_label, y_status = _combined_label(
                        r_label, t_label, removable["status"], stability["status"]
                    )
                    accounting["candidate_actions"] += 1
                    accounting["removability_rollouts"] += removable["rollout_count"]
                    _count_label(label_counts["R"], r_label)
                    _count_label(label_counts["Y"], y_label)
                    records.append(
                        {
                            "record_type": "action",
                            "schema_version": SCHEMA_VERSION,
                            "job_hash": job_hash,
                            "assembly_id": assembly_id,
                            "root_state_id": root_state_id,
                            "child_state_id": child_state_id,
                            "pose_id": pose_id,
                            "pose_probability": None if probability is None else float(probability),
                            "pose_source": pose_source,
                            "part_move": part_move,
                            "parts_fixed": parts_after,
                            "direction_id": direction_id,
                            "direction": list(vector),
                            "direction_frame": "world",
                            "stability_ref": stability_id,
                            "R": r_label,
                            "R_status": removable["status"],
                            "R_reason": removable["reason"],
                            "T": t_label,
                            "T_status": stability["status"],
                            "Y": y_label,
                            "Y_status": y_status,
                            "oracle_rollout_count": removable["rollout_count"],
                            "runtime_s": removable["runtime_s"],
                            "forward_steps": removable.get("forward_steps"),
                            "path_steps": removable.get("path_steps"),
                            "error": removable["error"],
                        }
                    )

    accounting["oracle_rollouts_total"] = (
        accounting["removability_rollouts"] + accounting["stability_rollouts"]
    )
    accounting["legacy_asap_part_evaluations"] = accounting["removability_batches"]
    summary = {
        "record_type": "summary",
        "schema_version": SCHEMA_VERSION,
        "job_hash": job_hash,
        "assembly_id": assembly_id,
        "complete": True,
        "pose_count": len(poses),
        "accounting": accounting,
        "label_counts": label_counts,
        "runtime_s": time.perf_counter() - started,
        "record_count": len(records) + 1,
    }
    records.append(summary)
    try:
        _write_atomic(output_path, records)
    except FileExistsError:
        # A concurrent worker may have completed the same immutable shard.
        existing = _existing_complete(output_path, job_hash)
        if existing is not None:
            return existing
        raise
    return summary


def _parser():
    parser = argparse.ArgumentParser(
        description="Generate one root-state ASAP simulation-label JSONL shard."
    )
    parser.add_argument("--assembly-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-folder", default=None)
    parser.add_argument("--max-poses", type=int, default=3)
    parser.add_argument("--pose-probability", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--disable-save-sdf",
        dest="save_sdf",
        action="store_false",
        help="disable RedMax SDF caching inside per-job scratch",
    )
    parser.set_defaults(save_sdf=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    summary = run_job(
        args.assembly_dir,
        args.output,
        asset_folder=args.asset_folder,
        max_poses=args.max_poses,
        pose_probability=args.pose_probability,
        timeout=args.timeout,
        save_sdf=args.save_sdf,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
