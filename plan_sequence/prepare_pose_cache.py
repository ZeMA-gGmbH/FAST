"""Prepare and validate an immutable stable-pose cache for one assembly."""

import argparse
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_sdf_cache as sdf_cache_tools


SCHEMA_VERSION = "fast.stable-pose-cache.v1"


def _generator_identity(repo_root, runtime):
    source_paths = [
        repo_root / "plan_sequence" / "stable_pose.py",
        repo_root / "assets" / "load.py",
        repo_root / "assets" / "transform.py",
    ]
    source_sha256, source_files = labels._hash_files(repo_root, source_paths)
    return {
        "source_sha256": source_sha256,
        "source_files": source_files,
        "python": runtime["python"],
        "dependencies": runtime["dependencies"],
    }


def _cache_identity(input_sha256, parts, max_poses, probability_threshold, generator):
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "part_ids": list(parts),
        "parameters": {
            "max_poses": max_poses,
            "probability_threshold": probability_threshold,
        },
        "generator": generator,
    }


def _valid_matrix(value):
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
        and all(
            not isinstance(number, bool)
            and isinstance(number, (int, float))
            and math.isfinite(number)
            for row in value
            for number in row
        )
    )


def validate_cache(
    cache_dir,
    *,
    expected_input_sha256=None,
    expected_parts=None,
    expected_max_poses=None,
    expected_probability_threshold=None,
    expected_generator=None,
    require_keyed_directory=True,
):
    """Validate a published stable-pose cache and return its manifest."""
    cache_dir = Path(cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("stable-pose cache manifest not found: {}".format(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("invalid stable-pose cache manifest: {}".format(exc)) from exc

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported stable-pose cache schema")
    if manifest.get("complete") is not True:
        raise ValueError("stable-pose cache is not complete")
    input_sha256 = manifest.get("input", {}).get("sha256")
    if expected_input_sha256 is not None and input_sha256 != expected_input_sha256:
        raise ValueError("stable-pose cache input SHA-256 does not match the assembly")
    if require_keyed_directory and cache_dir.name != input_sha256:
        raise ValueError("stable-pose cache directory is not keyed by its input SHA-256")

    parts = manifest.get("part_ids")
    if (
        not isinstance(parts, list)
        or not parts
        or parts != sorted(parts)
        or any(not isinstance(part, str) or not part for part in parts)
    ):
        raise ValueError("stable-pose cache has invalid part IDs")
    if expected_parts is not None and parts != list(expected_parts):
        raise ValueError("stable-pose cache part IDs do not match the assembly")

    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("stable-pose cache identity is missing")
    identity_hash = labels._sha256_bytes(
        labels._canonical_json(identity).encode("utf-8")
    )
    if manifest.get("cache_key") != identity_hash:
        raise ValueError("stable-pose cache identity hash is invalid")
    if identity.get("input_sha256") != input_sha256 or identity.get("part_ids") != parts:
        raise ValueError("stable-pose cache identity disagrees with its manifest")
    parameters = identity.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("stable-pose cache parameters are missing")
    max_poses = parameters.get("max_poses")
    threshold = parameters.get("probability_threshold")
    if type(max_poses) is not int or not 1 <= max_poses <= 3:
        raise ValueError("stable-pose cache max_poses is invalid")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError("stable-pose cache probability threshold is invalid")
    if expected_max_poses is not None and max_poses != expected_max_poses:
        raise ValueError("stable-pose cache max_poses does not match")
    if (
        expected_probability_threshold is not None
        and threshold != expected_probability_threshold
    ):
        raise ValueError("stable-pose cache probability threshold does not match")
    if expected_generator is not None and identity.get("generator") != expected_generator:
        raise ValueError("stable-pose cache generator does not match the runtime")

    poses = manifest.get("poses")
    if not isinstance(poses, list) or not 1 <= len(poses) <= max_poses:
        raise ValueError("stable-pose cache pose count is invalid")
    for index, entry in enumerate(poses):
        if not isinstance(entry, dict):
            raise ValueError("stable-pose cache pose entry is invalid")
        expected_id = "pose-{:03d}".format(index)
        if entry.get("index") != index or entry.get("pose_id") != expected_id:
            raise ValueError("stable-pose cache pose identity is invalid")
        pose = entry.get("pose")
        probability = entry.get("probability")
        source = entry.get("source")
        if source == "identity_fallback":
            if pose is not None or probability is not None or len(poses) != 1:
                raise ValueError("stable-pose cache identity fallback is invalid")
        elif source == "stable_pose":
            if not _valid_matrix(pose):
                raise ValueError("stable-pose cache matrix is invalid")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("stable-pose cache probability is invalid")
        else:
            raise ValueError("stable-pose cache pose source is invalid")
    if manifest.get("pose_count") != len(poses):
        raise ValueError("stable-pose cache aggregate count is invalid")
    actual_files = sorted(path.name for path in cache_dir.iterdir() if path.is_file())
    if actual_files != ["manifest.json"]:
        raise ValueError("stable-pose cache contains unexpected files")
    return manifest


def _publish_cache(cache_dir, manifest):
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".{}-".format(cache_dir.name), dir=str(cache_dir.parent))
    )
    try:
        manifest_path = stage / "manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(labels._canonical_json(manifest))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        validate_cache(
            stage,
            expected_input_sha256=manifest["input"]["sha256"],
            expected_parts=manifest["part_ids"],
            expected_max_poses=manifest["identity"]["parameters"]["max_poses"],
            expected_probability_threshold=manifest["identity"]["parameters"][
                "probability_threshold"
            ],
            expected_generator=manifest["identity"]["generator"],
            require_keyed_directory=False,
        )
        try:
            os.rename(str(stage), str(cache_dir))
        except OSError:
            if not cache_dir.exists():
                raise
            return validate_cache(
                cache_dir,
                expected_input_sha256=manifest["input"]["sha256"],
                expected_parts=manifest["part_ids"],
                expected_max_poses=manifest["identity"]["parameters"]["max_poses"],
                expected_probability_threshold=manifest["identity"]["parameters"][
                    "probability_threshold"
                ],
                expected_generator=manifest["identity"]["generator"],
            )
        return manifest
    finally:
        if stage.exists():
            if stage.parent.resolve() != cache_dir.parent.resolve():
                raise RuntimeError("refusing to remove an unexpected staging directory")
            shutil.rmtree(str(stage))


def prepare_cache(
    assembly_dir,
    cache_dir,
    *,
    expected_input_sha256=None,
    max_poses=3,
    probability_threshold=0.9,
    pose_factory=None,
):
    """Compute stable poses once in scratch and atomically publish them."""
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    repo_root = Path(__file__).resolve().parents[1]
    assembly_dir = Path(assembly_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    if not assembly_dir.is_dir():
        raise FileNotFoundError("assembly directory not found: {}".format(assembly_dir))
    labels._validate_output_target(assembly_dir.parent, cache_dir)
    if type(max_poses) is not int or not 1 <= max_poses <= 3:
        raise ValueError("max_poses must be an integer from 1 to 3")
    if (
        isinstance(probability_threshold, bool)
        or not isinstance(probability_threshold, (int, float))
        or not 0 < probability_threshold <= 1
    ):
        raise ValueError("probability_threshold must be in (0, 1]")

    parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    input_sha256, input_files = labels._hash_assembly(assembly_dir)
    if expected_input_sha256 is not None and input_sha256 != expected_input_sha256:
        raise ValueError("assembly input SHA-256 does not match expected value")
    if cache_dir.name != input_sha256:
        raise ValueError("cache directory name must equal the assembly input SHA-256")

    runtime = labels._runtime_provenance()
    generator = _generator_identity(repo_root, runtime)
    identity = _cache_identity(
        input_sha256, parts, max_poses, probability_threshold, generator
    )
    cache_key = labels._sha256_bytes(labels._canonical_json(identity).encode("utf-8"))
    if cache_dir.exists():
        manifest = validate_cache(
            cache_dir,
            expected_input_sha256=input_sha256,
            expected_parts=parts,
            expected_max_poses=max_poses,
            expected_probability_threshold=probability_threshold,
            expected_generator=generator,
        )
        if manifest["cache_key"] != cache_key:
            raise FileExistsError(
                "existing stable-pose cache belongs to a different specification"
            )
        result = dict(manifest)
        result["idempotent"] = True
        return result

    source_tree_before, source_tree_files = sdf_cache_tools._directory_snapshot(
        assembly_dir
    )
    with tempfile.TemporaryDirectory(
        prefix="fast-pose-cache-{}-".format(assembly_dir.name)
    ) as scratch:
        scratch_assembly = Path(scratch) / assembly_dir.name
        scratch_assembly.mkdir()
        copy_started = time.perf_counter()
        for source_path in labels._assembly_input_paths(assembly_dir):
            shutil.copy2(str(source_path), str(scratch_assembly / source_path.name))
        copy_wall_s = time.perf_counter() - copy_started
        if labels._hash_assembly(scratch_assembly)[0] != input_sha256:
            raise RuntimeError("scratch input hash differs from source")

        pose_started = time.perf_counter()
        if pose_factory is None:
            poses = labels._compute_stable_poses(
                scratch_assembly, parts, max_poses, probability_threshold
            )
        else:
            poses = pose_factory(
                scratch_assembly, parts, max_poses, probability_threshold
            )
        pose_wall_s = time.perf_counter() - pose_started
        if not poses:
            poses = [(None, None)]
        pose_records = []
        for index, (pose, probability) in enumerate(poses):
            pose_records.append(
                {
                    "index": index,
                    "pose_id": "pose-{:03d}".format(index),
                    "pose": labels._json_value(pose),
                    "probability": None if probability is None else float(probability),
                    "source": "stable_pose" if pose is not None else "identity_fallback",
                }
            )

    source_tree_after, _ = sdf_cache_tools._directory_snapshot(assembly_dir)
    if source_tree_after != source_tree_before:
        raise RuntimeError("authoritative assembly changed during stable-pose generation")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assembly_id": assembly_dir.name,
        "cache_key": cache_key,
        "identity": identity,
        "part_ids": parts,
        "pose_count": len(pose_records),
        "poses": pose_records,
        "input": {"sha256": input_sha256, "files": input_files},
        "source_directory": {
            "snapshot_sha256": source_tree_before,
            "files": source_tree_files,
            "unchanged": True,
        },
        "runtime": runtime,
        "provenance": {"source": labels._source_provenance(repo_root)},
        "host": {"hostname": socket.gethostname(), "visible_cpu_count": os.cpu_count()},
        "timings_s": {
            "copy_inputs_wall_s": copy_wall_s,
            "stable_pose_search_wall_s": pose_wall_s,
            "total_wall_s_before_publish": time.perf_counter() - started_wall,
            "total_cpu_s_before_publish": time.process_time() - started_cpu,
        },
    }
    return _publish_cache(cache_dir, manifest)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--max-poses", type=int, default=3)
    parser.add_argument("--pose-probability", type=float, default=0.9)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = prepare_cache(
        args.assembly_dir,
        args.cache_dir,
        expected_input_sha256=args.expected_input_sha256,
        max_poses=args.max_poses,
        probability_threshold=args.pose_probability,
    )
    print(labels._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
