"""Prepare and validate an immutable RedMax SDF cache for one assembly."""

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from plan_sequence import generate_sim_labels as labels


SCHEMA_VERSION = "fast.sdf-cache.v1"
SDF_SETTINGS = {
    "body_type": "SDF",
    "dx": 0.05,
    "load_sdf": True,
    "res": 20,
    "save_sdf": True,
    "transform_type": "OBJ_TO_JOINT",
}


def _peak_rss_mb():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)
    except (ImportError, OSError):
        return None


def _directory_snapshot(directory):
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    return labels._hash_files(directory, paths)


def _cache_identity(repo_root, input_sha256, parts, simulator_sha256):
    sim_string = repo_root / "plan_sequence" / "sim_string.py"
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "part_ids": list(parts),
        "generator": {
            "model": "contact",
            "settings": SDF_SETTINGS,
            "sim_string_sha256": labels._sha256_file(sim_string),
        },
        "simulator_sha256": simulator_sha256,
    }


def validate_cache(
    cache_dir,
    *,
    expected_input_sha256=None,
    expected_parts=None,
    expected_simulator_sha256=None,
    expected_sim_string_sha256=None,
    require_keyed_directory=True,
):
    """Validate a published cache and return its manifest."""
    cache_dir = Path(cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("SDF cache manifest not found: {}".format(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("invalid SDF cache manifest: {}".format(exc)) from exc

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported SDF cache schema")
    if manifest.get("complete") is not True:
        raise ValueError("SDF cache is not complete")
    input_sha256 = manifest.get("input", {}).get("sha256")
    if expected_input_sha256 is not None and input_sha256 != expected_input_sha256:
        raise ValueError("SDF cache input SHA-256 does not match the assembly")
    if require_keyed_directory and cache_dir.name != input_sha256:
        raise ValueError("SDF cache directory is not keyed by its input SHA-256")

    parts = manifest.get("part_ids")
    if not isinstance(parts, list) or not parts or any(not isinstance(part, str) for part in parts):
        raise ValueError("SDF cache has invalid part IDs")
    if parts != sorted(parts):
        raise ValueError("SDF cache part IDs are not sorted")
    if expected_parts is not None and parts != list(expected_parts):
        raise ValueError("SDF cache part IDs do not match the assembly")

    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("SDF cache identity is missing")
    expected_identity_hash = labels._sha256_bytes(
        labels._canonical_json(identity).encode("utf-8")
    )
    if manifest.get("cache_key") != expected_identity_hash:
        raise ValueError("SDF cache identity hash is invalid")
    if identity.get("input_sha256") != input_sha256 or identity.get("part_ids") != parts:
        raise ValueError("SDF cache identity disagrees with its manifest")
    if identity.get("generator", {}).get("settings") != SDF_SETTINGS:
        raise ValueError("SDF cache settings are incompatible")
    if (
        expected_simulator_sha256 is not None
        and identity.get("simulator_sha256") != expected_simulator_sha256
    ):
        raise ValueError("SDF cache RedMax SHA-256 does not match the runtime")
    if (
        expected_sim_string_sha256 is not None
        and identity.get("generator", {}).get("sim_string_sha256")
        != expected_sim_string_sha256
    ):
        raise ValueError("SDF cache generator source does not match the runtime")

    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or len(files) != len(parts)
        or any(not isinstance(entry, dict) for entry in files)
    ):
        raise ValueError("SDF cache file count does not match its parts")
    expected_names = ["{}.sdf".format(part) for part in parts]
    if [entry.get("path") for entry in files] != expected_names:
        raise ValueError("SDF cache file list does not match its parts")
    actual_names = sorted(path.name for path in cache_dir.glob("*.sdf") if path.is_file())
    if actual_names != expected_names:
        raise ValueError("SDF cache contains an unexpected SDF file set")
    total_bytes = 0
    for entry in files:
        path = cache_dir / entry["path"]
        if not path.is_file():
            raise FileNotFoundError("SDF cache file not found: {}".format(path))
        size = path.stat().st_size
        if size != entry.get("size") or labels._sha256_file(path) != entry.get("sha256"):
            raise ValueError("SDF cache file failed integrity check: {}".format(path))
        total_bytes += size
    if manifest.get("file_count") != len(files) or manifest.get("bytes") != total_bytes:
        raise ValueError("SDF cache aggregate counts are invalid")
    return manifest


def _publish_cache(cache_dir, scratch_assembly, manifest):
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".{}-".format(cache_dir.name), dir=str(cache_dir.parent))
    )
    try:
        for entry in manifest["files"]:
            shutil.copy2(
                str(scratch_assembly / entry["path"]), str(stage / entry["path"])
            )
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
            expected_simulator_sha256=manifest["identity"]["simulator_sha256"],
            expected_sim_string_sha256=manifest["identity"]["generator"][
                "sim_string_sha256"
            ],
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
                expected_simulator_sha256=manifest["identity"]["simulator_sha256"],
                expected_sim_string_sha256=manifest["identity"]["generator"][
                    "sim_string_sha256"
                ],
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
    simulation_factory=None,
):
    """Generate every part SDF in scratch and atomically publish the cache."""
    from plan_sequence.sim_string import get_contact_sim_string

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    repo_root = Path(__file__).resolve().parents[1]
    assembly_dir = Path(assembly_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    if not assembly_dir.is_dir():
        raise FileNotFoundError("assembly directory not found: {}".format(assembly_dir))
    labels._validate_output_target(assembly_dir.parent, cache_dir)

    parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    input_sha256, input_files = labels._hash_assembly(assembly_dir)
    if expected_input_sha256 is not None and input_sha256 != expected_input_sha256:
        raise ValueError("assembly input SHA-256 does not match expected value")
    if cache_dir.name != input_sha256:
        raise ValueError("cache directory name must equal the assembly input SHA-256")

    runtime = labels._runtime_provenance()
    simulator_sha256 = runtime["simulator"]["sha256"]
    if simulation_factory is None and simulator_sha256 is None:
        raise RuntimeError("redmax_py is unavailable or cannot be hashed")
    identity = _cache_identity(repo_root, input_sha256, parts, simulator_sha256)
    source = labels._source_provenance(repo_root)
    cache_key = labels._sha256_bytes(labels._canonical_json(identity).encode("utf-8"))
    if cache_dir.exists():
        manifest = validate_cache(
            cache_dir,
            expected_input_sha256=input_sha256,
            expected_parts=parts,
            expected_simulator_sha256=simulator_sha256,
            expected_sim_string_sha256=identity["generator"]["sim_string_sha256"],
        )
        if manifest["cache_key"] != cache_key:
            raise FileExistsError("existing SDF cache belongs to a different specification")
        result = dict(manifest)
        result["idempotent"] = True
        return result

    source_tree_before, source_tree_files = _directory_snapshot(assembly_dir)
    timings = {}
    with tempfile.TemporaryDirectory(
        prefix="fast-sdf-cache-{}-".format(assembly_dir.name)
    ) as scratch:
        scratch_assembly = Path(scratch) / assembly_dir.name
        scratch_assembly.mkdir()
        copy_started = time.perf_counter()
        for source_path in labels._assembly_input_paths(assembly_dir):
            shutil.copy2(str(source_path), str(scratch_assembly / source_path.name))
        timings["copy_inputs_wall_s"] = time.perf_counter() - copy_started
        if labels._hash_assembly(scratch_assembly)[0] != input_sha256:
            raise RuntimeError("scratch input hash differs from source")

        build_started = time.perf_counter()
        model = get_contact_sim_string(str(scratch_assembly), parts, save_sdf=True)
        if simulation_factory is None:
            import redmax_py as redmax

            redmax.Simulation(model, str(repo_root / "assets"))
        else:
            simulation_factory(model, str(repo_root / "assets"))
        timings["redmax_initialization_wall_s"] = time.perf_counter() - build_started

        sdf_paths = [scratch_assembly / "{}.sdf".format(part) for part in parts]
        missing = [path.name for path in sdf_paths if not path.is_file()]
        if missing:
            raise RuntimeError("RedMax did not generate SDF files: {}".format(", ".join(missing)))
        unexpected = sorted(
            path.name
            for path in scratch_assembly.glob("*.sdf")
            if path.name not in {item.name for item in sdf_paths}
        )
        if unexpected:
            raise RuntimeError("RedMax generated unexpected SDF files: {}".format(", ".join(unexpected)))
        files = [
            {
                "path": path.name,
                "sha256": labels._sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in sdf_paths
        ]

        source_tree_after, _ = _directory_snapshot(assembly_dir)
        if source_tree_after != source_tree_before:
            raise RuntimeError("authoritative assembly changed during SDF generation")
        timings["total_wall_s_before_publish"] = time.perf_counter() - started_wall
        timings["total_cpu_s_before_publish"] = time.process_time() - started_cpu
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "assembly_id": assembly_dir.name,
            "cache_key": cache_key,
            "identity": identity,
            "part_ids": parts,
            "file_count": len(files),
            "bytes": sum(entry["size"] for entry in files),
            "files": files,
            "input": {"sha256": input_sha256, "files": input_files},
            "source_directory": {
                "snapshot_sha256": source_tree_before,
                "files": source_tree_files,
                "unchanged": True,
            },
            "runtime": runtime,
            "provenance": {"source": source},
            "host": {"hostname": socket.gethostname(), "visible_cpu_count": os.cpu_count()},
            "peak_rss_mb": _peak_rss_mb(),
            "timings_s": timings,
        }
        return _publish_cache(cache_dir, scratch_assembly, manifest)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--expected-input-sha256")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = prepare_cache(
        args.assembly_dir,
        args.cache_dir,
        expected_input_sha256=args.expected_input_sha256,
    )
    print(labels._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
