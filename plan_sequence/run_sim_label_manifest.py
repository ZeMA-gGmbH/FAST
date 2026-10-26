"""Validate and run a frozen simulation-label manifest.

The runner is intentionally thin: it maps one manifest entry to the existing
``generate_sim_labels.run_job`` worker.  ``--index`` maps directly to a
batch array index, while ``--all`` is the sequential local-debug path.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path, PurePosixPath

from plan_sequence import generate_sim_labels as labels


MANIFEST_SCHEMA_VERSION = "fast.sim-label-manifest.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FROZEN_PROTOCOL = {
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
    "full_stability_coverage": True,
    "require_contact_graph": True,
    "removal_timeout_scope": "per_direction",
    "stability_timeout_scope": "per_candidate_greedy_search",
}
_ROOT_FIELDS = {
    "schema_version",
    "label_schema_version",
    "phase",
    "job_count",
    "protocol",
    "parameters",
    "jobs",
}
_PARAMETER_FIELDS = {"max_poses", "pose_probability", "timeout_s", "save_sdf"}
_JOB_FIELDS = {
    "job_id",
    "assembly",
    "part_count",
    "part_ids",
    "input_sha256",
    "output",
}


def _require_fields(value, expected, field):
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "{} fields differ from schema (missing={}, unexpected={})".format(
                field, missing, unexpected
            )
        )


def _read_manifest(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid manifest JSON: {}".format(exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def _component(value, field):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("{} must be one portable path component".format(field))
    return value


def _relative_path(value, field, suffix=None):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("{} must be a non-empty POSIX relative path".format(field))
    path = PurePosixPath(value)
    raw_parts = value.split("/")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("{} must be traversal-free and relative".format(field))
    if suffix is not None and path.suffix != suffix:
        raise ValueError("{} must end in {}".format(field, suffix))
    return Path(*path.parts)


def _is_within(parent, child):
    try:
        common = os.path.commonpath((str(parent), str(child)))
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(str(parent))


def _resolve_relative(root, relative, field):
    target = (root / relative).resolve()
    if not _is_within(root, target) or target == root:
        raise ValueError("{} resolves outside its root".format(field))
    return target


def _positive_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a number".format(field))
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("{} must be finite and positive".format(field))
    return value


def _validate_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    _require_fields(parameters, _PARAMETER_FIELDS, "parameters")
    max_poses = parameters.get("max_poses")
    if type(max_poses) is not int or not 1 <= max_poses <= 3:
        raise ValueError("parameters.max_poses must be an integer from 1 to 3")
    probability = _positive_number(
        parameters.get("pose_probability"), "parameters.pose_probability"
    )
    if probability > 1:
        raise ValueError("parameters.pose_probability must be at most 1")
    save_sdf = parameters.get("save_sdf")
    if not isinstance(save_sdf, bool):
        raise ValueError("parameters.save_sdf must be Boolean")
    return {
        "max_poses": max_poses,
        "pose_probability": probability,
        "timeout": _positive_number(parameters.get("timeout_s"), "parameters.timeout_s"),
        "save_sdf": save_sdf,
    }


def prepare_manifest(manifest_path, source_root, output_root):
    """Validate a manifest and its immutable source assemblies without simulation."""
    manifest_path = Path(manifest_path).resolve()
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("manifest not found: {}".format(manifest_path))
    if not source_root.is_dir():
        raise FileNotFoundError("source root not found: {}".format(source_root))
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("output root is not a directory: {}".format(output_root))
    if _is_within(source_root, output_root) or _is_within(output_root, source_root):
        raise ValueError("source and output roots must not contain one another")

    manifest = _read_manifest(manifest_path)
    _require_fields(manifest, _ROOT_FIELDS, "manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    if manifest.get("label_schema_version") != labels.SCHEMA_VERSION:
        raise ValueError("manifest label_schema_version does not match the worker")
    _component(manifest.get("phase"), "phase")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    _require_fields(protocol, set(_FROZEN_PROTOCOL), "protocol")
    for key, expected in _FROZEN_PROTOCOL.items():
        if key not in protocol or protocol[key] != expected:
            raise ValueError("protocol.{} must be {!r}".format(key, expected))
    parameters = _validate_parameters(manifest.get("parameters"))

    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs must be a non-empty array")
    job_count = manifest.get("job_count")
    if type(job_count) is not int or job_count != len(raw_jobs):
        raise ValueError("job_count must equal the number of jobs")

    seen_ids = set()
    seen_assemblies = set()
    seen_outputs = set()
    jobs = []
    for index, raw in enumerate(raw_jobs):
        field = "jobs[{}]".format(index)
        if not isinstance(raw, dict):
            raise ValueError("{} must be an object".format(field))
        _require_fields(raw, _JOB_FIELDS, field)
        job_id = _component(raw.get("job_id"), field + ".job_id")
        assembly_relative = _relative_path(raw.get("assembly"), field + ".assembly")
        output_relative = _relative_path(raw.get("output"), field + ".output", ".jsonl")
        if job_id in seen_ids:
            raise ValueError("duplicate job_id: {}".format(job_id))
        seen_ids.add(job_id)

        part_count = raw.get("part_count")
        if type(part_count) is not int or part_count < 2:
            raise ValueError("{}.part_count must be an integer of at least 2".format(field))
        part_ids = raw.get("part_ids")
        if (
            not isinstance(part_ids, list)
            or len(part_ids) != part_count
            or any(not isinstance(part, str) or not part for part in part_ids)
            or len(set(part_ids)) != len(part_ids)
            or sorted(part_ids) != part_ids
        ):
            raise ValueError(
                "{}.part_ids must contain part_count unique IDs in ASAP order".format(field)
            )
        expected_hash = raw.get("input_sha256")
        if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
            raise ValueError("{}.input_sha256 must be lowercase SHA-256".format(field))

        assembly_dir = _resolve_relative(source_root, assembly_relative, field + ".assembly")
        output_path = _resolve_relative(output_root, output_relative, field + ".output")
        if assembly_dir in seen_assemblies:
            raise ValueError("duplicate resolved assembly: {}".format(assembly_dir))
        if output_path in seen_outputs:
            raise ValueError("duplicate resolved output: {}".format(output_path))
        seen_assemblies.add(assembly_dir)
        seen_outputs.add(output_path)
        if not assembly_dir.is_dir():
            raise FileNotFoundError("assembly not found: {}".format(assembly_dir))
        if not (assembly_dir / "contact_graph.json").is_file():
            raise FileNotFoundError("contact graph not found: {}".format(assembly_dir))
        actual_parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
        if actual_parts != part_ids:
            raise ValueError("part IDs do not match for {}".format(assembly_dir))
        actual_hash = labels._hash_assembly(assembly_dir)[0]
        if actual_hash != expected_hash:
            raise ValueError("input SHA-256 does not match for {}".format(assembly_dir))

        jobs.append(
            {
                "index": index,
                "job_id": job_id,
                "assembly_dir": assembly_dir,
                "output_path": output_path,
                "part_count": part_count,
                "input_sha256": expected_hash,
            }
        )

    return {
        "path": manifest_path,
        "sha256": labels._sha256_bytes(
            labels._canonical_json(manifest).encode("utf-8")
        ),
        "phase": manifest["phase"],
        "parameters": parameters,
        "jobs": jobs,
    }


def _select_jobs(prepared, index=None, run_all=False):
    if run_all == (index is not None):
        raise ValueError("select exactly one of index or run_all")
    if run_all:
        return prepared["jobs"]
    if type(index) is not int or not 0 <= index < len(prepared["jobs"]):
        raise IndexError("job index out of range: {}".format(index))
    return [prepared["jobs"][index]]


def execute_manifest(
    manifest_path,
    source_root,
    output_root,
    *,
    index=None,
    run_all=False,
    asset_folder=None
):
    """Run one or all validated jobs in manifest order, failing fast."""
    prepared = prepare_manifest(manifest_path, source_root, output_root)
    results = []
    for job in _select_jobs(prepared, index=index, run_all=run_all):
        summary = labels.run_job(
            job["assembly_dir"],
            job["output_path"],
            asset_folder=asset_folder,
            expected_input_sha256=job["input_sha256"],
            manifest_context={
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "sha256": prepared["sha256"],
                "phase": prepared["phase"],
                "job_id": job["job_id"],
                "index": job["index"],
            },
            **prepared["parameters"]
        )
        results.append(
            {
                "manifest_sha256": prepared["sha256"],
                "phase": prepared["phase"],
                "index": job["index"],
                "job_id": job["job_id"],
                "summary": summary,
            }
        )
    return results


def _parser():
    parser = argparse.ArgumentParser(
        description="Validate or run a frozen FAST simulation-label manifest."
    )
    parser.add_argument(
        "--manifest", required=True
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--asset-folder", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=int, help="zero-based job index (batch array index)")
    mode.add_argument("--all", action="store_true", help="run every job sequentially")
    mode.add_argument(
        "--validate-only", action="store_true", help="validate inputs without simulation"
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.validate_only:
        prepared = prepare_manifest(args.manifest, args.source_root, args.output_root)
        print(
            labels._canonical_json(
                {
                    "status": "validated",
                    "manifest_sha256": prepared["sha256"],
                    "phase": prepared["phase"],
                    "job_count": len(prepared["jobs"]),
                }
            ),
            flush=True,
        )
        return 0
    results = execute_manifest(
        args.manifest,
        args.source_root,
        args.output_root,
        index=args.index,
        run_all=args.all,
        asset_folder=args.asset_folder,
    )
    for result in results:
        print(labels._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
