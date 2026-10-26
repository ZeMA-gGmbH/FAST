"""Build, run, and audit the frozen simulation-label production."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "fast.production-label-manifest.v1"
AUDIT_SCHEMA_VERSION = "fast.production-label-audit.v1"
JOB_COLUMNS = (
    "split",
    "assembly_id",
    "tier",
    "part_count",
    "input_sha256",
    "worker_memory",
    "worker_disk",
)
SOURCE_SPLITS = ["training_assembly", "test_assembly"]
COMMON_PARAMETERS = {
    "max_poses": 3,
    "pose_probability": 0.9,
    "removal_timeout_s": 60.0,
    "stability_timeout_s": 900.0,
    "save_sdf": True,
    "seed": 0,
}
TIER_SPECS = {
    "small": {
        "part_counts": [3, 6],
        "generator": "exhaustive-reachable",
        "resources": {"memory": "4G", "disk": "15G"},
    },
    "medium": {
        "part_counts": [7, 20],
        "generator": "adaptive-mixed",
        "candidate_time_budget_s": 3600.0,
        "state_part_attempt_multiplier": 2,
        "candidate_evaluation_multiplier": 6,
        "resources": {"memory": "8G", "disk": "30G"},
    },
    "large": {
        "part_counts": [21, 70],
        "generator": "asap-dfs-heur-out",
        "state_part_attempt_budget": 400,
        "candidate_evaluation_budget": 400,
        "resources": {"memory": "12G", "disk": "50G"},
    },
}
OUTPUT_PATHS = {
    "cache_root": "cache/sdf",
    "labels_root": "labels/fast-labels-v1",
    "audit_root": "runs/fast-labels-v1/assemblies",
    "combined_audit": "runs/fast-labels-v1/audit.json",
}
EXCLUDED_ASSEMBLIES = [
    {
        "split": "training_assembly",
        "assembly_id": assembly_id,
        "reason": "repeatable simulator SIGSEGV in production and retry",
    }
    for assembly_id in (
        "00155",
        "02033",
        "02337",
        "04183",
        "04490",
        "05288",
        "05480",
        "05810",
        "05838",
        "05847",
        "07588",
        "08073",
    )
] + [
    {
        "split": "test_assembly",
        "assembly_id": assembly_id,
        "reason": "repeatable simulator SIGSEGV in production and retry",
    }
    for assembly_id in ("04489", "07802")
]


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_files(root, paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_assembly(assembly_dir):
    paths = [path for path in assembly_dir.glob("*.obj") if path.is_file()]
    for name in ("contact_graph.json", "config.json"):
        path = assembly_dir / name
        if path.is_file():
            paths.append(path)
    if not any(path.suffix.lower() == ".obj" for path in paths):
        raise ValueError("assembly contains no OBJ parts: {}".format(assembly_dir))
    return _hash_files(assembly_dir, paths)


def _require_fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("{} fields must be exactly {}".format(name, sorted(expected)))


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("{} must be lowercase SHA-256".format(name))
    return value


def _relative_path(value, name):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("{} must be a POSIX repository-relative path".format(name))
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("{} must be traversal-free".format(name))
    return path


def _is_within(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _tier(part_count):
    for name, spec in TIER_SPECS.items():
        if spec["part_counts"][0] <= part_count <= spec["part_counts"][1]:
            return name
    raise ValueError("assembly part count is outside the approved 3-to-70 range")


def _read_spec(manifest_path, *, verify_inventory=True):
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("production manifest not found: {}".format(manifest_path))
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    _require_fields(
        manifest,
        {
            "schema_version",
            "phase",
            "source_splits",
            "common_parameters",
            "tiers",
            "inventory",
            "paths",
            "known_duplicate_groups",
            "excluded_assemblies",
        },
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported production manifest schema")
    if manifest["phase"] != "fast-labels-v1-production":
        raise ValueError("production phase must be fast-labels-v1-production")
    if manifest["source_splits"] != SOURCE_SPLITS:
        raise ValueError("production source splits differ from the approved splits")
    if manifest["common_parameters"] != COMMON_PARAMETERS:
        raise ValueError("production common parameters differ from the approved protocol")
    if manifest["tiers"] != TIER_SPECS:
        raise ValueError("production tier definitions differ from the approved protocol")
    if manifest["paths"] != OUTPUT_PATHS:
        raise ValueError("production output paths differ from the approved layout")
    if not isinstance(manifest["known_duplicate_groups"], list):
        raise ValueError("known_duplicate_groups must be a list")
    if manifest["excluded_assemblies"] != EXCLUDED_ASSEMBLIES:
        raise ValueError("production exclusions differ from the documented failures")

    inventory = manifest["inventory"]
    _require_fields(inventory, {"path", "sha256", "expected"}, "inventory")
    inventory_relative = _relative_path(inventory["path"], "inventory.path")
    inventory_sha256 = _sha256(inventory["sha256"], "inventory.sha256")
    expected = inventory["expected"]
    _require_fields(expected, {"total", "per_split"}, "inventory.expected")
    if set(expected["per_split"]) != set(SOURCE_SPLITS):
        raise ValueError("inventory expected splits are invalid")
    for split in SOURCE_SPLITS:
        _require_fields(
            expected["per_split"][split],
            {"total", "small", "medium", "large"},
            "inventory.expected.per_split.{}".format(split),
        )

    repository_root = manifest_path.parent.parent
    inventory_path = (repository_root / inventory_relative).resolve()
    if not _is_within(inventory_path, repository_root):
        raise ValueError("inventory.path escapes the repository")
    if verify_inventory:
        if not inventory_path.is_file():
            raise FileNotFoundError("production inventory not found: {}".format(inventory_path))
        if _sha256_file(inventory_path) != inventory_sha256:
            raise ValueError("production inventory SHA-256 differs from the manifest")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "inventory_path": inventory_path,
        "inventory_sha256": inventory_sha256,
    }


def _read_jobs(prepared):
    jobs = []
    with prepared["inventory_path"].open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError("blank production inventory row at line {}".format(line_number))
            values = line.split("\t")
            if len(values) != len(JOB_COLUMNS):
                raise ValueError("invalid production inventory row at line {}".format(line_number))
            row = dict(zip(JOB_COLUMNS, values))
            if row["split"] not in SOURCE_SPLITS:
                raise ValueError("invalid split at inventory line {}".format(line_number))
            if not row["assembly_id"].isdigit():
                raise ValueError("invalid assembly ID at inventory line {}".format(line_number))
            try:
                row["part_count"] = int(row["part_count"])
            except ValueError:
                raise ValueError("invalid part count at inventory line {}".format(line_number))
            if row["tier"] != _tier(row["part_count"]):
                raise ValueError("tier mismatch at inventory line {}".format(line_number))
            _sha256(row["input_sha256"], "inventory input_sha256")
            resources = TIER_SPECS[row["tier"]]["resources"]
            if (row["worker_memory"], row["worker_disk"]) != (
                resources["memory"],
                resources["disk"],
            ):
                raise ValueError("resource mismatch at inventory line {}".format(line_number))
            jobs.append(row)
    expected_order = sorted(
        jobs, key=lambda row: (SOURCE_SPLITS.index(row["split"]), row["assembly_id"])
    )
    if jobs != expected_order:
        raise ValueError("production inventory rows are not in canonical order")
    identities = [(row["split"], row["assembly_id"]) for row in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError("production inventory contains duplicate assemblies")
    if _job_counts(jobs) != prepared["manifest"]["inventory"]["expected"]:
        raise ValueError("production inventory counts differ from the manifest")
    if _duplicate_groups(jobs) != prepared["manifest"]["known_duplicate_groups"]:
        raise ValueError("production duplicate groups differ from the manifest")
    return jobs


def _job_counts(jobs):
    per_split = {}
    for split in SOURCE_SPLITS:
        rows = [row for row in jobs if row["split"] == split]
        per_split[split] = {
            "total": len(rows),
            "small": sum(row["tier"] == "small" for row in rows),
            "medium": sum(row["tier"] == "medium" for row in rows),
            "large": sum(row["tier"] == "large" for row in rows),
        }
    return {"total": len(jobs), "per_split": per_split}


def _duplicate_groups(jobs):
    by_hash = {}
    for row in jobs:
        by_hash.setdefault(row["input_sha256"], []).append(
            [row["split"], row["assembly_id"]]
        )
    return [
        {"input_sha256": input_sha256, "members": members}
        for input_sha256, members in sorted(by_hash.items())
        if len(members) > 1
    ]


def build_inventory(manifest_path, source_root):
    prepared = _read_spec(manifest_path, verify_inventory=False)
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError("ASAP source dataset not found: {}".format(source_root))
    excluded = {
        (entry["split"], entry["assembly_id"])
        for entry in prepared["manifest"]["excluded_assemblies"]
    }
    lines = []
    jobs = []
    for split in SOURCE_SPLITS:
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError("source split not found: {}".format(split_root))
        for assembly_dir in sorted(
            (path for path in split_root.iterdir() if path.is_dir()), key=lambda path: path.name
        ):
            if (split, assembly_dir.name) in excluded:
                continue
            if not assembly_dir.name.isdigit():
                raise ValueError("assembly ID must be numeric: {}".format(assembly_dir))
            part_ids = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
            if not (assembly_dir / "contact_graph.json").is_file():
                raise FileNotFoundError("contact graph not found: {}".format(assembly_dir))
            tier = _tier(len(part_ids))
            resources = TIER_SPECS[tier]["resources"]
            row = {
                "split": split,
                "assembly_id": assembly_dir.name,
                "tier": tier,
                "part_count": len(part_ids),
                "input_sha256": _hash_assembly(assembly_dir),
                "worker_memory": resources["memory"],
                "worker_disk": resources["disk"],
            }
            jobs.append(row)
            lines.append("\t".join(str(row[column]) for column in JOB_COLUMNS))
    if _job_counts(jobs) != prepared["manifest"]["inventory"]["expected"]:
        raise ValueError("live dataset counts differ from the production manifest")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    output = prepared["inventory_path"]
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different inventory: {}".format(output))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(".{}.{}.tmp".format(output.name, os.getpid()))
        temporary.write_bytes(payload)
        os.replace(str(temporary), str(output))
    return {
        "path": str(output),
        "sha256": _sha256_bytes(payload),
        **_job_counts(jobs),
        "known_duplicate_groups": _duplicate_groups(jobs),
    }


def _validate_live_job(row, source_root):
    source_root = Path(source_root).resolve()
    assembly_dir = (source_root / row["split"] / row["assembly_id"]).resolve()
    if not _is_within(assembly_dir, source_root) or not assembly_dir.is_dir():
        raise FileNotFoundError("assembly not found: {}".format(assembly_dir))
    if not (assembly_dir / "contact_graph.json").is_file():
        raise FileNotFoundError("contact graph not found: {}".format(assembly_dir))
    part_ids = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    if len(part_ids) != row["part_count"]:
        raise ValueError("assembly part count differs from the production inventory")
    if _hash_assembly(assembly_dir) != row["input_sha256"]:
        raise ValueError("assembly input SHA-256 differs from the production inventory")
    return assembly_dir, part_ids


def validate(manifest_path, source_root, output_root):
    prepared = _read_spec(manifest_path)
    jobs = _read_jobs(prepared)
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError("ASAP source dataset not found: {}".format(source_root))
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("output root is not a directory: {}".format(output_root))
    if _is_within(source_root, output_root) or _is_within(output_root, source_root):
        raise ValueError("source and output roots must not contain one another")
    for row in jobs:
        _validate_live_job(row, source_root)
    return prepared, jobs


def _protocols():
    from plan_sequence import generate_adaptive_sim_labels as adaptive
    from plan_sequence import generate_greedy_sim_labels as greedy
    from plan_sequence import generate_reachable_sim_labels as reachable

    return {
        "small": reachable._PROTOCOL,
        "medium": adaptive._PROTOCOL,
        "large": greedy._PROTOCOL,
    }


def _job_manifest(spec, row, part_ids, protocols=None):
    protocols = protocols or _protocols()
    common = spec["common_parameters"]
    part_count = row["part_count"]
    base = {
        "label_schema_version": "fast.reachable-sim-labels.v1",
        "phase": spec["phase"],
        "protocol": protocols[row["tier"]],
        "assembly": {
            "path": "{}/{}".format(row["split"], row["assembly_id"]),
            "part_count": part_count,
            "part_ids": part_ids,
            "input_sha256": row["input_sha256"],
        },
        "cache": "{}/{}".format(spec["paths"]["cache_root"], row["input_sha256"]),
        "output_root": "{}/{}/{}".format(
            spec["paths"]["labels_root"], row["split"], row["assembly_id"]
        ),
        "audit_output": "{}/{}/{}.json".format(
            spec["paths"]["audit_root"], row["split"], row["assembly_id"]
        ),
    }
    if row["tier"] == "small":
        max_states = 2 ** part_count - 1
        return {
            "schema_version": "fast.reachable-label-manifest.v1",
            **base,
            "parameters": {
                "max_poses": common["max_poses"],
                "pose_probability": common["pose_probability"],
                "removal_timeout_s": common["removal_timeout_s"],
                "stability_timeout_s": common["stability_timeout_s"],
                "save_sdf": common["save_sdf"],
                "max_states": max_states,
            },
            "root_reference": "runs/fast-labels-v1/root-reference-not-used/{}/{}".format(
                row["split"], row["assembly_id"]
            ),
            "expected": {
                "root_candidate_count": common["max_poses"] * part_count,
                "root_action_count": common["max_poses"] * part_count * 6,
                "theoretical_max_nonempty_states": max_states,
            },
        }
    if row["tier"] == "medium":
        tier = spec["tiers"]["medium"]
        return {
            "schema_version": "fast.adaptive-label-manifest.v1",
            **base,
            "parameters": {
                **common,
                "candidate_time_budget_s": tier["candidate_time_budget_s"],
            },
            "expected": {
                "state_part_attempt_budget": tier["state_part_attempt_multiplier"]
                * (part_count - 1),
                "candidate_evaluation_budget": tier["candidate_evaluation_multiplier"]
                * (part_count - 1),
            },
        }
    tier = spec["tiers"]["large"]
    return {
        "schema_version": "fast.greedy-label-manifest.v1",
        **base,
        "parameters": dict(common),
        "expected": {
            "state_part_attempt_budget": tier["state_part_attempt_budget"],
            "candidate_evaluation_budget": tier["candidate_evaluation_budget"],
        },
    }


def _select_job(prepared, split, assembly_id):
    jobs = _read_jobs(prepared)
    matches = [
        row for row in jobs if row["split"] == split and row["assembly_id"] == assembly_id
    ]
    if len(matches) != 1:
        raise ValueError("assembly is not uniquely present in the production inventory")
    return matches[0]


@contextmanager
def _derived_manifest_path(manifest):
    with tempfile.TemporaryDirectory(prefix="fast-label-manifest-") as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        yield path


def run_job(manifest_path, source_root, output_root, split, assembly_id, action, asset_folder=None):
    prepared = _read_spec(manifest_path)
    row = _select_job(prepared, split, assembly_id)
    _, part_ids = _validate_live_job(row, source_root)
    job_manifest = _job_manifest(prepared["manifest"], row, part_ids)
    with _derived_manifest_path(job_manifest) as derived_path:
        if action == "labels":
            from plan_sequence import generate_adaptive_sim_labels as adaptive
            from plan_sequence import generate_budgeted_sim_labels as budgeted
            from plan_sequence import generate_greedy_sim_labels as greedy
            from plan_sequence import generate_reachable_sim_labels as reachable

            if row["tier"] == "small":
                job = reachable._validate_manifest(derived_path, source_root, output_root)
                return reachable.run(job, asset_folder=asset_folder)
            validator = adaptive if row["tier"] == "medium" else greedy
            job = validator._validate_manifest(derived_path, source_root, output_root)
            return budgeted.run(job, asset_folder=asset_folder)
        if action == "audit":
            if row["tier"] == "small":
                from plan_sequence import audit_reachable_sim_labels as audit

                return audit.audit(
                    derived_path, source_root, output_root, check_root_reference=False
                )
            if row["tier"] == "medium":
                from plan_sequence import audit_adaptive_sim_labels as audit
            else:
                from plan_sequence import audit_greedy_sim_labels as audit
            return audit.audit(derived_path, source_root, output_root)
    raise ValueError("unsupported production action: {}".format(action))


def _read_single(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        lines = [line for line in stream if line.strip()]
    if len(lines) != 1:
        raise ValueError("artifact must contain exactly one JSON record: {}".format(path))
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise ValueError("artifact must contain a JSON object: {}".format(path))
    return value


def combined_audit(manifest_path, source_root, output_root):
    prepared = _read_spec(manifest_path)
    jobs = _read_jobs(prepared)
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    audit_root = output_root / prepared["manifest"]["paths"]["audit_root"]
    expected_paths = {
        (audit_root / row["split"] / (row["assembly_id"] + ".json")).resolve()
        for row in jobs
    }
    actual_paths = (
        {path.resolve() for path in audit_root.rglob("*.json")} if audit_root.is_dir() else set()
    )
    if actual_paths != expected_paths:
        missing = len(expected_paths - actual_paths)
        unexpected = len(actual_paths - expected_paths)
        raise FileNotFoundError(
            "production audits are incomplete: {} missing, {} unexpected".format(
                missing, unexpected
            )
        )

    schemas = {
        "small": "fast.reachable-label-audit.v1",
        "medium": "fast.adaptive-label-audit.v1",
        "large": "fast.greedy-label-audit.v1",
    }
    counts = {
        key: {
            "job_count": 0,
            "status_passed": 0,
            "status_review_required": 0,
            "complete_sequence_count": 0,
            "censored_action_count": 0,
            "censored_queried_stability_count": 0,
            "label_counts": {
                name: {"true": 0, "false": 0, "null": 0} for name in ("R", "T", "Y")
            },
        }
        for key in SOURCE_SPLITS + list(TIER_SPECS)
    }
    audit_hashes = []
    integrity_passed = True
    for row in jobs:
        _, part_ids = _validate_live_job(row, source_root)
        logical_manifest = _job_manifest(prepared["manifest"], row, part_ids)
        expected_manifest_sha256 = _sha256_bytes(
            _canonical_json(logical_manifest).encode("utf-8")
        )
        path = audit_root / row["split"] / (row["assembly_id"] + ".json")
        report = _read_single(path)
        if (
            report.get("schema_version") != schemas[row["tier"]]
            or report.get("phase") != prepared["manifest"]["phase"]
            or report.get("manifest_sha256") != expected_manifest_sha256
        ):
            raise ValueError("production audit identity mismatch: {}".format(path))
        integrity_passed = integrity_passed and report.get("integrity_status") == "passed"
        audit_hashes.append(
            {
                "split": row["split"],
                "assembly_id": row["assembly_id"],
                "sha256": _sha256_file(path),
            }
        )
        complete_sequence = report.get("coverage", {}).get("complete_sequence_found")
        if complete_sequence is None:
            complete_sequence = bool(report.get("terminal_states"))
        censoring = report.get("censoring", {})
        for key in (row["split"], row["tier"]):
            item = counts[key]
            item["job_count"] += 1
            item["status_passed"] += report.get("status") == "passed"
            item["status_review_required"] += report.get("status") == "review_required"
            item["complete_sequence_count"] += complete_sequence is True
            item["censored_action_count"] += censoring.get("action_count", 0)
            item["censored_queried_stability_count"] += censoring.get(
                "queried_stability_count", 0
            )
            for label in ("R", "T", "Y"):
                for value in ("true", "false", "null"):
                    item["label_counts"][label][value] += report["label_counts"][label][
                        value
                    ]

    spec = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "manifest_sha256": prepared["manifest_sha256"],
        "inventory_sha256": prepared["inventory_sha256"],
        "audit_sha256": audit_hashes,
        "auditor_sha256": _sha256_file(Path(__file__)),
    }
    spec_sha256 = _sha256_bytes(_canonical_json(spec).encode("utf-8"))
    output = output_root / prepared["manifest"]["paths"]["combined_audit"]
    if output.exists():
        existing = _read_single(output)
        if existing.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise RuntimeError("existing combined production audit has the wrong schema")
        if existing.get("spec_sha256") != spec_sha256:
            raise FileExistsError("existing combined production audit belongs to another run")
        existing["idempotent"] = True
        return existing
    policy_checks = {
        "all_jobs_audited": len(audit_hashes) == len(jobs),
        "all_integrity_passed": integrity_passed,
        "inventory_count_matched": len(jobs)
        == prepared["manifest"]["inventory"]["expected"]["total"],
    }
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": spec_sha256,
        "status": "passed" if all(policy_checks.values()) else "review_required",
        "integrity_status": "passed" if integrity_passed else "review_required",
        "phase": prepared["manifest"]["phase"],
        "manifest_sha256": prepared["manifest_sha256"],
        "inventory_sha256": prepared["inventory_sha256"],
        "expected": prepared["manifest"]["inventory"]["expected"],
        "known_duplicate_groups": prepared["manifest"]["known_duplicate_groups"],
        "excluded_assemblies": prepared["manifest"]["excluded_assemblies"],
        "unseen_test_exclusions": [
            {
                "split": "test_assembly",
                "assembly_id": "07386",
                "reason": "exact input duplicate of training_assembly/07385",
            }
        ],
        "policy_checks": policy_checks,
        "per_split": {split: counts[split] for split in SOURCE_SPLITS},
        "per_tier": {tier: counts[tier] for tier in TIER_SPECS},
        "audit_count": len(audit_hashes),
    }
    from plan_sequence import generate_sim_labels as labels

    try:
        labels._write_atomic(output, [report])
    except FileExistsError:
        return combined_audit(manifest_path, source_root, output_root)
    return report


def _parser():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(root / "configs" / "label_production.json")
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--asset-folder")
    parser.add_argument("--split", choices=SOURCE_SPLITS)
    parser.add_argument("--assembly-id")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--build-inventory", action="store_true")
    action.add_argument("--labels", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--combined-audit", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.build_inventory:
        result = build_inventory(args.manifest, args.source_root)
    elif args.validate_only:
        prepared, jobs = validate(args.manifest, args.source_root, args.output_root)
        result = {
            "status": "validated",
            "phase": prepared["manifest"]["phase"],
            "manifest_sha256": prepared["manifest_sha256"],
            **_job_counts(jobs),
        }
    elif args.combined_audit:
        result = combined_audit(args.manifest, args.source_root, args.output_root)
    else:
        if args.split is None or args.assembly_id is None:
            raise SystemExit("--split and --assembly-id are required for labels and audit")
        result = run_job(
            args.manifest,
            args.source_root,
            args.output_root,
            args.split,
            args.assembly_id,
            "labels" if args.labels else "audit",
            asset_folder=args.asset_folder,
        )
    print(_canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
