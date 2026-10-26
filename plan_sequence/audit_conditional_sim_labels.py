"""Audit conditional root-state candidate shards against invariants and the baseline root labels."""

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from plan_sequence import generate_conditional_sim_labels as conditional
from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_pose_cache as pose_cache_tools


SCHEMA_VERSION = "fast.conditional-label-audit.v2"
POSE_PROBABILITY_ABS_TOLERANCE = 1e-12
POSE_PROBABILITY_REL_TOLERANCE = 0.0
_TIMING_FIELDS = (
    "copy_inputs_wall_s",
    "copy_sdf_cache_wall_s",
    "stable_pose_search_wall_s",
    "removability_wall_s",
    "stability_wall_s",
    "total_wall_s",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_jsonl(path):
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        raise RuntimeError("invalid JSONL {}: {}".format(path, exc)) from exc
    _require(records, "empty JSONL: {}".format(path))
    return records


def _count(counts, value):
    counts["true" if value is True else "false" if value is False else "null"] += 1


def _runtime_contract(runtime):
    _require(isinstance(runtime, dict), "runtime provenance must be an object")
    dependencies = runtime.get("dependencies")
    simulator = runtime.get("simulator")
    _require(isinstance(dependencies, dict), "runtime dependencies must be an object")
    _require(isinstance(simulator, dict), "runtime simulator must be an object")
    return {
        "python": runtime.get("python"),
        "dependencies": dependencies,
        "simulator": {
            "module": simulator.get("module"),
            "available": simulator.get("available"),
            "sha256": simulator.get("sha256"),
        },
    }


def _timing_summary(values):
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _existing_report(path, spec_sha256):
    if not path.exists():
        return None
    records = _read_jsonl(path)
    _require(len(records) == 1, "existing audit must contain one JSON record")
    if records[0].get("spec_sha256") != spec_sha256:
        raise FileExistsError("existing audit belongs to a different specification")
    return records[0]


def _same_probability(left, right):
    if left is None or right is None:
        return left is right
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return False
    return math.isclose(
        left,
        right,
        rel_tol=POSE_PROBABILITY_REL_TOLERANCE,
        abs_tol=POSE_PROBABILITY_ABS_TOLERANCE,
    )


def _semantic_mismatches(records, reference, location):
    mismatches = []
    header, stability = records[0], records[1]
    reference_header, reference_stability = reference[0], reference[1]
    if header.get("pose") != reference_header.get("pose") or not _same_probability(
        header.get("pose_probability"), reference_header.get("pose_probability")
    ):
        mismatches.append("{}: pose differs from the conditional root reference".format(location))
    for field in ("queried", "T", "T_status"):
        if stability.get(field) != reference_stability.get(field):
            mismatches.append(
                "{}: stability {} differs from the conditional root reference".format(location, field)
            )
    for action, reference_action in zip(records[2:8], reference[2:8]):
        if action.get("direction_id") != reference_action.get("direction_id"):
            mismatches.append("{}: action order differs from the conditional root reference".format(location))
            continue
        for field in ("R", "R_status", "T", "T_status", "Y", "Y_status"):
            if action.get(field) != reference_action.get(field):
                mismatches.append(
                    "{} {}: {} differs from the conditional root reference".format(
                        location, action.get("direction_id"), field
                    )
                )
    return mismatches


def audit(manifest_path, source_root, output_root, audit_output=None):
    prepared = conditional._validate_manifest(manifest_path, source_root, output_root)
    if audit_output is None:
        audit_path = prepared["audit_output"]
    else:
        relative = conditional.manifest_tools._relative_path(
            audit_output, "audit_output", ".json"
        )
        audit_path = conditional.manifest_tools._resolve_relative(
            Path(output_root).resolve(), relative, "audit_output"
        )
        if audit_path == prepared["audit_output"]:
            raise ValueError("audit output override must differ from the manifest path")
    pose_cache_manifest = None
    if prepared["pose_cache_dir"] is not None:
        runtime = labels._runtime_provenance()
        pose_cache_manifest = pose_cache_tools.validate_cache(
            prepared["pose_cache_dir"],
            expected_input_sha256=prepared["input_sha256"],
            expected_parts=prepared["part_ids"],
            expected_max_poses=prepared["parameters"]["max_poses"],
            expected_probability_threshold=prepared["parameters"][
                "pose_probability"
            ],
            expected_generator=pose_cache_tools._generator_identity(
                Path(__file__).resolve().parents[1], runtime
            ),
        )
    baseline_records = _read_jsonl(prepared["baseline_path"])
    _require(baseline_records[0].get("record_type") == "header", "invalid baseline header")
    _require(
        baseline_records[-1].get("record_type") == "summary"
        and baseline_records[-1].get("complete") is True,
        "baseline is incomplete",
    )
    baseline_action_records = [
        record for record in baseline_records if record.get("record_type") == "action"
    ]
    baseline_stability_records = [
        record for record in baseline_records if record.get("record_type") == "stability"
    ]
    baseline_actions = {
        (record["pose_id"], record["part_move"], record["direction_id"]): record
        for record in baseline_action_records
    }
    baseline_stability = {
        (record["pose_id"], record["removed_part"]): record
        for record in baseline_stability_records
    }
    _require(
        len(baseline_actions) == len(prepared["candidates"]) * len(labels.A6),
        "baseline action coverage is incomplete",
    )
    _require(
        len(baseline_stability) == len(prepared["candidates"]),
        "baseline stability coverage is incomplete",
    )

    mismatches = []
    files = []
    label_counts = {
        "R": labels._label_counts(),
        "T": labels._label_counts(),
        "Y": labels._label_counts(),
    }
    accounting = {
        "candidate_actions": 0,
        "removability_batches": 0,
        "removability_rollouts": 0,
        "stability_queries": 0,
        "stability_rollouts": 0,
        "oracle_rollouts_total": 0,
    }
    timing_values = {field: [] for field in _TIMING_FIELDS}
    reference_cache = None
    reference_source = None
    reference_runtime = None
    reference_runtime_contract = None
    reference_pose_cache = None
    reference_files = []
    candidates_with_removal = 0

    for candidate in prepared["candidates"]:
        path = candidate["output_path"]
        if not path.is_file():
            raise FileNotFoundError("candidate shard not found: {}".format(path))
        records = _read_jsonl(path)
        _require(len(records) == 9, "candidate shard must contain 9 records: {}".format(path))
        _require(
            [record.get("record_type") for record in records]
            == ["header", "stability"] + ["action"] * 6 + ["summary"],
            "candidate record order is invalid: {}".format(path),
        )
        _require(
            all(
                record.get("schema_version") == prepared["label_schema_version"]
                for record in records
            ),
            "candidate label schema is invalid: {}".format(path),
        )
        header, stability, summary = records[0], records[1], records[-1]
        location = path.relative_to(prepared["labels_root"]).as_posix()
        job_hash = header.get("job_hash")
        _require(
            all(record.get("job_hash") == job_hash for record in records),
            "candidate job hashes disagree: {}".format(path),
        )
        _require(
            summary.get("complete") is True and summary.get("record_count") == len(records),
            "candidate summary is incomplete: {}".format(path),
        )
        _require(
            header.get("candidate_index") == candidate["index"]
            and header.get("pose_index") == candidate["pose_index"]
            and header.get("part_move") == candidate["part_move"],
            "candidate identity is invalid: {}".format(path),
        )
        _require(
            header.get("manifest", {}).get("sha256") == prepared["manifest_sha256"],
            "candidate manifest hash is invalid: {}".format(path),
        )
        _require(
            header.get("input", {}).get("sha256") == prepared["input_sha256"]
            and header.get("input", {}).get("source_directory_unchanged") is True,
            "candidate source input is invalid: {}".format(path),
        )
        cache = header.get("sdf_cache")
        _require(
            isinstance(cache, dict)
            and cache.get("file_count") == len(prepared["part_ids"])
            and cache.get("files_unchanged") is True,
            "candidate cache evidence is invalid: {}".format(path),
        )
        source = header.get("provenance", {}).get("source")
        runtime = header.get("provenance", {}).get("runtime")
        runtime_contract = _runtime_contract(runtime)
        if reference_cache is None:
            reference_cache, reference_source, reference_runtime = cache, source, runtime
            reference_runtime_contract = runtime_contract
        _require(cache == reference_cache, "candidate cache provenance differs")
        _require(source == reference_source, "candidate source provenance differs")
        _require(
            runtime_contract == reference_runtime_contract,
            "candidate runtime contract differs",
        )
        if prepared["pose_cache_dir"] is not None:
            pose_cache = header.get("stable_pose_cache")
            _require(
                isinstance(pose_cache, dict)
                and pose_cache.get("manifest_sha256")
                == labels._sha256_file(prepared["pose_cache_dir"] / "manifest.json")
                and pose_cache.get("pose_count") == prepared["parameters"]["max_poses"],
                "candidate stable-pose cache evidence is invalid: {}".format(path),
            )
            if reference_pose_cache is None:
                reference_pose_cache = pose_cache
            _require(
                pose_cache == reference_pose_cache,
                "candidate stable-pose cache provenance differs",
            )

        actions = records[2:-1]
        _require(
            [(record.get("direction_id"), tuple(record.get("direction", []))) for record in actions]
            == [(name, vector) for name, vector in labels.A6],
            "candidate directions are invalid: {}".format(path),
        )
        has_removal = any(record.get("R") is True for record in actions)
        _require(
            stability.get("queried") is has_removal,
            "conditional stability policy was violated: {}".format(path),
        )
        if has_removal:
            candidates_with_removal += 1
            _require(
                stability.get("T_status") != "not_evaluated",
                "queried stability is marked not evaluated: {}".format(path),
            )
        else:
            _require(
                stability.get("T") is None
                and stability.get("T_status") == "not_evaluated"
                and stability.get("oracle_rollout_count") == 0,
                "skipped stability record is invalid: {}".format(path),
            )

        baseline_t = baseline_stability.get((header["pose_id"], header["part_move"]))
        _require(baseline_t is not None, "baseline stability record is missing")
        if header.get("pose") != baseline_t.get("pose") or not _same_probability(
            header.get("pose_probability"), baseline_t.get("pose_probability")
        ):
            mismatches.append("{}: pose differs from the baseline root labels".format(location))
        if has_removal and (
            stability.get("T") != baseline_t.get("T")
            or stability.get("T_status") != baseline_t.get("T_status")
        ):
            mismatches.append("{}: conditional T differs from the baseline root labels".format(location))

        _count(label_counts["T"], stability.get("T"))
        for action in actions:
            _require(
                action.get("stability_ref") == stability.get("stability_id")
                and action.get("T") == stability.get("T")
                and action.get("T_status") == stability.get("T_status")
                and action.get("T_queried") is has_removal,
                "action stability reference is invalid: {}".format(path),
            )
            expected_y, expected_status = labels._combined_label(
                action.get("R"),
                action.get("T"),
                action.get("R_status"),
                action.get("T_status"),
            )
            _require(
                action.get("Y") is expected_y and action.get("Y_status") == expected_status,
                "R/T/Y algebra is invalid: {}".format(path),
            )
            baseline = baseline_actions.get(
                (action["pose_id"], action["part_move"], action["direction_id"])
            )
            _require(baseline is not None, "baseline action is missing")
            if action.get("R") != baseline.get("R"):
                mismatches.append(
                    "{} {}: R differs from the baseline root labels".format(
                        location, action["direction_id"]
                    )
                )
            if action.get("Y") != baseline.get("Y"):
                mismatches.append(
                    "{} {}: Y differs from the baseline root labels".format(
                        location, action["direction_id"]
                    )
                )
            _count(label_counts["R"], action.get("R"))
            _count(label_counts["Y"], action.get("Y"))

        for field in accounting:
            accounting[field] += summary.get("accounting", {}).get(field, 0)
        timings = summary.get("timings_s")
        _require(
            isinstance(timings, dict) and set(timings) == set(_TIMING_FIELDS),
            "candidate timing fields are invalid: {}".format(path),
        )
        for field in _TIMING_FIELDS:
            value = timings[field]
            _require(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value >= 0,
                "candidate {} is invalid: {}".format(field, path),
            )
            timing_values[field].append(float(value))
        if prepared["pose_cache_dir"] is not None:
            _require(
                timings["stable_pose_search_wall_s"] == 0.0,
                "candidate recomputed stable poses: {}".format(path),
            )
            cached_pose = pose_cache_manifest["poses"][candidate["pose_index"]]
            _require(
                header.get("pose") == cached_pose.get("pose")
                and _same_probability(
                    header.get("pose_probability"), cached_pose.get("probability")
                ),
                "candidate pose differs from stable-pose cache: {}".format(path),
            )
            reference_path = (
                prepared["reference_labels_root"]
                / "pose-{:03d}".format(candidate["pose_index"])
                / "part-{}.jsonl".format(candidate["part_move"])
            )
            if not reference_path.is_file():
                raise FileNotFoundError(
                    "conditional root reference shard not found: {}".format(reference_path)
                )
            reference_records = _read_jsonl(reference_path)
            _require(
                len(reference_records) == 9,
                "conditional root reference shard must contain 9 records: {}".format(
                    reference_path
                ),
            )
            _require(
                [record.get("record_type") for record in reference_records]
                == ["header", "stability"] + ["action"] * 6 + ["summary"],
                "conditional root reference record order is invalid: {}".format(
                    reference_path
                ),
            )
            _require(
                all(
                    record.get("schema_version") == conditional.SCHEMA_VERSION
                    for record in reference_records
                ),
                "conditional root reference label schema is invalid: {}".format(
                    reference_path
                ),
            )
            mismatches.extend(
                _semantic_mismatches(records, reference_records, location)
            )
            reference_files.append(
                {
                    "path": reference_path.relative_to(
                        Path(output_root).resolve()
                    ).as_posix(),
                    "sha256": labels._sha256_file(reference_path),
                }
            )
        files.append(
            {
                "candidate_index": candidate["index"],
                "path": path.relative_to(Path(output_root).resolve()).as_posix(),
                "sha256": labels._sha256_file(path),
                "size": path.stat().st_size,
            }
        )

    expected = prepared["manifest"]["expected"]
    actual = {
        "candidate_count": len(files),
        "action_count": accounting["candidate_actions"],
        "candidates_with_removable_direction": candidates_with_removal,
        "stability_queries": accounting["stability_queries"],
        "stability_queries_avoided": len(files) - accounting["stability_queries"],
        "stability_rollouts": accounting["stability_rollouts"],
        "oracle_rollouts_total": accounting["oracle_rollouts_total"],
        "label_counts": label_counts,
    }
    baseline_accounting = {
        "candidate_actions": len(baseline_action_records),
        "removability_rollouts": sum(
            record.get("oracle_rollout_count", 0) for record in baseline_action_records
        ),
        "stability_queries": len(baseline_stability_records),
        "stability_rollouts": sum(
            record.get("oracle_rollout_count", 0)
            for record in baseline_stability_records
        ),
    }
    baseline_accounting["oracle_rollouts_total"] = (
        baseline_accounting["removability_rollouts"]
        + baseline_accounting["stability_rollouts"]
    )
    savings = {
        "stability_queries": baseline_accounting["stability_queries"]
        - accounting["stability_queries"],
        "stability_rollouts": baseline_accounting["stability_rollouts"]
        - accounting["stability_rollouts"],
        "oracle_rollouts_total": baseline_accounting["oracle_rollouts_total"]
        - accounting["oracle_rollouts_total"],
    }
    for field, expected_value in expected.items():
        if actual.get(field) != expected_value:
            mismatches.append(
                "aggregate {} differs: expected {!r}, observed {!r}".format(
                    field, expected_value, actual.get(field)
                )
            )

    baseline_header = baseline_records[0]
    baseline_summary = baseline_records[-1]
    if reference_source.get("sha256") != baseline_header.get("provenance", {}).get(
        "source", {}
    ).get("sha256"):
        mismatches.append("core source SHA-256 differs from the baseline root labels")
    baseline_runtime = baseline_header.get("provenance", {}).get("runtime")
    if reference_runtime_contract != _runtime_contract(baseline_runtime):
        mismatches.append("runtime contract differs from the baseline root labels")

    spec = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": prepared["manifest_sha256"],
        "baseline_sha256": labels._sha256_file(prepared["baseline_path"]),
        "auditor_sha256": labels._sha256_file(Path(__file__)),
        "candidate_sha256": [entry["sha256"] for entry in files],
        "reference_candidate_sha256": [
            entry["sha256"] for entry in reference_files
        ],
        "pose_comparison": {
            "matrix": "exact",
            "probability_abs_tolerance": POSE_PROBABILITY_ABS_TOLERANCE,
            "probability_rel_tolerance": POSE_PROBABILITY_REL_TOLERANCE,
        },
    }
    spec_sha256 = labels._sha256_bytes(labels._canonical_json(spec).encode("utf-8"))
    existing = _existing_report(audit_path, spec_sha256)
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
        "baseline": {
            "path": prepared["manifest"]["baseline"],
            "sha256": spec["baseline_sha256"],
            "runtime_s": baseline_summary.get("runtime_s"),
        },
        "cache": reference_cache,
        "source": reference_source,
        "runtime": reference_runtime,
        "stable_pose_cache": reference_pose_cache,
        "pose_comparison": spec["pose_comparison"],
        "expected": expected,
        "observed": actual,
        "accounting": accounting,
        "baseline_full_coverage_accounting": baseline_accounting,
        "savings_vs_baseline": savings,
        "timings_s": {
            field: _timing_summary(timing_values[field]) for field in _TIMING_FIELDS
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "candidate_files": files,
        "reference_candidate_files": reference_files,
    }
    labels._write_atomic(audit_path, [report])
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--audit-output",
        help="optional output-root-relative .json path; never overwrites the manifest path",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report = audit(
        args.manifest,
        args.source_root,
        args.output_root,
        audit_output=args.audit_output,
    )
    print(labels._canonical_json(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
