"""Freeze audited label-production labels into candidate/action indexes and a profile."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from plan_sequence import production_sim_labels as production


CONFIG_SCHEMA_VERSION = "fast.dataset-index-manifest.v1"
BUILD_SCHEMA_VERSION = "fast.dataset-index-build.v1"
PROFILE_SCHEMA_VERSION = "fast.dataset-profile.v1"
ASSEMBLY_SCHEMA_VERSION = "fast.dataset-assembly-index.v1"
CANDIDATE_SCHEMA_VERSION = "fast.dataset-candidate-index.v1"
ACTION_SCHEMA_VERSION = "fast.dataset-action-index.v1"
LABEL_SCHEMA_VERSION = "fast.reachable-sim-labels.v1"
AUDIT_SCHEMAS = {
    "small": "fast.reachable-label-audit.v1",
    "medium": "fast.adaptive-label-audit.v1",
    "large": "fast.greedy-label-audit.v1",
}
DIRECTIONS = (
    ("+Z", [0, 0, 1]),
    ("-Z", [0, 0, -1]),
    ("+X", [1, 0, 0]),
    ("-X", [-1, 0, 0]),
    ("+Y", [0, 1, 0]),
    ("-Y", [0, -1, 0]),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _resolve_under(root, value, name):
    relative = production._relative_path(value, name)
    path = (Path(root).resolve() / relative).resolve()
    if not production._is_within(path, root):
        raise ValueError("{} escapes its root".format(name))
    return path


def _read_config(path, data_root):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("dataset-index manifest not found: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    production._require_fields(
        config,
        {
            "schema_version",
            "phase",
            "production_manifest",
            "combined_audit",
            "output_root",
            "validation",
            "primary_test_exclusions",
            "expected_assemblies",
        },
        "dataset-index manifest",
    )
    _require(
        config["schema_version"] == CONFIG_SCHEMA_VERSION,
        "unsupported dataset-index manifest schema",
    )
    validation = config["validation"]
    production._require_fields(
        validation,
        {"fraction", "seed", "group_by", "part_count_bins"},
        "validation",
    )
    _require(
        isinstance(validation["fraction"], (int, float))
        and not isinstance(validation["fraction"], bool)
        and 0 < validation["fraction"] < 0.5,
        "validation.fraction must be between zero and one half",
    )
    _require(type(validation["seed"]) is int and validation["seed"] >= 0, "invalid seed")
    _require(validation["group_by"] == "input_sha256", "validation must group by input hash")
    bins = validation["part_count_bins"]
    _require(
        isinstance(bins, list)
        and all(
            isinstance(item, list)
            and len(item) == 2
            and all(type(value) is int for value in item)
            and item[0] <= item[1]
            for item in bins
        ),
        "validation.part_count_bins is invalid",
    )
    covered = [count for lower, upper in bins for count in range(lower, upper + 1)]
    _require(covered == list(range(3, 71)), "part-count bins must cover 3 through 70 once")

    repository_root = path.parent.parent
    production_manifest = _resolve_under(
        repository_root, config["production_manifest"], "production_manifest"
    )
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError("data root not found: {}".format(data_root))
    combined_audit = _resolve_under(data_root, config["combined_audit"], "combined_audit")
    output_root = _resolve_under(data_root, config["output_root"], "output_root")
    _require(combined_audit.is_file(), "combined production audit is missing")
    _require(not production._is_within(combined_audit, output_root), "output overlaps audit")
    return {
        "config": config,
        "config_path": path,
        "config_sha256": production._sha256_file(path),
        "repository_root": repository_root,
        "data_root": data_root,
        "production_manifest": production_manifest,
        "combined_audit": combined_audit,
        "output_root": output_root,
    }


def _part_count_bin(part_count, bins):
    matches = [index for index, bounds in enumerate(bins) if bounds[0] <= part_count <= bounds[1]]
    _require(len(matches) == 1, "part count does not belong to one validation stratum")
    return matches[0]


def _split_jobs(jobs, config):
    exclusions = {
        (item["split"], item["assembly_id"]): item["reason"]
        for item in config["primary_test_exclusions"]
    }
    _require(
        len(exclusions) == len(config["primary_test_exclusions"]),
        "primary test exclusions contain duplicates",
    )
    by_identity = {(row["split"], row["assembly_id"]): row for row in jobs}
    training_hashes = {
        row["input_sha256"] for row in jobs if row["split"] == "training_assembly"
    }
    for identity in exclusions:
        _require(identity in by_identity, "primary test exclusion is absent from production")
        row = by_identity[identity]
        _require(
            row["split"] == "test_assembly" and row["input_sha256"] in training_hashes,
            "primary test exclusion must be an exact training duplicate",
        )

    groups = {}
    for row in jobs:
        if row["split"] == "training_assembly":
            groups.setdefault(row["input_sha256"], []).append(row)
    strata = {}
    bins = config["validation"]["part_count_bins"]
    for input_sha256, rows in groups.items():
        keys = {(row["tier"], _part_count_bin(row["part_count"], bins)) for row in rows}
        _require(len(keys) == 1, "duplicate input hash spans validation strata")
        strata.setdefault(keys.pop(), []).append((input_sha256, rows))

    validation_hashes = set()
    fraction = config["validation"]["fraction"]
    seed = config["validation"]["seed"]
    for groups_in_stratum in strata.values():
        total = sum(len(rows) for _, rows in groups_in_stratum)
        target = max(1, int(total * fraction + 0.5)) if len(groups_in_stratum) > 1 else 0
        selected = 0
        ranked = sorted(
            groups_in_stratum,
            key=lambda item: hashlib.sha256(
                "{}:{}".format(seed, item[0]).encode("ascii")
            ).hexdigest(),
        )
        for input_sha256, rows in ranked:
            if selected >= target:
                break
            if total - selected - len(rows) < 1:
                continue
            validation_hashes.add(input_sha256)
            selected += len(rows)

    assignments = {}
    for row in jobs:
        identity = (row["split"], row["assembly_id"])
        if identity in exclusions:
            assignments[identity] = "excluded"
        elif row["split"] == "test_assembly":
            assignments[identity] = "test"
        elif row["input_sha256"] in validation_hashes:
            assignments[identity] = "validation"
        else:
            assignments[identity] = "train"
    return assignments


def _assembly_counts(jobs, assignments):
    result = {}
    for dataset_split in ("train", "validation", "test", "excluded"):
        rows = [
            row
            for row in jobs
            if assignments[(row["split"], row["assembly_id"])] == dataset_split
        ]
        result[dataset_split] = {
            "total": len(rows),
            **{tier: sum(row["tier"] == tier for row in rows) for tier in production.TIER_SPECS},
        }
    return result


def _empty_stats():
    return {
        "assembly_count": 0,
        "complete_sequence_count": 0,
        "audit_status": Counter(),
        "orphan_state_checkpoint_count": 0,
        "candidate_count": 0,
        "action_count": 0,
        "stability_queried": Counter(),
        "labels": {name: Counter() for name in ("R", "T", "Y")},
        "statuses": {name: Counter() for name in ("R", "T", "Y")},
        "reasons": {name: Counter() for name in ("R", "T")},
    }


def _serialise_stats(stats):
    labels = {
        name: {key: stats["labels"][name][key] for key in ("true", "false", "null")}
        for name in ("R", "T", "Y")
    }
    label_rates = {}
    for name, counts in labels.items():
        valid_count = counts["true"] + counts["false"]
        total_count = valid_count + counts["null"]
        label_rates[name] = {
            "valid_count": valid_count,
            "positive_fraction_among_valid": (
                counts["true"] / valid_count if valid_count else None
            ),
            "null_fraction": counts["null"] / total_count if total_count else None,
        }
    return {
        "assembly_count": stats["assembly_count"],
        "complete_sequence_count": stats["complete_sequence_count"],
        "complete_sequence_fraction": (
            stats["complete_sequence_count"] / stats["assembly_count"]
            if stats["assembly_count"]
            else None
        ),
        "audit_status": dict(sorted(stats["audit_status"].items())),
        "orphan_state_checkpoint_count": stats["orphan_state_checkpoint_count"],
        "candidate_count": stats["candidate_count"],
        "action_count": stats["action_count"],
        "stability_queried": dict(sorted(stats["stability_queried"].items())),
        "stability_queried_fraction": (
            stats["stability_queried"]["true"] / stats["candidate_count"]
            if stats["candidate_count"]
            else None
        ),
        "labels": labels,
        "label_rates": label_rates,
        "statuses": {
            name: dict(sorted(stats["statuses"][name].items())) for name in ("R", "T", "Y")
        },
        "reasons": {
            name: dict(sorted(stats["reasons"][name].items())) for name in ("R", "T")
        },
    }


def _label_key(value):
    return "true" if value is True else "false" if value is False else "null"


def _text_key(value):
    return value if isinstance(value, str) and value else "null"


def _validate_candidate(path, expected_sha256, row, entry):
    _require(production._sha256_file(path) == expected_sha256, "candidate SHA-256 differs")
    records = _read_jsonl(path)
    _require(len(records) == 9, "candidate must contain nine records")
    _require(
        [record.get("record_type") for record in records]
        == ["header", "stability"] + ["action"] * 6 + ["summary"],
        "candidate record order is invalid",
    )
    _require(
        all(record.get("schema_version") == LABEL_SCHEMA_VERSION for record in records),
        "candidate label schema is invalid",
    )
    header, stability, summary = records[0], records[1], records[-1]
    _require(
        header.get("assembly_id") == row["assembly_id"]
        and header.get("state_id") == entry["state_id"]
        and header.get("pose_id") == entry["pose_id"]
        and header.get("part_move") == entry["part_move"],
        "candidate identity differs from its summary entry",
    )
    _require(
        summary.get("complete") is True and summary.get("record_count") == 9,
        "candidate summary is incomplete",
    )
    _require(
        [(action.get("direction_id"), action.get("direction")) for action in records[2:8]]
        == list(DIRECTIONS),
        "candidate action directions are invalid",
    )
    _require(
        all(type(action.get("R")) in (bool, type(None)) for action in records[2:8])
        and all(type(action.get("Y")) in (bool, type(None)) for action in records[2:8])
        and type(stability.get("T")) in (bool, type(None)),
        "candidate labels must be bool or null",
    )
    return header, stability, records[2:8]


def _matching_lines(path, prefix):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if prefix in line:
                yield json.loads(line)


def index_candidates(dataset_root, row):
    """Candidate headers, stability labels, and actions of one assembly, read from the dataset index.

    Returns (header, stability, actions) tuples in index order, with the fields that the feature and
    graph builders use, as `_validate_candidate` returns them from the label files.
    """
    prefix = '"candidate_id":"{}:{}:'.format(row["source_split"], row["assembly_id"])
    actions = {}
    for action in _matching_lines(Path(dataset_root) / "actions.jsonl", prefix):
        actions.setdefault(action["candidate_id"], []).append(action)
    result = []
    for candidate in _matching_lines(Path(dataset_root) / "candidates.jsonl", prefix):
        _require(candidate["assembly_id"] == row["assembly_id"] and candidate["source_split"] == row["source_split"],
                 "index candidate belongs to another assembly")
        header = {name: candidate[name] for name in
                  ("state_id", "pose_id", "pose_index", "pose_probability", "pose", "parts", "parts_after", "part_move")}
        header["assembly_id"] = candidate["assembly_id"]
        stability = {"T": candidate["T"], "T_status": candidate["T_status"], "reason": candidate["T_reason"],
                     "queried": candidate["T_queried"]}
        rows = actions.pop(candidate["candidate_id"], [])
        _require([(action["direction_id"], action["direction"]) for action in rows] == list(DIRECTIONS),
                 "index actions are incomplete or out of order")
        result.append((header, stability, rows))
    _require(not actions and len(result) == row["candidate_count"], "index candidates differ from the assembly row")
    return result


def _scope_stats(stats, dataset_split, tier):
    return (
        stats["overall"],
        stats["per_split"].setdefault(dataset_split, _empty_stats()),
        stats["per_split_tier"].setdefault(
            "{}/{}".format(dataset_split, tier), _empty_stats()
        ),
    )


def _write_jsonl_record(stream, record):
    stream.write(production._canonical_json(record) + "\n")


def _existing_build(output_root, build_spec_sha256):
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        if output_root.exists():
            raise FileExistsError("dataset-index output exists but is incomplete")
        return None
    manifest = production._read_single(manifest_path)
    _require(manifest.get("schema_version") == BUILD_SCHEMA_VERSION, "invalid existing build")
    if manifest.get("build_spec_sha256") != build_spec_sha256:
        raise FileExistsError("dataset-index output belongs to another build specification")
    for artifact in manifest.get("artifacts", {}).values():
        path = _resolve_under(output_root, artifact["path"], "existing artifact path")
        _require(path.is_file(), "existing dataset-index artifact is missing")
        _require(production._sha256_file(path) == artifact["sha256"], "artifact hash differs")
    result = dict(manifest)
    result["idempotent"] = True
    return result


def build(manifest_path, data_root):
    prepared = _read_config(manifest_path, data_root)
    production_prepared = production._read_spec(prepared["production_manifest"])
    jobs = production._read_jobs(production_prepared)
    combined = production._read_single(prepared["combined_audit"])
    _require(
        combined.get("schema_version") == production.AUDIT_SCHEMA_VERSION
        and combined.get("status") == "passed"
        and combined.get("integrity_status") == "passed",
        "combined production audit has not passed",
    )
    _require(
        combined.get("manifest_sha256") == production_prepared["manifest_sha256"]
        and combined.get("inventory_sha256") == production_prepared["inventory_sha256"]
        and combined.get("audit_count") == len(jobs),
        "combined production audit provenance differs",
    )
    _require(
        combined.get("excluded_assemblies")
        == production_prepared["manifest"]["excluded_assemblies"],
        "production exclusions differ from the combined audit",
    )
    _require(
        combined.get("primary_test_exclusions", combined.get("unseen_test_exclusions"))
        == prepared["config"]["primary_test_exclusions"],
        "primary test exclusions differ from the combined audit",
    )

    assignments = _split_jobs(jobs, prepared["config"])
    assembly_counts = _assembly_counts(jobs, assignments)
    _require(
        assembly_counts == prepared["config"]["expected_assemblies"],
        "deterministic assembly split differs from the frozen expectation",
    )
    build_spec = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "config_sha256": prepared["config_sha256"],
        "production_manifest_sha256": production_prepared["manifest_sha256"],
        "production_inventory_sha256": production_prepared["inventory_sha256"],
        "combined_audit_sha256": production._sha256_file(prepared["combined_audit"]),
        "builder_sha256": production._sha256_file(Path(__file__)),
    }
    build_spec_sha256 = production._sha256_bytes(
        production._canonical_json(build_spec).encode("utf-8")
    )
    existing = _existing_build(prepared["output_root"], build_spec_sha256)
    if existing is not None:
        return existing

    labels_root = prepared["data_root"] / production_prepared["manifest"]["paths"]["labels_root"]
    audit_root = prepared["data_root"] / production_prepared["manifest"]["paths"]["audit_root"]
    stats = {"overall": _empty_stats(), "per_split": {}, "per_split_tier": {}}
    candidate_ids = set()
    row_counts = Counter()
    prepared["output_root"].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".dataset-index-", dir=str(prepared["output_root"].parent)
    ) as temporary:
        staging = Path(temporary)
        with (staging / "assemblies.jsonl").open("w", encoding="utf-8") as assemblies_stream, (
            staging / "candidates.jsonl"
        ).open("w", encoding="utf-8") as candidates_stream, (
            staging / "actions.jsonl"
        ).open("w", encoding="utf-8") as actions_stream:
            for row in jobs:
                identity = (row["split"], row["assembly_id"])
                dataset_split = assignments[identity]
                assembly_root = labels_root / row["split"] / row["assembly_id"]
                summary_path = assembly_root / "summary.json"
                audit_path = audit_root / row["split"] / (row["assembly_id"] + ".json")
                summary = production._read_single(summary_path)
                audit = production._read_single(audit_path)
                summary_sha256 = production._sha256_file(summary_path)
                _require(
                    summary.get("schema_version") == LABEL_SCHEMA_VERSION
                    and summary.get("complete") is True
                    and summary.get("input", {}).get("sha256") == row["input_sha256"],
                    "worker summary is invalid: {}".format(summary_path),
                )
                _require(
                    audit.get("schema_version") == AUDIT_SCHEMAS[row["tier"]]
                    and audit.get("integrity_status") == "passed"
                    and audit.get("completion_sha256") == summary_sha256,
                    "assembly audit is invalid: {}".format(audit_path),
                )
                complete_sequence = audit.get("coverage", {}).get("complete_sequence_found")
                if complete_sequence is None:
                    complete_sequence = bool(audit.get("terminal_states"))
                exclusion_reason = next(
                    (
                        item["reason"]
                        for item in prepared["config"]["primary_test_exclusions"]
                        if (item["split"], item["assembly_id"]) == identity
                    ),
                    None,
                )
                assembly_record = {
                    "schema_version": ASSEMBLY_SCHEMA_VERSION,
                    "dataset_split": dataset_split,
                    "included": dataset_split != "excluded",
                    "exclusion_reason": exclusion_reason,
                    "source_split": row["split"],
                    "assembly_id": row["assembly_id"],
                    "input_sha256": row["input_sha256"],
                    "tier": row["tier"],
                    "part_count": row["part_count"],
                    "candidate_count": summary["accounting"]["candidate_count"],
                    "complete_sequence_found": complete_sequence,
                    "audit_status": audit.get("status"),
                    "orphan_state_checkpoints": audit.get("orphan_state_checkpoints", []),
                }
                _write_jsonl_record(assemblies_stream, assembly_record)
                row_counts["assemblies"] += 1
                if dataset_split == "excluded":
                    continue

                scopes = _scope_stats(stats, dataset_split, row["tier"])
                for scope in scopes:
                    scope["assembly_count"] += 1
                    scope["complete_sequence_count"] += complete_sequence is True
                    scope["audit_status"][_text_key(audit.get("status"))] += 1
                    scope["orphan_state_checkpoint_count"] += len(
                        audit.get("orphan_state_checkpoints", [])
                    )

                entries = summary.get("candidate_files")
                _require(
                    isinstance(entries, list)
                    and len(entries) == summary["accounting"]["candidate_count"],
                    "worker candidate list is invalid",
                )
                relative_paths = [entry.get("path") for entry in entries]
                _require(
                    len(relative_paths) == len(set(relative_paths)),
                    "duplicate candidate path",
                )
                for entry in entries:
                    candidate_path = _resolve_under(assembly_root, entry["path"], "candidate path")
                    if not candidate_path.is_file():
                        raise FileNotFoundError("candidate is missing: {}".format(candidate_path))
                    header, stability, actions = _validate_candidate(
                        candidate_path, entry["sha256"], row, entry
                    )
                    candidate_id = "{}:{}:{}:{}:{}".format(
                        row["split"],
                        row["assembly_id"],
                        header["state_id"],
                        header["pose_id"],
                        header["part_move"],
                    )
                    _require(candidate_id not in candidate_ids, "duplicate candidate identity")
                    candidate_ids.add(candidate_id)
                    candidate_record = {
                        "schema_version": CANDIDATE_SCHEMA_VERSION,
                        "candidate_id": candidate_id,
                        "dataset_split": dataset_split,
                        "source_split": row["split"],
                        "assembly_id": row["assembly_id"],
                        "input_sha256": row["input_sha256"],
                        "tier": row["tier"],
                        "assembly_part_count": row["part_count"],
                        "state_id": header["state_id"],
                        "state_part_count": len(header["parts"]),
                        "depth": row["part_count"] - len(header["parts"]),
                        "parts": header["parts"],
                        "parts_after": header["parts_after"],
                        "part_move": header["part_move"],
                        "pose_id": header["pose_id"],
                        "pose_index": header["pose_index"],
                        "pose_probability": header["pose_probability"],
                        "pose": header["pose"],
                        "T": stability.get("T"),
                        "T_status": stability.get("T_status"),
                        "T_reason": stability.get("reason"),
                        "T_queried": stability.get("queried"),
                        "T_valid": stability.get("queried") is True
                        and type(stability.get("T")) is bool,
                    }
                    _write_jsonl_record(candidates_stream, candidate_record)
                    row_counts["candidates"] += 1
                    for scope in scopes:
                        scope["candidate_count"] += 1
                        scope["stability_queried"][
                            "true" if stability.get("queried") is True else "false"
                        ] += 1
                        scope["labels"]["T"][_label_key(stability.get("T"))] += 1
                        scope["statuses"]["T"][_text_key(stability.get("T_status"))] += 1
                        scope["reasons"]["T"][_text_key(stability.get("reason"))] += 1

                    for action in actions:
                        action_record = {
                            "schema_version": ACTION_SCHEMA_VERSION,
                            "action_id": "{}:{}".format(candidate_id, action["direction_id"]),
                            "candidate_id": candidate_id,
                            "dataset_split": dataset_split,
                            "direction_id": action["direction_id"],
                            "direction": action["direction"],
                            "R": action.get("R"),
                            "R_status": action.get("R_status"),
                            "R_reason": action.get("R_reason"),
                            "R_valid": type(action.get("R")) is bool,
                            "Y": action.get("Y"),
                            "Y_status": action.get("Y_status"),
                            "Y_valid": type(action.get("Y")) is bool,
                        }
                        _write_jsonl_record(actions_stream, action_record)
                        row_counts["actions"] += 1
                        for scope in scopes:
                            scope["action_count"] += 1
                            for name in ("R", "Y"):
                                scope["labels"][name][_label_key(action.get(name))] += 1
                                scope["statuses"][name][
                                    _text_key(action.get(name + "_status"))
                                ] += 1
                            scope["reasons"]["R"][
                                _text_key(action.get("R_reason"))
                            ] += 1

        profile = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "phase": prepared["config"]["phase"],
            "build_spec_sha256": build_spec_sha256,
            "source": {
                "production_manifest_sha256": production_prepared["manifest_sha256"],
                "production_inventory_sha256": production_prepared["inventory_sha256"],
                "combined_audit_sha256": build_spec["combined_audit_sha256"],
            },
            "assembly_splits": assembly_counts,
            "overall": _serialise_stats(stats["overall"]),
            "per_split": {
                key: _serialise_stats(value) for key, value in sorted(stats["per_split"].items())
            },
            "per_split_tier": {
                key: _serialise_stats(value)
                for key, value in sorted(stats["per_split_tier"].items())
            },
            "production_exclusions": production_prepared["manifest"]["excluded_assemblies"],
            "primary_test_exclusions": prepared["config"]["primary_test_exclusions"],
            "index_policy": {
                "candidate_source": "worker summary candidate_files only",
                "null_targets": "censored; validity masks are false",
                "stability_unit": "one candidate-level target; not repeated per direction",
            },
        }
        (staging / "profile.json").write_text(
            production._canonical_json(profile) + "\n", encoding="utf-8"
        )
        artifacts = {}
        for name, filename in (
            ("assemblies", "assemblies.jsonl"),
            ("candidates", "candidates.jsonl"),
            ("actions", "actions.jsonl"),
            ("profile", "profile.json"),
        ):
            path = staging / filename
            artifacts[name] = {
                "path": filename,
                "sha256": production._sha256_file(path),
                "row_count": 1 if name == "profile" else row_counts[name],
            }
        result = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "phase": prepared["config"]["phase"],
            "build_spec_sha256": build_spec_sha256,
            "build_spec": build_spec,
            "assembly_splits": assembly_counts,
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            production._canonical_json(result) + "\n", encoding="utf-8"
        )
        if prepared["output_root"].exists():
            raise FileExistsError("dataset-index output appeared while building")
        os.replace(str(staging), str(prepared["output_root"]))
    return result


def _parser():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(root / "configs" / "dataset_index.json")
    )
    parser.add_argument("--data-root", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    print(production._canonical_json(build(args.manifest, args.data_root)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
