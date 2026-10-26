"""Aggregate-only Experiment E (test prediction quality) on the frozen 237 test assemblies.

The development loader and its zero-test-row guard are untouched. This separate evaluator
accepts only test rows bound by an immutable specification and sealed registry, rebuilds test
graph records with the unchanged ranker transformation, scores them with the three frozen
FAST checkpoints, and publishes aggregate-only prediction quality. No identifier, probability,
feature vector, or individual target leaves the sealed run root.
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from fast import cad_features, graph_contract, graph_materialize, ranker_models, ranker_train
from fast import ablation_closure as closure
from fast import ablation_common as common
from fast import ablation_offline as offline
from fast import ablation_runner as runner
from plan_sequence import production_sim_labels as production


RUN_ID = "test-prediction-quality-v1"
SCHEMA_VERSION = "fast.test-prediction-quality.v1"
SPEC_PATH = common.REPOSITORY_ROOT / "configs" / "test_prediction_quality.json"
TEST_SPLIT, TEST_SOURCE = "test", "test_assembly"
PLACEHOLDER = {"dataset_split": "validation", "source_split": "training_assembly"}
DATASET_INDEX = "datasets/fast-labels-v1"
FEATURE_ROOT = "features/scalar-features-v3"
LABEL_ROOT = "labels/fast-labels-v1"
FIXTURES_PER_TIER = 2
IMPLEMENTATION_FILES = (
    "fast/prediction_quality.py", "fast/ablation_offline.py", "fast/ablation_common.py",
    "fast/ablation_closure.py", "fast/graph_contract.py", "fast/graph_materialize.py",
    "fast/ranker_models.py", "fast/ranker_train.py", "fast/ranker_validate.py",
    "fast/cad_features.py", "plan_sequence/build_dataset_index.py", "plan_sequence/production_sim_labels.py",
    "assets/load.py", "assets/transform.py", "assets/color.py",
    "configs/test_prediction_quality.json", "configs/features.json", "configs/ranker_models.json",
    "environment/cu121/uv.lock",
)
EXPORTED_OUTPUTS = ("specification.json", "input-audit.json", "feature-materialization-audit.json",
                    "validation-reference-reproduction.json", "test-prediction-quality.json", "final-audit.json")
SEALED_ENTRIES = ("registry.json", "test-transfer-manifest.json", "fixture-transfer-manifest.json", "d-decision.json",
                  "closure", "fixtures", "test-graphs", "logs", "preflight-attempts")
FORBIDDEN_KEYS = {"assembly_id", "assembly_ids", "state_id", "pose_id", "part_id", "part_ids", "query_id", "node_x",
                  "action_x", "edge_index", "targets", "R_score", "T_score", "Y_score", "R_logit", "T_logit", "Y_logit",
                  "probability", "probabilities", "score", "scores", "feature", "features", "prediction", "predictions"}
FORBIDDEN_TEXT = re.compile(r"(^|[^0-9a-f])[0-9]{5}($|[^0-9a-f])|state-|pose-|test_assembly[/:]")
MAX_NUMERIC_LIST = 16


def run_root(data_root):
    return Path(data_root).resolve() / "runs" / RUN_ID


def require_fixed_hash_seed():
    """Frozen scalar-feature geometry iterates string sets; a fixed hash seed makes float summation order reproducible."""
    common.require(os.environ.get("PYTHONHASHSEED") == "0" and sys.flags.hash_randomization == 0,
                   "materialization requires PYTHONHASHSEED=0 for reproducible frozen-feature recomputation")


def load_spec():
    spec = common.read_json(SPEC_PATH)
    common.require(spec.get("status") == "frozen" and spec.get("run_id") == RUN_ID, "test-quality specification is not frozen")
    return spec, common.sha256_file(SPEC_PATH)


def implementation_hashes():
    return OrderedDict((path, common.sha256_file(common.REPOSITORY_ROOT / path)) for path in IMPLEMENTATION_FILES)


# ---------------------------------------------------------------- aggregate-only and path guards


def aggregate_only(value, path="$"):
    """Fail if an exported structure could carry an identifier, probability, feature vector, or target."""
    if isinstance(value, dict):
        for key, item in value.items():
            common.require(key not in FORBIDDEN_KEYS, "aggregate-only output rejects key {} at {}".format(key, path))
            common.require(not FORBIDDEN_TEXT.search(str(key)), "aggregate-only output rejects identifier-like key at " + path)
            aggregate_only(item, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        numeric = sum(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        common.require(numeric <= MAX_NUMERIC_LIST, "aggregate-only output rejects a numeric vector at " + path)
        for index, item in enumerate(value):
            aggregate_only(item, "{}[{}]".format(path, index))
    elif isinstance(value, str):
        common.require(not FORBIDDEN_TEXT.search(value), "aggregate-only output rejects identifier-like text at " + path)
    return True


def publish(data_root, name, value):
    """Atomic, write-once, aggregate-only publication inside the run root."""
    common.require(name in EXPORTED_OUTPUTS, "unregistered output: " + name)
    aggregate_only(value)
    return common.write_json_once(safe_output(data_root, name), value)


def safe_output(data_root, relative):
    root = run_root(data_root)
    path = (root / relative).resolve()
    common.require(production._is_within(path, root), "output escapes the run root: " + str(relative))
    for protected in (root / "closure", root / "fixtures" / "closure", common.closure_root(data_root)):
        common.require(not production._is_within(path, protected), "output would write into a source closure")
    return path


def make_read_only(root):
    for path in sorted(Path(root).rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    Path(root).chmod(Path(root).stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def write_refused(path):
    try:
        with Path(path).open("a"):
            return False
    except (PermissionError, IsADirectoryError):
        return True


# ---------------------------------------------------------------- test records


def validate_test_record(record):
    """Separate test validator; the frozen development validator is never relaxed."""
    common.require(isinstance(record, dict) and record.get("dataset_split") == TEST_SPLIT and record.get("source_split") == TEST_SOURCE,
                   "test evaluator accepts only test rows")
    validated = graph_contract.validate_development_record(dict(record, **PLACEHOLDER))
    validated.update(dataset_split=TEST_SPLIT, source_split=TEST_SOURCE)
    return validated


def development_shape(record):
    """Present a validated test record to the frozen batching code, which keys only on split labels."""
    return dict(record, **PLACEHOLDER)


# ---------------------------------------------------------------- registry


def feature_build(data_root):
    path = common.closure_root(data_root) / "fast-data" / FEATURE_ROOT / "manifest.json"
    common.require(common.sha256_file(path) == closure.FROZEN_SHA256[FEATURE_ROOT + "/manifest.json"], "scalar-feature build manifest differs")
    records = common.read_jsonl(path)
    common.require(len(records) == 1 and records[0]["status"] == "passed", "scalar-feature build manifest is invalid")
    return records[0]


def dataset_rows(index_root):
    manifest_path = Path(index_root) / "manifest.json"
    common.require(common.sha256_file(manifest_path) == closure.FROZEN_SHA256[DATASET_INDEX + "/manifest.json"], "dataset manifest differs")
    manifest = common.read_json(manifest_path)
    artifact = manifest["artifacts"]["assemblies"]
    path = Path(index_root) / "assemblies.jsonl"
    common.require(common.sha256_file(path) == artifact["sha256"], "dataset assembly index differs from its manifest")
    rows = {}
    for row in common.read_jsonl(path):
        key = (row["source_split"], row["assembly_id"])
        common.require(key not in rows, "dataset assembly index has a duplicate")
        rows[key] = row
    return rows


def validate_registry(entries, spec, rows=None):
    population = spec["population"]
    identifiers = [entry["assembly_id"] for entry in entries]
    common.require(len(identifiers) == len(set(identifiers)), "registry has a duplicated assembly")
    common.require(all(entry["dataset_split"] == TEST_SPLIT and entry["source_split"] == TEST_SOURCE for entry in entries),
                   "registry has a foreign, training, or validation row")
    common.require(len(entries) == population["assembly_count"], "registry is missing assemblies")
    common.require(dict(Counter(entry["tier"] for entry in entries)) == population["tier_counts"], "registry tier counts differ")
    common.require(sum(entry["candidate_count"] for entry in entries) == population["feature_candidate_count"]
                   and sum(entry["action_count"] for entry in entries) == population["feature_action_count"], "registry scalar-feature totals differ")
    common.require([entry["index"] for entry in entries] == list(range(len(entries))), "registry order differs")
    if rows is not None:
        included = {key[1] for key, row in rows.items() if row["dataset_split"] == TEST_SPLIT and row["included"]}
        common.require(included == set(identifiers), "registry differs from the included test assemblies of the frozen index")
    return True


def label_summary_path(source_split, assembly_id):
    """The label store keeps one search summary per assembly; its candidate files carry their own hashes."""
    return "{}/{}/{}/summary.json".format(LABEL_ROOT, source_split, assembly_id)


def build_registry(build, rows, spec):
    entries = []
    shards = sorted((shard for shard in build["shards"] if shard["dataset_split"] == TEST_SPLIT), key=lambda shard: shard["path"])
    for index, shard in enumerate(shards):
        parts = Path(shard["path"]).parts
        common.require(len(parts) == 3 and parts[0] == "shards" and parts[1] == TEST_SOURCE, "invalid scalar-feature test shard path")
        assembly_id = Path(parts[2]).stem
        row = rows.get((TEST_SOURCE, assembly_id))
        common.require(row is not None and row["dataset_split"] == TEST_SPLIT and row["included"] and row["tier"] == shard["tier"],
                       "scalar-feature test shard has no matching included test index row")
        entries.append(OrderedDict([
            ("index", index), ("assembly_id", assembly_id), ("dataset_split", TEST_SPLIT), ("source_split", TEST_SOURCE),
            ("tier", shard["tier"]), ("feature_shard_path", shard["path"]), ("feature_shard_sha256", shard["sha256"]),
            ("candidate_count", shard["candidate_count"]), ("action_count", shard["action_count"]),
            ("input_sha256", row["input_sha256"]), ("label_summary_path", label_summary_path(TEST_SOURCE, assembly_id)),
        ]))
    validate_registry(entries, spec, rows)
    return entries


# ---------------------------------------------------------------- closures


def _select(requests, listing, allowed_prefixes):
    selected, excluded, seen = [], Counter(), set()
    by_namespace = {}
    for (namespace, path), size in listing.items():
        by_namespace.setdefault(namespace, {})[path] = size
    for namespace, item, kind, request in requests:
        available = by_namespace.get(namespace, {})
        matches = [request] if kind == "file" and request in available else sorted(
            path for path in available if kind != "file" and path.startswith(request.rstrip("/") + "/"))
        common.require(matches, "requested input is absent")
        for path in matches:
            common.require(any(path.startswith(prefix) for prefix in allowed_prefixes[namespace]), "input outside the registered prefixes")
            if kind == "geometry":
                name = path[len(request) + 1:]
                if "/" in name or not closure.GEOMETRY_FILE.match(name):
                    excluded[Path(name).suffix or name] += 1
                    continue
            if kind == "tree":
                common.require(path.endswith((".json", ".jsonl")), "unexpected label payload")
            common.require((namespace, path) not in seen, "input requested twice")
            seen.add((namespace, path))
            selected.append({"namespace": namespace, "item": item, "path": path, "bytes": int(available[path])})
    return sorted(selected, key=lambda row: (row["namespace"], row["path"])), dict(excluded)


def _plan(requests, allowed_prefixes):
    listing = closure._source_listing(requests)
    files, excluded = _select(requests, listing, allowed_prefixes)
    hashes = closure._source_hashes(files)
    for row in files:
        row["sha256"] = hashes[(row["namespace"], row["path"])]
    return files, excluded


def _transfer(files, destination):
    host, roots = closure.source()
    for namespace, source_root in roots.items():
        paths = [row["path"] for row in files if row["namespace"] == namespace]
        if not paths:
            continue
        (destination / namespace).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".files") as listing:
            listing.write("\n".join(paths) + "\n")
            listing.flush()
            result = subprocess.run(["rsync", "-a", "--partial", "--files-from=" + listing.name,
                                     closure.rsync_source(host, source_root), str(destination / namespace) + "/"],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        common.require(result.returncode == 0, "rsync failed for " + namespace)


def _verify_files(files, destination):
    expected = {(row["namespace"], row["path"]): row for row in files}
    found = closure._local_files(destination)
    common.require(set(found) == set(expected), "closure has unlisted or missing files")
    for key, row in expected.items():
        common.require(found[key].stat().st_size == row["bytes"] and common.sha256_file(found[key]) == row["sha256"], "closure hash differs")
    return len(expected), sum(row["bytes"] for row in expected.values())


def inventory_sha256(rows):
    return common.canonical_sha256(sorted([row["namespace"], row["path"], row["sha256"]] for row in rows))


def fixture_ids(data_root):
    audit = common.read_json(common.closure_root(data_root) / "fast-data" / closure.GRAPH_AUDIT)
    chosen = []
    for tier in common.TIERS:
        chosen += sorted(row["assembly_id"] for row in audit["inventory"] if row["dataset_split"] == "validation" and row["tier"] == tier)[:FIXTURES_PER_TIER]
    return chosen + [closure.REPLAY_FIXTURE]


def fixtures(data_root):
    """Pre-access fixture closure: training/validation inputs plus the dataset index only."""
    root = run_root(data_root) / "fixtures"
    manifest_path = run_root(data_root) / "fixture-transfer-manifest.json"
    common.require(not manifest_path.exists(), "fixture closure already exists")
    identifiers = fixture_ids(data_root)
    requests = [("fast-data", "dataset_index", "file", DATASET_INDEX + "/" + name) for name in ("manifest.json", "assemblies.jsonl")]
    for assembly_id in identifiers:
        requests += [("fast-data", "feature_shards", "file", FEATURE_ROOT + "/shards/training_assembly/{}.jsonl".format(assembly_id)),
                     ("fast-data", "labels", "tree", LABEL_ROOT + "/training_assembly/" + assembly_id),
                     ("asap-data", "geometry", "geometry", "training_assembly/" + assembly_id)]
    for _namespace, _item, _kind, path in requests:
        common.require_development_path(path)
    allowed = {"fast-data": (DATASET_INDEX + "/", FEATURE_ROOT + "/shards/training_assembly/", LABEL_ROOT + "/training_assembly/"),
               "asap-data": ("training_assembly/",)}
    files, excluded = _plan(requests, allowed)
    for row in files:
        common.require_development_path(row["path"])
    common.write_json_once(manifest_path, {"schema_version": SCHEMA_VERSION, "fixture_assembly_ids": identifiers,
                                           "splits": "validation fixtures per tier plus training replay fixture 00014",
                                           "files": files, "geometry_excluded_file_counts": excluded, "test_paths": 0})
    _transfer(files, root / "closure")
    count, size = _verify_files(files, root / "closure")
    make_read_only(root / "closure")
    return {"fixtures": len(identifiers), "files": count, "bytes": size}


# ---------------------------------------------------------------- materialization


def _prepared(data_root):
    return {
        "feature_config": common.read_json(common.REPOSITORY_ROOT / "configs" / "features.json"),
        "feature_build": feature_build(data_root),
    }


def materialize_assembly(prepared, entry, row, data_closure, geometry_closure, output_path=None):
    """Rebuild graph records with the unchanged frozen transformation; returns (records, counts)."""
    test = entry["dataset_split"] == TEST_SPLIT
    job = {"dataset_split": entry["dataset_split"], "source_split": entry["source_split"], "assembly_id": entry["assembly_id"],
           "tier": entry["tier"], "feature_shard_sha256": entry["feature_shard_sha256"]}
    assembly_dir = Path(geometry_closure) / entry["source_split"] / entry["assembly_id"]
    common.require(production._hash_assembly(assembly_dir) == row["input_sha256"] == entry["input_sha256"], "source assembly hash differs")
    feature_path = Path(data_closure) / FEATURE_ROOT / entry["feature_shard_path"]
    common.require(common.sha256_file(feature_path) == entry["feature_shard_sha256"], "scalar-feature shard hash differs")
    feature_records = common.read_jsonl(feature_path)
    common.require(feature_records[0]["record_type"] == "header" and feature_records[-1]["record_type"] == "summary", "scalar-feature shard envelope differs")
    feature_records = feature_records[1:-1]
    summary_path = Path(data_closure) / entry["label_summary_path"]
    label_summary = common.read_json(summary_path)
    common.require(len(label_summary["candidate_files"]) == row["candidate_count"], "label summary candidate count differs")
    raw_headers = {}
    for candidate in label_summary["candidate_files"]:
        header, _, _ = cad_features.dataset_index._validate_candidate(summary_path.parent / candidate["path"], candidate["sha256"], row, candidate)
        raw_headers["{}:{}:{}:{}:{}".format(entry["source_split"], entry["assembly_id"], header["state_id"], header["pose_id"], header["part_move"])] = header
    common.require(len(raw_headers) == len(feature_records), "raw and scalar-feature candidate counts differ")
    static_parts = cad_features._load_parts(assembly_dir)
    graph = cad_features._load_contact_graph(assembly_dir / "contact_graph.json", sorted(static_parts))
    full_volume = sum(part["volume"] for part in static_parts.values())
    contexts = OrderedDict()
    for record in feature_records:
        contexts.setdefault(record["query_id"], []).append(record)
    transform_job = dict(job, **PLACEHOLDER) if test else job
    records = []
    for grouped in contexts.values():
        record = graph_materialize._context_record(prepared, transform_job, row, grouped, raw_headers, static_parts, graph, full_volume)
        if test:
            record = dict(record, dataset_split=TEST_SPLIT, source_split=TEST_SOURCE)
            validate_test_record(record)
        records.append(record)
    counts = OrderedDict([
        ("context_count", len(records)), ("node_count", sum(len(record["node_x"]) for record in records)),
        ("edge_count", sum(len(record["edge_index"][0]) // 2 for record in records)),
        ("candidate_count", sum(len(record["candidate_index"]) for record in records)),
        ("action_count", 6 * sum(len(record["candidate_index"]) for record in records)),
    ])
    common.require(counts["candidate_count"] == entry["candidate_count"] and counts["action_count"] == entry["action_count"], "materialized scalar-feature totals differ")
    if output_path is not None:
        header = {"schema_version": graph_materialize.SHARD_SCHEMA_VERSION, "record_type": "header", "run_id": RUN_ID,
                  "registry_index": entry["index"], "dataset_split": entry["dataset_split"], "source_split": entry["source_split"],
                  "tier": entry["tier"], "feature_shard_sha256": entry["feature_shard_sha256"], "input_sha256": entry["input_sha256"]}
        summary = {"schema_version": graph_materialize.SHARD_SCHEMA_VERSION, "record_type": "summary", "complete": True, "counts": counts}
        common.require(not Path(output_path).exists(), "test graph shard cannot be overwritten")
        graph_materialize._write_jsonl(output_path, [header] + records + [summary])
    return records, counts


def _null_counts(records):
    counts = Counter()
    for record in records:
        targets = record["targets"]
        counts["R_labels"] += int(np.asarray(targets["R_valid"]).size)
        counts["R_unresolved"] += int((~np.asarray(targets["R_valid"], dtype=bool)).sum())
        counts["T_labels"] += int(np.asarray(targets["T_valid"]).size)
        counts["T_unresolved"] += int((~np.asarray(targets["T_valid"], dtype=bool)).sum())
        counts["Y_labels"] += int(np.asarray(targets["Y_valid"]).size)
        counts["Y_unresolved"] += int((~np.asarray(targets["Y_valid"], dtype=bool)).sum())
    return counts


def fixture_equivalence(data_root, models, normalization):
    """Pre-access: the test-path materializer reproduces frozen ranker records on fixtures.

    Structural fields, targets, and stored candidate values must be exact; node features recomputed
    from geometry for non-candidate parts must match within the frozen materializer's 1e-7 tolerance;
    and the frozen checkpoints must give identical threshold decisions on both record sets. The
    split-relabelled test path must yield identical arrays."""
    fixture_root = run_root(data_root) / "fixtures" / "closure"
    rows = dataset_rows(fixture_root / "fast-data" / DATASET_INDEX)
    graph_audit = common.read_json(common.closure_root(data_root) / "fast-data" / closure.GRAPH_AUDIT)
    inventory = {row["assembly_id"]: row for row in graph_audit["inventory"]}
    build = feature_build(data_root)
    shards = {Path(shard["path"]).stem: shard for shard in build["shards"] if shard["dataset_split"] in ("train", "validation")}
    prepared = _prepared(data_root)
    results = []
    for assembly_id in fixture_ids(data_root):
        shard, row = shards[assembly_id], rows[("training_assembly", assembly_id)]
        entry = {"index": 0, "assembly_id": assembly_id, "dataset_split": row["dataset_split"], "source_split": "training_assembly",
                 "tier": row["tier"], "feature_shard_path": shard["path"], "feature_shard_sha256": shard["sha256"],
                 "candidate_count": shard["candidate_count"], "action_count": shard["action_count"], "input_sha256": row["input_sha256"],
                 "label_summary_path": label_summary_path("training_assembly", assembly_id)}
        records, _ = materialize_assembly(prepared, entry, row, fixture_root / "fast-data", fixture_root / "asap-data")
        rebuilt = json.loads(common.canonical_json(records))
        frozen = common.read_jsonl(common.closure_root(data_root) / "fast-data" / closure.GRAPH_ROOT / inventory[assembly_id]["path"])[1:-1]
        structural, candidate_exact, within, cells, worst = len(rebuilt) == len(frozen), True, True, 0, 0.0
        for new_record, old_record in zip(rebuilt, frozen):
            structural = structural and all(new_record[key] == old_record[key] for key in new_record if key != "node_x") and set(new_record) == set(old_record)
            new_x, old_x = np.asarray(new_record["node_x"], dtype=np.float64), np.asarray(old_record["node_x"], dtype=np.float64)
            candidates = np.asarray(new_record["candidate_index"], dtype=np.int64)
            candidate_exact = candidate_exact and new_x.shape == old_x.shape and np.array_equal(new_x[candidates], old_x[candidates])
            within = within and new_x.shape == old_x.shape and bool(np.allclose(new_x, old_x, rtol=1e-7, atol=1e-7))
            if new_x.shape == old_x.shape:
                cells += int((new_x != old_x).sum())
                worst = max(worst, float(np.abs(new_x - old_x).max()) if new_x.size else 0.0)
        new_valid = [graph_contract.validate_development_record(record) for record in rebuilt]
        old_valid = [graph_contract.validate_development_record(record) for record in frozen]
        probability_difference, flips = 0.0, 0
        for model in models:
            for a, b in zip(predict_records(model, new_valid, normalization), predict_records(model, old_valid, normalization)):
                for key in ("R_score", "T_score", "Y_score"):
                    probability_difference = max(probability_difference, float(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64)).max()))
                    flips += int(((a[key] >= offline.THRESHOLD) != (b[key] >= offline.THRESHOLD)).sum())
        relabelled = [validate_test_record(dict(record, dataset_split=TEST_SPLIT, source_split=TEST_SOURCE)) for record in records]
        original = [graph_contract.validate_development_record(record) for record in records]
        arrays_identical = all(
            all(np.array_equal(a[name], b[name]) for name in ("node_x", "edge_index", "candidate_index", "action_x"))
            and all(np.array_equal(a["targets"][name], b["targets"][name]) for name in a["targets"])
            for a, b in zip(relabelled, original))
        results.append(OrderedDict([
            ("tier", row["tier"]), ("dataset_split", row["dataset_split"]), ("contexts", len(records)),
            ("byte_identical_to_frozen_ranker_records", rebuilt == frozen), ("structural_and_target_fields_exact", structural),
            ("candidate_node_features_exact", candidate_exact), ("non_candidate_recomputation_within_frozen_1e-7", within),
            ("differing_node_feature_cells", cells), ("max_abs_node_feature_difference", worst),
            ("max_abs_probability_difference_three_seeds", probability_difference), ("threshold_decision_flips", flips),
            ("test_path_arrays_identical", arrays_identical),
        ]))
    return results


def fixture_digest(data_root):
    """SHA-256 of all re-materialized fixture records; used to prove fresh-process reproducibility."""
    require_fixed_hash_seed()
    fixture_root = run_root(data_root) / "fixtures" / "closure"
    rows = dataset_rows(fixture_root / "fast-data" / DATASET_INDEX)
    shards = {Path(shard["path"]).stem: shard for shard in feature_build(data_root)["shards"] if shard["dataset_split"] in ("train", "validation")}
    prepared, digest = _prepared(data_root), hashlib.sha256()
    for assembly_id in fixture_ids(data_root):
        shard, row = shards[assembly_id], rows[("training_assembly", assembly_id)]
        entry = {"index": 0, "assembly_id": assembly_id, "dataset_split": row["dataset_split"], "source_split": "training_assembly",
                 "tier": row["tier"], "feature_shard_path": shard["path"], "feature_shard_sha256": shard["sha256"],
                 "candidate_count": shard["candidate_count"], "action_count": shard["action_count"], "input_sha256": row["input_sha256"],
                 "label_summary_path": label_summary_path("training_assembly", assembly_id)}
        records, _ = materialize_assembly(prepared, entry, row, fixture_root / "fast-data", fixture_root / "asap-data")
        digest.update(common.canonical_json(records).encode("utf-8"))
    return {"fixture_records_sha256": digest.hexdigest()}


def _materialize_one(arguments):
    data_root, entry, row = arguments
    prepared = _prepared(data_root)
    root = run_root(data_root)
    output = root / "test-graphs" / "{:03d}.jsonl".format(entry["index"])
    started = time.perf_counter()
    records, counts = materialize_assembly(prepared, entry, row, root / "closure" / "fast-data", root / "closure" / "asap-data", output)
    nulls = _null_counts(records)
    return entry["index"], entry["tier"], counts, dict(nulls), common.sha256_file(output), time.perf_counter() - started


# ---------------------------------------------------------------- prediction and quality


def predict_records(model, records, normalization):
    """ranker_train.predict arithmetic over already-validated records (8-assembly batches, CPU)."""
    by_assembly = OrderedDict()
    for record in records:
        by_assembly.setdefault(record["assembly_id"], []).append(record)
    assemblies = list(by_assembly)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(assemblies), 8):
            batch_records = [record for assembly in assemblies[start:start + 8] for record in by_assembly[assembly]]
            batch = ranker_models.torch_batch([development_shape(record) for record in batch_records], normalization, torch.device("cpu"))
            outputs = model(batch)
            probabilities = ranker_models.probabilities(outputs)
            for context, record in enumerate(batch_records):
                begin, end = int(batch["candidate_ptr"][context]), int(batch["candidate_ptr"][context + 1])
                target = record["targets"]
                predictions.append({
                    "assembly_id": record["assembly_id"], "tier": record["tier"], "query_id": record["query_id"],
                    "R": target["R"], "R_valid": target["R_valid"], "T": target["T"], "T_valid": target["T_valid"],
                    "Y": target["Y"], "Y_valid": target["Y_valid"],
                    "fully_observed": bool(target["R_valid"].all() and target["T_valid"].all()),
                    "Y_score": probabilities["Y"][begin:end].numpy(), "Y_logit": None,
                    "R_score": probabilities["R"][begin:end].numpy(), "R_logit": outputs["R"][begin:end].numpy(),
                    "T_score": probabilities["T"][begin:end].numpy(), "T_logit": outputs["T"][begin:end].numpy(),
                })
    return predictions


def undefined_reasons(rows):
    for target_rows in rows.values():
        for row in target_rows.values():
            counts, reasons = row["counts"], OrderedDict()
            resolved = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
            if resolved == 0:
                reasons["accuracy_percent"] = "no resolved labels"
            if counts["TP"] + counts["FP"] == 0:
                reasons["precision_percent"] = "no predicted positives among resolved labels"
            if counts["TP"] + counts["FN"] == 0:
                reasons["recall_percent"] = "no positive resolved labels"
                reasons["average_precision"] = "no positive resolved labels"
            if counts["TP"] + counts["FN"] == 0 or counts["TN"] + counts["FP"] == 0:
                reasons["auroc"] = "resolved labels contain one class"
            row["undefined_reasons"] = reasons
    return rows


def _quality(predictions_by_seed):
    per_seed = OrderedDict((str(seed), offline.prediction_quality(predictions_by_seed[seed])) for seed in offline.SEEDS)
    return per_seed, offline._mean_quality(per_seed)


def _validation_section(path):
    frozen = common.read_json(path)
    return common.canonical_json({"seed_mean": frozen["seed_mean"], "per_seed": frozen["per_seed"]})


# ---------------------------------------------------------------- timing gate


def timing_gate(data_root):
    """Test access requires a recorded D continue/stop decision, so no D choice can use the test result."""
    decision = run_root(data_root) / "d-decision.json"
    common.require(decision.exists(), "the D continue/stop decision is not recorded")
    return OrderedDict([("mode", "d_decision_recorded"), ("d_decision_sha256", common.sha256_file(decision)),
                        ("d_jobs_running", len(runner.running(common.run_root(data_root)))), ("checked_at", common.utc_now())])


def record_d_decision(data_root, decision, statement):
    common.require(decision in ("continue", "stop"), "D decision must be continue or stop")
    common.require(statement.strip(), "a decision statement is required")
    common.require(not (run_root(data_root) / "test-prediction-quality.json").exists(), "the decision must precede test unblinding")
    return common.write_json_once(run_root(data_root) / "d-decision.json", {
        "decision": decision, "statement": statement, "recorded_at": common.utc_now(), "test_aggregate_seen": False})


# ---------------------------------------------------------------- commands


def specify(data_root):
    """Publish the immutable specification and sealed registry before any test path is opened."""
    root = run_root(data_root)
    spec, spec_sha256 = load_spec()
    identity = common.implementation_identity(require_clean=True)
    rows = dataset_rows(root / "fixtures" / "closure" / "fast-data" / DATASET_INDEX)
    entries = build_registry(feature_build(data_root), rows, spec)
    registry = {"schema_version": SCHEMA_VERSION, "run_id": RUN_ID, "sealed": True, "entries": entries}
    if (root / "registry.json").exists():
        common.require(common.canonical_json(common.read_json(root / "registry.json")) == common.canonical_json(registry),
                       "existing sealed registry differs from the frozen rebuild")
        registry_sha256 = common.sha256_file(root / "registry.json")
    else:
        registry_sha256 = common.write_json_once(root / "registry.json", registry)
    specification = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("run_id", RUN_ID), ("created_at", common.utc_now()), ("implementation", identity),
        ("specification_manifest_sha256", spec_sha256), ("specification", spec), ("implementation_file_sha256", implementation_hashes()),
        ("registry_sha256", registry_sha256), ("registry_sealed", True),
        ("population", OrderedDict([("assembly_count", len(entries)), ("tier_counts", dict(Counter(entry["tier"] for entry in entries))),
                                    ("feature_candidate_count", sum(entry["candidate_count"] for entry in entries)),
                                    ("feature_action_count", sum(entry["action_count"] for entry in entries))])),
        ("registered_run_root_entries", list(EXPORTED_OUTPUTS) + list(SEALED_ENTRIES)),
        ("test_paths_opened", 0),
    ])
    return {"specification.json": publish(data_root, "specification.json", specification), "registry_sha256": registry_sha256}


def preflight(data_root):
    """Pre-access gate using only synthetic, training, and validation fixtures."""
    require_fixed_hash_seed()
    root = run_root(data_root)
    spec, spec_sha256 = load_spec()
    identity = common.implementation_identity(require_clean=True)
    specification = common.read_json(root / "specification.json")
    common.require(specification["specification_manifest_sha256"] == spec_sha256
                   and specification["implementation_file_sha256"] == implementation_hashes(), "specification binding differs")
    checks = OrderedDict()

    context = offline.load_context(data_root)
    common.require(common.sha256_file(context["data"] / closure.GRAPH_AUDIT) == spec["frozen_inputs"]["ranker_graph_audit_normalization"]["sha256"], "normalization source differs")
    ranker_train.configure_determinism(0)
    predictions_by_seed, checkpoints = {}, {}
    validation_records = list(offline.validation_records(context))
    for seed in offline.SEEDS:
        model, checkpoints[seed] = offline._load_frozen(context, "graph_transformer", seed)
        common.require(checkpoints[seed]["sha256"] == spec["frozen_inputs"]["checkpoints"][str(seed)]["sha256"], "checkpoint differs")
        predictions_by_seed[seed] = predict_records(model, validation_records, context["audit"]["normalization"])
        frozen_predictions, _ = ranker_train.predict(model, context["prepared"], context["validation_rows"], torch.device("cpu"))
        common.require(all(np.array_equal(a["Y_score"], b["Y_score"]) and np.array_equal(a["R_score"], b["R_score"]) and np.array_equal(a["T_score"], b["T_score"])
                           for a, b in zip(predictions_by_seed[seed], frozen_predictions)), "test evaluator predictions differ from the frozen predict")
    per_seed, seed_mean = _quality(predictions_by_seed)
    frozen_path = context["run_root"] / "prediction-quality.json"
    common.require(common.sha256_file(frozen_path) == spec["frozen_inputs"]["validation_experiment_e_output"]["sha256"], "frozen validation E output differs")
    reproduced = common.canonical_json({"seed_mean": seed_mean, "per_seed": per_seed})
    checks["validation_E_tables_byte_identical"] = reproduced == _validation_section(frozen_path)
    checks["validation_E_tables_sha256"] = hashlib.sha256(reproduced.encode("utf-8")).hexdigest()
    checks["test_evaluator_predictions_equal_frozen_predict"] = True

    try:
        graph_contract.validate_development_record({"dataset_split": "test"})
        checks["development_loader_refuses_test_row"] = False
    except ValueError as error:
        checks["development_loader_refuses_test_row"] = "rejects test" in str(error)
    try:
        validate_test_record(dict(validation_records[0]))
        checks["test_evaluator_refuses_non_test_row"] = False
    except ValueError:
        checks["test_evaluator_refuses_non_test_row"] = True
    checks["schema_and_feature_widths"] = ((len(cad_features.CANDIDATE_FEATURES), len(cad_features.ACTION_FEATURES)) == (41, 16)
                                           and sum(p.numel() for p in ranker_models.build_model("graph_transformer").parameters()) == 90142)
    checks["normalization_shape"] = (len(context["audit"]["normalization"]["node"]["mean"]), len(context["audit"]["normalization"]["action"]["mean"])) == (41, 16)
    registry = common.read_json(root / "registry.json")
    checks["registry_valid_sealed_and_tier_counts"] = validate_registry(registry["entries"], spec) and registry["sealed"] is True

    unit = subprocess.run([sys.executable, "-m", "unittest", "tests.test_prediction_quality"], cwd=str(common.REPOSITORY_ROOT),
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    match = re.search(r"Ran (\d+) tests", unit.stdout)
    checks["unit_tests_passed"] = unit.returncode == 0
    checks["unit_tests_run"] = int(match.group(1)) if match else 0

    protected = [context["data"] / DATASET_INDEX / "manifest.json", context["data"] / FEATURE_ROOT / "manifest.json",
                 context["data"] / closure.LABEL_ROOT / closure.REPLAY_FIXTURE / "summary.json",
                 offline._checkpoint_path(context, closure.RANKER_RUN, "graph_transformer", 0, closure.FAST_CHECKPOINTS[0]),
                 root / "fixtures" / "closure" / "fast-data" / DATASET_INDEX / "assemblies.jsonl"]
    checks["dry_run_source_paths_not_writable"] = all(write_refused(path) for path in protected)
    try:
        safe_output(data_root, "../component-ablation-v1/prediction-quality.json")
        checks["dry_run_output_escape_refused"] = False
    except ValueError:
        checks["dry_run_output_escape_refused"] = True
    models = [offline._load_frozen(context, "graph_transformer", seed)[0] for seed in offline.SEEDS]
    fixtures_report = fixture_equivalence(data_root, models, context["audit"]["normalization"])
    checks["fixture_structural_target_and_candidate_values_exact"] = all(
        row["structural_and_target_fields_exact"] and row["candidate_node_features_exact"] for row in fixtures_report)
    checks["fixture_non_candidate_recomputation_within_frozen_tolerance"] = all(row["non_candidate_recomputation_within_frozen_1e-7"] for row in fixtures_report)
    checks["fixture_threshold_decisions_identical"] = all(row["threshold_decision_flips"] == 0 for row in fixtures_report)
    checks["fixture_test_path_arrays_identical"] = all(row["test_path_arrays_identical"] for row in fixtures_report)
    checks["fixture_tiers_covered"] = sorted({row["tier"] for row in fixtures_report}) == sorted(common.TIERS)
    digests = []
    for _ in range(2):
        fresh = subprocess.run([sys.executable, "-m", "fast.prediction_quality", "--data-root", str(data_root), "fixture-digest"],
                               cwd=str(common.REPOSITORY_ROOT), env=dict(os.environ, PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES=""),
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        digests.append(json.loads(fresh.stdout.strip().splitlines()[-1])["fixture_records_sha256"] if fresh.returncode == 0 else None)
    checks["fixture_materialization_reproducible_in_fresh_processes"] = digests[0] is not None and digests[0] == digests[1]
    information = OrderedDict([
        ("fixed_hash_seed", "PYTHONHASHSEED=0"), ("fresh_process_fixture_records_sha256", digests[0]),
        ("validation_E_tables_sha256", checks.pop("validation_E_tables_sha256")), ("unit_tests_run", checks.pop("unit_tests_run")),
        ("fixture_byte_identical", "{} of {}".format(sum(row["byte_identical_to_frozen_ranker_records"] for row in fixtures_report), len(fixtures_report))),
        ("fixture_max_abs_node_feature_difference", max(row["max_abs_node_feature_difference"] for row in fixtures_report)),
        ("fixture_max_abs_probability_difference", max(row["max_abs_probability_difference_three_seeds"] for row in fixtures_report)),
    ])
    passed = all(value is True for value in checks.values())
    report = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "pre_access_gate"), ("run_id", RUN_ID), ("created_at", common.utc_now()),
        ("implementation", identity), ("specification_sha256", common.sha256_file(root / "specification.json")),
        ("environment", offline.compatibility_environment()), ("checks", checks), ("information", information),
        ("fixture_materialization", fixtures_report),
        ("validation_reference", OrderedDict([
            ("frozen_output_sha256", spec["frozen_inputs"]["validation_experiment_e_output"]["sha256"]),
            ("scope", "seed_mean and per_seed tables recomputed through the test evaluator path and compared as canonical bytes"),
            ("whole_file_note", "the frozen file also carries created_at, implementation, and bound-file hashes, which cannot recur on a later run; the deterministic re-run of the complete validation output is proven by the component-ablation pretraining audit"),
        ])),
        ("checkpoints", OrderedDict((str(seed), checkpoints[seed]["sha256"]) for seed in offline.SEEDS)),
        ("status", "passed" if passed else "failed"), ("test_paths_opened", 0), ("test_rows_loaded", 0),
    ])
    digest = publish(data_root, "validation-reference-reproduction.json", report)
    common.require(passed, "pre-access gate failed; no test path may be opened")
    return {"status": "passed", "validation-reference-reproduction.json": digest, "checks": checks}


def _require_preflight(data_root):
    report = common.read_json(run_root(data_root) / "validation-reference-reproduction.json")
    common.require(report["status"] == "passed", "pre-access gate has not passed")
    return report


def transfer_test_inputs(data_root):
    """Open test paths: list, hash, transfer, and verify exactly the registered test inputs."""
    root = run_root(data_root)
    _require_preflight(data_root)
    gate = timing_gate(data_root)
    spec, _ = load_spec()
    entries = common.read_json(root / "registry.json")["entries"]
    validate_registry(entries, spec)
    requests, allowed = [], {"fast-data": [], "asap-data": []}
    for entry in entries:
        label_dir = str(Path(entry["label_summary_path"]).parent)
        requests += [("fast-data", "test_feature_shards", "file", FEATURE_ROOT + "/" + entry["feature_shard_path"]),
                     ("fast-data", "test_labels", "tree", label_dir),
                     ("asap-data", "test_geometry", "geometry", TEST_SOURCE + "/" + entry["assembly_id"])]
        allowed["fast-data"] += [FEATURE_ROOT + "/" + entry["feature_shard_path"], label_dir + "/"]
        allowed["asap-data"].append(TEST_SOURCE + "/" + entry["assembly_id"] + "/")
    started = time.perf_counter()
    files, excluded = _plan(requests, allowed)
    common.write_json_once(root / "test-transfer-manifest.json", {"schema_version": SCHEMA_VERSION, "sealed": True, "files": files,
                                                                  "geometry_excluded_file_counts": excluded})
    _transfer(files, root / "closure")
    count, size = _verify_files(files, root / "closure")
    make_read_only(root / "closure")
    items = Counter(row["item"] for row in files)
    audit = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "input_audit"), ("run_id", RUN_ID), ("created_at", common.utc_now()),
        ("implementation", common.implementation_identity(require_clean=True)), ("timing_gate", gate),
        ("registry_sha256", common.sha256_file(root / "registry.json")),
        ("test_transfer_manifest_sha256", common.sha256_file(root / "test-transfer-manifest.json")),
        ("assembly_count", len(entries)), ("tier_counts", dict(Counter(entry["tier"] for entry in entries))),
        ("file_count", count), ("total_bytes", size), ("file_counts_by_item", dict(sorted(items.items()))),
        ("geometry_excluded_file_counts", excluded), ("input_inventory_sha256", inventory_sha256(files)),
        ("unlisted_or_missing_files", 0), ("closure_read_only", True), ("transfer_seconds", time.perf_counter() - started),
        ("status", "passed"),
    ])
    return {"input-audit.json": publish(data_root, "input-audit.json", audit), "file_count": count, "total_bytes": size}


def materialize(data_root, workers=12):
    require_fixed_hash_seed()
    root = run_root(data_root)
    _require_preflight(data_root)
    gate = timing_gate(data_root)
    spec, _ = load_spec()
    common.require(common.read_json(root / "input-audit.json")["status"] == "passed", "input audit has not passed")
    entries = common.read_json(root / "registry.json")["entries"]
    validate_registry(entries, spec)
    rows = dataset_rows(root / "fixtures" / "closure" / "fast-data" / DATASET_INDEX)
    (root / "test-graphs").mkdir(exist_ok=False)
    started = time.perf_counter()
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = pool.map(_materialize_one, [(str(data_root), entry, rows[(TEST_SOURCE, entry["assembly_id"])]) for entry in entries], chunksize=1)
    wall = time.perf_counter() - started
    common.require(sorted(result[0] for result in results) == list(range(len(entries))), "materialized registry coverage differs")
    totals, nulls = OrderedDict(), OrderedDict()
    for row in ("all",) + common.TIERS:
        selected = [result for result in results if row == "all" or result[1] == row]
        totals[row] = OrderedDict((name, sum(result[2][name] for result in selected)) for name in results[0][2])
        nulls[row] = OrderedDict((name, sum(result[3].get(name, 0) for result in selected)) for name in sorted(results[0][3]))
    population = spec["population"]
    common.require(totals["all"]["candidate_count"] == population["feature_candidate_count"]
                   and totals["all"]["action_count"] == population["feature_action_count"], "materialized totals differ from the specification")
    shard_hashes = sorted(result[4] for result in results)
    audit = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "feature_materialization_audit"), ("run_id", RUN_ID), ("created_at", common.utc_now()),
        ("implementation", common.implementation_identity(require_clean=True)), ("timing_gate", gate),
        ("transformation", spec["materialization"]["transformation"]), ("split_handling", spec["materialization"]["split_handling"]),
        ("assembly_count", len(results)), ("tier_counts", dict(Counter(result[1] for result in results))),
        ("totals", totals), ("label_counts", nulls),
        ("recomputed_cad_features_checked", "every stored candidate feature vector matched recomputation within the frozen 1e-7 tolerance"),
        ("normalization_sha256", ranker_train._canonical_hash(offline.load_context(data_root)["audit"]["normalization"])),
        ("normalization_computed_from_test", False),
        ("test_graph_inventory_sha256", common.canonical_sha256(shard_hashes)), ("test_graph_shard_count", len(shard_hashes)),
        ("workers", workers), ("fixed_hash_seed", "PYTHONHASHSEED=0"), ("wall_seconds", wall), ("status", "passed"),
    ])
    make_read_only(root / "test-graphs")
    return {"feature-materialization-audit.json": publish(data_root, "feature-materialization-audit.json", audit), "wall_seconds": wall}


def _load_test_records(data_root, entries):
    root = run_root(data_root)
    records, test_rows = [], 0
    for entry in entries:
        rows = common.read_jsonl(root / "test-graphs" / "{:03d}.jsonl".format(entry["index"]))
        header, summary = rows[0], rows[-1]
        common.require(header["registry_index"] == entry["index"] and header["dataset_split"] == TEST_SPLIT and header["tier"] == entry["tier"]
                       and header["feature_shard_sha256"] == entry["feature_shard_sha256"] and summary["complete"] is True, "test graph shard differs")
        for row in rows[1:-1]:
            common.require(row["assembly_id"] == entry["assembly_id"], "test graph shard assembly differs")
            records.append(validate_test_record(row))
            test_rows += 1
    return records, test_rows


def evaluate(data_root):
    root = run_root(data_root)
    _require_preflight(data_root)
    gate = timing_gate(data_root)
    spec, _ = load_spec()
    common.require(common.read_json(root / "feature-materialization-audit.json")["status"] == "passed", "feature materialization has not passed")
    entries = common.read_json(root / "registry.json")["entries"]
    validate_registry(entries, spec)
    ranker_train.configure_determinism(0)
    context = offline.load_context(data_root)
    environment = offline.compatibility_environment()
    started = time.perf_counter()
    records, test_rows = _load_test_records(data_root, entries)
    predictions_by_seed, checkpoints = {}, OrderedDict()
    for seed in offline.SEEDS:
        model, checkpoint = offline._load_frozen(context, "graph_transformer", seed)
        common.require(checkpoint["sha256"] == spec["frozen_inputs"]["checkpoints"][str(seed)]["sha256"], "checkpoint differs")
        checkpoints[str(seed)] = checkpoint["sha256"]
        predictions_by_seed[seed] = predict_records(model, records, context["audit"]["normalization"])
    per_seed, seed_mean = _quality(predictions_by_seed)
    for rows in per_seed.values():
        undefined_reasons(rows)
    materialized = common.read_json(root / "feature-materialization-audit.json")
    for target in ("R", "T", "Y"):
        for row in ("all",) + common.TIERS:
            common.require(per_seed["0"][target][row]["counts"]["total"] == materialized["label_counts"][row][target + "_labels"]
                           and per_seed["0"][target][row]["counts"]["unresolved"] == materialized["label_counts"][row][target + "_unresolved"],
                           "evaluated label counts differ from the materialization audit")
    report = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "test_prediction_quality"), ("run_id", RUN_ID), ("created_at", common.utc_now()),
        ("implementation", common.implementation_identity(require_clean=True)), ("environment", environment), ("timing_gate", gate),
        ("specification_sha256", common.sha256_file(root / "specification.json")), ("registry_sha256", common.sha256_file(root / "registry.json")),
        ("feature_materialization_audit_sha256", common.sha256_file(root / "feature-materialization-audit.json")),
        ("checkpoints", checkpoints), ("normalization_sha256", ranker_train._canonical_hash(context["audit"]["normalization"])),
        ("threshold", spec["analysis"]["threshold"]), ("denominators", spec["analysis"]["denominators"]),
        ("null_label_name", "unresolved/censored"), ("population", OrderedDict([("assembly_count", len(entries)),
                                                                               ("tier_counts", dict(Counter(entry["tier"] for entry in entries))),
                                                                               ("context_count", len(records))])),
        ("seed_mean", seed_mean), ("per_seed", per_seed), ("evaluation_seconds", time.perf_counter() - started),
        ("post_hoc", True), ("test_rows_evaluated", test_rows), ("claim_boundary", spec["claim_boundary"]),
    ])
    return {"test-prediction-quality.json": publish(data_root, "test-prediction-quality.json", report), "status": "published"}


def final_audit(data_root):
    root = run_root(data_root)
    spec, spec_sha256 = load_spec()
    checks = OrderedDict()
    specification = common.read_json(root / "specification.json")
    checks["specification_binding"] = specification["specification_manifest_sha256"] == spec_sha256
    registry = common.read_json(root / "registry.json")
    entries = registry["entries"]
    checks["registry_hash"] = common.sha256_file(root / "registry.json") == specification["registry_sha256"]
    checks["population_237_20_128_89"] = validate_registry(entries, spec) and len(entries) == 237
    preflight_report = common.read_json(root / "validation-reference-reproduction.json")
    checks["pre_access_gate_passed_before_access"] = preflight_report["status"] == "passed" and preflight_report["test_paths_opened"] == 0
    inputs = common.read_json(root / "input-audit.json")
    manifest = common.read_json(root / "test-transfer-manifest.json")
    checks["inputs_rehashed_read_only"] = (_verify_files(manifest["files"], root / "closure")[0] == inputs["file_count"]
                                           and inputs["input_inventory_sha256"] == inventory_sha256(manifest["files"]))
    materialized = common.read_json(root / "feature-materialization-audit.json")
    shards = sorted(path for path in (root / "test-graphs").iterdir())
    checks["one_test_graph_shard_per_registry_entry"] = [path.name for path in shards] == ["{:03d}.jsonl".format(index) for index in range(len(entries))]
    checks["test_graph_inventory_hash"] = common.canonical_sha256(sorted(common.sha256_file(path) for path in shards)) == materialized["test_graph_inventory_sha256"]
    require_fixed_hash_seed()
    rows = dataset_rows(root / "fixtures" / "closure" / "fast-data" / DATASET_INDEX)
    prepared = _prepared(data_root)
    with tempfile.TemporaryDirectory(prefix="test-quality-rematerialize-") as scratch:
        sample = entries[:5]
        for entry in sample:
            materialize_assembly(prepared, entry, rows[(TEST_SOURCE, entry["assembly_id"])], root / "closure" / "fast-data",
                                 root / "closure" / "asap-data", Path(scratch) / "{:03d}.jsonl".format(entry["index"]))
        checks["rematerialized_sample_byte_identical"] = all(
            common.sha256_file(Path(scratch) / "{:03d}.jsonl".format(entry["index"])) == common.sha256_file(root / "test-graphs" / "{:03d}.jsonl".format(entry["index"]))
            for entry in sample)
    result = common.read_json(root / "test-prediction-quality.json")
    checks["three_checkpoint_hashes"] = result["checkpoints"] == {seed: value["sha256"] for seed, value in spec["frozen_inputs"]["checkpoints"].items()}
    context = offline.load_context(data_root)
    checks["training_normalization_unchanged"] = result["normalization_sha256"] == materialized["normalization_sha256"] == ranker_train._canonical_hash(context["audit"]["normalization"])
    checks["context_and_label_totals_consistent"] = result["population"]["context_count"] == materialized["totals"]["all"]["context_count"] and all(
        result["per_seed"][seed][target][row]["counts"]["total"] == materialized["label_counts"][row][target + "_labels"]
        for seed in ("0", "1", "2") for target in ("R", "T", "Y") for row in ("all",) + common.TIERS)
    checks["candidate_action_totals"] = (materialized["totals"]["all"]["candidate_count"], materialized["totals"]["all"]["action_count"]) == (
        spec["population"]["feature_candidate_count"], spec["population"]["feature_action_count"])
    checks["aggregate_only_exports"] = all(aggregate_only(common.read_json(root / name)) for name in EXPORTED_OUTPUTS if (root / name).exists())
    ablation = context["data"]
    first_input = root / "closure" / manifest["files"][0]["namespace"] / manifest["files"][0]["path"]
    checks["no_source_writes"] = all(common.sha256_file(ablation / path) == digest for path, digest in closure.FROZEN_SHA256.items()) and all(
        write_refused(path) for path in (ablation / DATASET_INDEX / "manifest.json", first_input))
    allowed = set(EXPORTED_OUTPUTS) | set(SEALED_ENTRIES)
    present = {path.name for path in root.iterdir()}
    checks["run_root_inventory_registered"] = present - {"final-audit.json"} <= allowed
    decision = common.read_json(root / "d-decision.json")
    checks["d_decision_recorded_before_unblinding"] = decision["recorded_at"] <= result["created_at"] and decision["test_aggregate_seen"] is False
    status = "passed" if all(checks.values()) else "failed"
    report = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "final_audit"), ("run_id", RUN_ID), ("audited_at", common.utc_now()),
        ("implementation", common.implementation_identity(require_clean=True)), ("checks", checks),
        ("output_sha256", OrderedDict((name, common.sha256_file(root / name)) for name in EXPORTED_OUTPUTS[:-1])),
        ("registry_sha256", common.sha256_file(root / "registry.json")), ("status", status),
    ])
    digest = publish(data_root, "final-audit.json", report)
    return {"status": status, "final-audit.json": digest, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fixtures", "specify", "preflight", "fixture-digest", "transfer", "materialize", "evaluate", "final-audit"):
        commands.add_parser(name)
    decision = commands.add_parser("record-d-decision")
    decision.add_argument("--decision", required=True, choices=("continue", "stop"))
    decision.add_argument("--statement", required=True)
    args = parser.parse_args(argv)
    handlers = {"fixtures": fixtures, "specify": specify, "preflight": preflight, "fixture-digest": fixture_digest, "transfer": transfer_test_inputs,
                "materialize": materialize, "evaluate": evaluate, "final-audit": final_audit}
    if args.command == "record-d-decision":
        result = record_d_decision(args.data_root, args.decision, args.statement)
    else:
        result = handlers[args.command](args.data_root)
    print(common.canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
