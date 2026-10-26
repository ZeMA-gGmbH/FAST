"""Materialize and audit train/validation-only ranker graph records."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from fast import cad_features, graph_contract
from plan_sequence import production_sim_labels as production


SHARD_SCHEMA_VERSION = "fast.graph-shard.v1"
SETUP_SCHEMA_VERSION = "fast.materialization-setup.v1"
AUDIT_SCHEMA_VERSION = "fast.materialization-audit.v1"
FOLD_SCHEMA_VERSION = "fast.validation-folds.v1"


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(production._canonical_json(value) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(production._canonical_json(record) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve(root, relative, name):
    path = (Path(root).resolve() / production._relative_path(relative, name)).resolve()
    graph_contract._require(production._is_within(path, root), "{} escapes its root".format(name))
    return path


def _single_jsonl(path):
    records = _read_jsonl(path)
    graph_contract._require(len(records) == 1, "expected one JSON record: {}".format(path))
    return records[0]


def _prepare(spec_path, source_root, data_root, output_root=None):
    spec_path = Path(spec_path).resolve()
    repository_root = spec_path.parent.parent
    source_root = Path(source_root).resolve()
    data_root = Path(data_root).resolve()
    graph_contract._require(source_root.is_dir(), "source root is missing")
    graph_contract._require(data_root.is_dir(), "data root is missing")
    spec = graph_contract.load_spec(spec_path)
    inputs = spec["inputs"]

    protocol_path = _resolve(repository_root, inputs["dataset_protocol"]["path"], "protocol")
    feature_config_path = _resolve(repository_root, inputs["feature_config"]["path"], "feature config")
    graph_contract._require(production._sha256_file(protocol_path) == inputs["dataset_protocol"]["sha256"], "protocol hash differs")
    graph_contract._require(production._sha256_file(feature_config_path) == inputs["feature_config"]["sha256"], "feature config hash differs")

    dataset_root = data_root / "datasets" / "fast-labels-v1"
    dataset_manifest_path = dataset_root / "manifest.json"
    graph_contract._require(production._sha256_file(dataset_manifest_path) == inputs["dataset_manifest_sha256"], "dataset manifest hash differs")
    feature_root = _resolve(data_root, spec["artifacts"]["cad_features"], "scalar feature root")
    feature_build_path = feature_root / "manifest.json"
    graph_contract._require(production._sha256_file(feature_build_path) == inputs["feature_build_sha256"], "scalar-feature build hash differs")
    feature_build = _single_jsonl(feature_build_path)
    graph_contract._require(feature_build.get("status") == "passed", "scalar-feature build did not pass")
    graph_contract._require(feature_build.get("candidate_features") == list(cad_features.CANDIDATE_FEATURES), "candidate feature schema differs")
    graph_contract._require(feature_build.get("action_features") == list(cad_features.ACTION_FEATURES), "action feature schema differs")

    output_root = Path(output_root).resolve() if output_root else _resolve(data_root, spec["artifacts"]["ranker_graphs"], "ranker graph root")
    graph_contract._require(
        not production._is_within(output_root, feature_root)
        and not production._is_within(feature_root, output_root),
        "graph output overlaps scalar-feature input",
    )
    return {
        "spec": spec,
        "spec_path": spec_path,
        "spec_sha256": production._sha256_file(spec_path),
        "repository_root": repository_root,
        "source_root": source_root,
        "data_root": data_root,
        "dataset_root": dataset_root,
        "feature_root": feature_root,
        "feature_config": _read_json(feature_config_path),
        "feature_build": feature_build,
        "feature_build_path": feature_build_path,
        "output_root": output_root,
        "materializer_sha256": production._sha256_file(Path(__file__)),
        "contract_sha256": production._sha256_file(Path(graph_contract.__file__)),
    }


def _development_jobs(prepared):
    jobs = []
    for shard in prepared["feature_build"]["shards"]:
        if shard["dataset_split"] not in graph_contract.DEVELOPMENT_SPLITS:
            continue
        parts = Path(shard["path"]).parts
        graph_contract._require(len(parts) == 3 and parts[0] == "shards", "invalid scalar-feature shard path")
        source_split = parts[1]
        assembly_id = Path(parts[2]).stem
        graph_contract._require(source_split == "training_assembly", "development job references test source")
        jobs.append({
            "dataset_split": shard["dataset_split"],
            "source_split": source_split,
            "assembly_id": assembly_id,
            "tier": shard["tier"],
            "feature_shard_path": shard["path"],
            "feature_shard_sha256": shard["sha256"],
            "candidate_count": shard["candidate_count"],
            "action_count": shard["action_count"],
        })
    jobs.sort(key=lambda row: (graph_contract.DEVELOPMENT_SPLITS.index(row["dataset_split"]), row["assembly_id"]))
    expected = prepared["spec"]["materialization"]["development"]
    graph_contract._require(len(jobs) == expected["assembly_count"], "development assembly count differs")
    graph_contract._require(sum(row["candidate_count"] for row in jobs) == expected["candidate_count"], "development candidate count differs")
    graph_contract._require(sum(row["action_count"] for row in jobs) == expected["action_count"], "development action count differs")
    for index, row in enumerate(jobs):
        row["process"] = index
    return jobs


def prepare(spec_path, source_root, data_root, output_root=None):
    prepared = _prepare(spec_path, source_root, data_root, output_root)
    jobs = _development_jobs(prepared)
    output_root = prepared["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    job_path = output_root / "development-jobs.tsv"
    lines = ["process\tdataset_split\tsource_split\tassembly_id\ttier\tfeature_shard_path\tfeature_shard_sha256\tcandidate_count\taction_count"]
    for row in jobs:
        lines.append("\t".join(str(row[name]) for name in (
            "process", "dataset_split", "source_split", "assembly_id", "tier",
            "feature_shard_path", "feature_shard_sha256", "candidate_count", "action_count",
        )))
    descriptor, temporary = tempfile.mkstemp(prefix="." + job_path.name, dir=str(output_root))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        os.replace(temporary, job_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report = {
        "schema_version": SETUP_SCHEMA_VERSION,
        "status": "passed",
        "spec_sha256": prepared["spec_sha256"],
        "dataset_manifest_sha256": prepared["spec"]["inputs"]["dataset_manifest_sha256"],
        "feature_build_sha256": prepared["spec"]["inputs"]["feature_build_sha256"],
        "materializer_sha256": prepared["materializer_sha256"],
        "contract_sha256": prepared["contract_sha256"],
        "job_table": job_path.name,
        "job_table_sha256": production._sha256_file(job_path),
        "assembly_count": len(jobs),
        "train_assembly_count": sum(row["dataset_split"] == "train" for row in jobs),
        "validation_assembly_count": sum(row["dataset_split"] == "validation" for row in jobs),
        "candidate_count": sum(row["candidate_count"] for row in jobs),
        "action_count": sum(row["action_count"] for row in jobs),
        "test_rows_loaded": 0,
    }
    _write_json(output_root / "setup.json", report)
    return report


def _job(prepared, assembly_id=None, job_index=None):
    jobs = _development_jobs(prepared)
    if job_index is not None:
        graph_contract._require(0 <= job_index < len(jobs), "job index is out of range")
        return jobs[job_index]
    matches = [row for row in jobs if row["source_split"] == "training_assembly" and row["assembly_id"] == assembly_id]
    graph_contract._require(len(matches) == 1, "development assembly is absent or duplicated")
    return matches[0]


def _dataset_row(prepared, job):
    matches = []
    with (prepared["dataset_root"] / "assemblies.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source_split"] == job["source_split"] and row["assembly_id"] == job["assembly_id"]:
                matches.append(row)
    graph_contract._require(len(matches) == 1, "dataset assembly is absent or duplicated")
    row = matches[0]
    graph_contract._require(row["dataset_split"] == job["dataset_split"] and row["tier"] == job["tier"] and row["included"], "dataset assembly metadata differs")
    return row


def _serializable_record(record):
    return {
        **{name: record[name] for name in (
            "schema_version", "query_id", "dataset_split", "source_split", "assembly_id",
            "tier", "state_id", "pose_id", "input_sha256", "feature_shard_sha256",
            "feature_extractor_sha256", "part_ids",
        )},
        **{name: record[name].tolist() for name in (
            "node_x", "edge_index", "candidate_index", "candidate_mask", "base_mask", "action_x",
        )},
        "targets": {name: value.tolist() for name, value in record["targets"].items()},
    }


def _context_record(prepared, job, row, feature_records, raw_headers, static_parts, graph, full_volume):
    first = feature_records[0]
    headers = [raw_headers[record["candidate_id"]] for record in feature_records]
    first_header = headers[0]
    for header in headers[1:]:
        graph_contract._require(
            header["parts"] == first_header["parts"]
            and header["pose"] == first_header["pose"]
            and header["state_id"] == first_header["state_id"]
            and header["pose_id"] == first_header["pose_id"],
            "candidate context ownership differs",
        )
    part_ids = sorted(str(part_id) for part_id in first_header["parts"])
    posed = cad_features._posed_parts(static_parts, first_header["pose"], {})
    stored = {record["part_move"]: record for record in feature_records}
    node_features = []
    for part_id in part_ids:
        hypothetical = {
            "parts": part_ids,
            "parts_after": [other for other in part_ids if other != part_id],
            "part_move": part_id,
        }
        computed, _ = cad_features._candidate_features(
            hypothetical,
            static_parts,
            posed,
            graph,
            full_volume,
            prepared["feature_config"]["support_tolerance_fraction"],
        )
        values = np.asarray([computed[name] for name in cad_features.CANDIDATE_FEATURES], dtype=np.float32)
        if part_id in stored:
            frozen = np.asarray([stored[part_id]["features"][name] for name in cad_features.CANDIDATE_FEATURES], dtype=np.float32)
            graph_contract._require(np.allclose(values, frozen, rtol=1e-7, atol=1e-7), "recomputed scalar features differ")
            values = frozen
        node_features.append(values)

    node_index = {part_id: index for index, part_id in enumerate(part_ids)}
    candidate_index = np.asarray([node_index[record["part_move"]] for record in feature_records], dtype=np.int64)
    candidate_mask = np.zeros(len(part_ids), dtype=bool)
    candidate_mask[candidate_index] = True
    edges = []
    subgraph = graph.subgraph(part_ids)
    for source, target in subgraph.edges:
        source_index, target_index = node_index[str(source)], node_index[str(target)]
        edges.extend(((source_index, target_index), (target_index, source_index)))
    edges = sorted(edges)
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)

    action_x = np.asarray([
        [[action["features"][name] for name in cad_features.ACTION_FEATURES] for action in record["actions"]]
        for record in feature_records
    ], dtype=np.float32)
    t_valid = np.asarray([record["targets"]["T"]["valid"] for record in feature_records], dtype=bool)
    t = np.asarray([float(record["targets"]["T"]["value"] or False) for record in feature_records], dtype=np.float32)
    r_valid = np.asarray([[action["targets"]["R"]["valid"] for action in record["actions"]] for record in feature_records], dtype=bool)
    r = np.asarray([[float(action["targets"]["R"]["value"] or False) for action in record["actions"]] for record in feature_records], dtype=np.float32)
    y_valid = np.asarray([[action["targets"]["Y"]["valid"] for action in record["actions"]] for record in feature_records], dtype=bool)
    y = np.asarray([[float(action["targets"]["Y"]["value"] or False) for action in record["actions"]] for record in feature_records], dtype=np.float32)
    record = {
        "schema_version": graph_contract.RECORD_SCHEMA_VERSION,
        "query_id": first["query_id"],
        "dataset_split": job["dataset_split"],
        "source_split": job["source_split"],
        "assembly_id": job["assembly_id"],
        "tier": job["tier"],
        "state_id": first["state_id"],
        "pose_id": first["pose_id"],
        "input_sha256": row["input_sha256"],
        "feature_shard_sha256": job["feature_shard_sha256"],
        "feature_extractor_sha256": prepared["feature_build"]["extractor_sha256"],
        "part_ids": part_ids,
        "node_x": np.asarray(node_features, dtype=np.float32),
        "edge_index": edge_index,
        "candidate_index": candidate_index,
        "candidate_mask": candidate_mask,
        "base_mask": np.zeros(len(part_ids), dtype=bool),
        "action_x": action_x,
        "targets": {"R": r, "R_valid": r_valid, "T": t, "T_valid": t_valid, "Y": y, "Y_valid": y_valid},
    }
    return _serializable_record(graph_contract.validate_development_record(record))


def _shard_path(prepared, job, pilot):
    if pilot:
        return prepared["output_root"] / "pilots" / (job["assembly_id"] + ".jsonl")
    return prepared["output_root"] / "shards" / job["dataset_split"] / job["source_split"] / (job["assembly_id"] + ".jsonl")


def _validate_shard(path, prepared, job):
    records = _read_jsonl(path)
    graph_contract._require(len(records) >= 3, "graph shard is empty")
    header, summary = records[0], records[-1]
    contexts = records[1:-1]
    graph_contract._require(
        header.get("schema_version") == SHARD_SCHEMA_VERSION
        and header.get("record_type") == "header"
        and summary.get("schema_version") == SHARD_SCHEMA_VERSION
        and summary.get("record_type") == "summary"
        and summary.get("complete") is True,
        "invalid graph shard envelope",
    )
    graph_contract._require(
        header.get("spec_sha256") == prepared["spec_sha256"]
        and header.get("materializer_sha256") == prepared["materializer_sha256"]
        and header.get("contract_sha256") == prepared["contract_sha256"]
        and header.get("feature_shard_sha256") == job["feature_shard_sha256"],
        "graph shard provenance differs",
    )
    graph_contract._require(
        header.get("dataset_split") == job["dataset_split"]
        and header.get("source_split") == job["source_split"]
        and header.get("assembly_id") == job["assembly_id"]
        and header.get("tier") == job["tier"],
        "graph shard identity differs",
    )
    validated = [graph_contract.validate_development_record(record) for record in contexts]
    counts = {
        "context_count": len(validated),
        "node_count": sum(len(record["node_x"]) for record in validated),
        "edge_count": sum(record["edge_index"].shape[1] // 2 for record in validated),
        "candidate_count": sum(len(record["candidate_index"]) for record in validated),
        "action_count": 6 * sum(len(record["candidate_index"]) for record in validated),
    }
    graph_contract._require(summary.get("counts") == counts, "graph shard counts differ")
    graph_contract._require(counts["candidate_count"] == job["candidate_count"] and counts["action_count"] == job["action_count"], "graph shard scalar-feature totals differ")
    return validated, counts


def materialize(spec_path, source_root, data_root, output_root=None, assembly_id=None, job_index=None, pilot=False):
    prepared = _prepare(spec_path, source_root, data_root, output_root)
    job = _job(prepared, assembly_id, job_index)
    if pilot:
        graph_contract._require(job["assembly_id"] in prepared["spec"]["materialization"]["pilot_training_assemblies"], "assembly is not in the frozen pilot")
    output_path = _shard_path(prepared, job, pilot)
    if output_path.is_file():
        _, counts = _validate_shard(output_path, prepared, job)
        return {"status": "already_complete", "path": str(output_path), **counts, "test_rows_loaded": 0}

    row = _dataset_row(prepared, job)
    assembly_dir = prepared["source_root"] / job["source_split"] / job["assembly_id"]
    graph_contract._require(assembly_dir.is_dir(), "source assembly is missing")
    graph_contract._require(production._hash_assembly(assembly_dir) == row["input_sha256"], "source assembly hash differs")
    feature_path = prepared["feature_root"] / job["feature_shard_path"]
    graph_contract._require(production._sha256_file(feature_path) == job["feature_shard_sha256"], "scalar-feature shard hash differs")
    feature_records = _read_jsonl(feature_path)
    feature_header, feature_summary = feature_records[0], feature_records[-1]
    feature_records = feature_records[1:-1]
    graph_contract._require(feature_header["record_type"] == "header" and feature_summary["record_type"] == "summary", "invalid scalar-feature shard envelope")

    raw_headers = {}
    for header, _, _ in cad_features.dataset_index.index_candidates(prepared["dataset_root"], row):
        candidate_id = "{}:{}:{}:{}:{}".format(job["source_split"], job["assembly_id"], header["state_id"], header["pose_id"], header["part_move"])
        raw_headers[candidate_id] = header
    graph_contract._require(len(raw_headers) == len(feature_records), "raw and scalar-feature candidate counts differ")

    static_parts = cad_features._load_parts(assembly_dir)
    graph = cad_features._load_contact_graph(assembly_dir / "contact_graph.json", sorted(static_parts))
    full_volume = sum(part["volume"] for part in static_parts.values())
    contexts = OrderedDict()
    for record in feature_records:
        contexts.setdefault(record["query_id"], []).append(record)
    graph_records = [
        _context_record(prepared, job, row, records, raw_headers, static_parts, graph, full_volume)
        for records in contexts.values()
    ]
    counts = {
        "context_count": len(graph_records),
        "node_count": sum(len(record["node_x"]) for record in graph_records),
        "edge_count": sum(len(record["edge_index"][0]) // 2 for record in graph_records),
        "candidate_count": sum(len(record["candidate_index"]) for record in graph_records),
        "action_count": 6 * sum(len(record["candidate_index"]) for record in graph_records),
    }
    header = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "record_type": "header",
        "spec_sha256": prepared["spec_sha256"],
        "materializer_sha256": prepared["materializer_sha256"],
        "contract_sha256": prepared["contract_sha256"],
        "feature_build_sha256": prepared["spec"]["inputs"]["feature_build_sha256"],
        "feature_shard_sha256": job["feature_shard_sha256"],
        "dataset_split": job["dataset_split"],
        "source_split": job["source_split"],
        "assembly_id": job["assembly_id"],
        "tier": job["tier"],
        "input_sha256": row["input_sha256"],
        "pilot": bool(pilot),
    }
    summary = {"schema_version": SHARD_SCHEMA_VERSION, "record_type": "summary", "complete": True, "counts": counts, "test_rows_loaded": 0}
    _write_jsonl(output_path, [header] + graph_records + [summary])
    _validate_shard(output_path, prepared, job)
    return {"status": "complete", "path": str(output_path), "sha256": production._sha256_file(output_path), **counts, "test_rows_loaded": 0}


def _normalization_update(state, values):
    values = values.astype(np.float64)
    state["count"] += len(values)
    state["sum"] += values.sum(axis=0)
    state["sum_sq"] += np.square(values).sum(axis=0)


def _normalization_finish(state):
    mean = state["sum"] / state["count"]
    variance = np.maximum(state["sum_sq"] / state["count"] - np.square(mean), 0.0)
    std = np.sqrt(variance)
    zero = np.flatnonzero(std == 0.0).tolist()
    std[std == 0.0] = 1.0
    return {"count": state["count"], "mean": mean.tolist(), "std": std.tolist(), "zero_variance_columns": zero}


def _folds(validation_jobs):
    folds = {}
    for tier in ("small", "medium", "large"):
        rows = [row for row in validation_jobs if row["tier"] == tier]
        rows.sort(key=lambda row: hashlib.sha256(("ranker-calibration-v1|" + row["source_split"] + "|" + row["assembly_id"]).encode("utf-8")).hexdigest())
        for index, row in enumerate(rows):
            folds["{}:{}".format(row["source_split"], row["assembly_id"])] = index % 5
    return dict(sorted(folds.items()))


def audit(spec_path, source_root, data_root, output_root=None, pilot=False):
    prepared = _prepare(spec_path, source_root, data_root, output_root)
    jobs = _development_jobs(prepared)
    if pilot:
        pilot_ids = prepared["spec"]["materialization"]["pilot_training_assemblies"]
        jobs = [row for row in jobs if row["assembly_id"] in pilot_ids]
        graph_contract._require([row["assembly_id"] for row in jobs] == sorted(pilot_ids), "pilot job set differs")
    inventory = []
    totals = {name: 0 for name in ("context_count", "node_count", "edge_count", "candidate_count", "action_count")}
    denominators = {name: 0 for name in ("rank_context", "rank_context_direct", "R", "T", "Y")}
    component_assemblies = {name: 0 for name in ("rank", "rank_direct", "R", "T", "Y", "Y_direct")}
    node_norm = {"count": 0, "sum": np.zeros(41), "sum_sq": np.zeros(41)}
    action_norm = {"count": 0, "sum": np.zeros(16), "sum_sq": np.zeros(16)}
    for job in jobs:
        path = _shard_path(prepared, job, pilot)
        graph_contract._require(path.is_file(), "graph shard is missing: {}".format(path))
        records, counts = _validate_shard(path, prepared, job)
        for name, count in counts.items():
            totals[name] += count
        assembly_defined = {name: False for name in component_assemblies}
        for record in records:
            targets = record["targets"]
            strict_y = targets["R_valid"] & targets["T_valid"][:, None]
            denominators["R"] += int(targets["R_valid"].sum())
            denominators["T"] += int(targets["T_valid"].sum())
            denominators["Y"] += int(strict_y.sum())
            assembly_defined["R"] |= bool(targets["R_valid"].any())
            assembly_defined["T"] |= bool(targets["T_valid"].any())
            assembly_defined["Y"] |= bool(strict_y.any())
            mixed = targets["R_valid"].all() and targets["T_valid"].all() and targets["Y"].any() and not targets["Y"].all()
            mixed_direct = targets["Y_valid"].all() and targets["Y"].any() and not targets["Y"].all()
            denominators["rank_context"] += int(mixed)
            denominators["rank_context_direct"] += int(mixed_direct)
            assembly_defined["rank"] |= bool(mixed)
            assembly_defined["rank_direct"] |= bool(mixed_direct)
            assembly_defined["Y_direct"] |= bool(targets["Y_valid"].any())
            if not pilot and job["dataset_split"] == "train":
                _normalization_update(node_norm, record["node_x"])
                _normalization_update(action_norm, record["action_x"].reshape(-1, 16))
        if job["dataset_split"] == "train":
            for name, defined in assembly_defined.items():
                component_assemblies[name] += int(defined)
        inventory.append({
            "dataset_split": job["dataset_split"],
            "source_split": job["source_split"],
            "assembly_id": job["assembly_id"],
            "tier": job["tier"],
            "path": path.relative_to(prepared["output_root"]).as_posix(),
            "sha256": production._sha256_file(path),
            **counts,
        })

    if not pilot:
        expected = prepared["spec"]["materialization"]["development"]
        graph_contract._require(len(inventory) == expected["assembly_count"], "audit assembly count differs")
        graph_contract._require(totals["candidate_count"] == expected["candidate_count"], "audit candidate count differs")
        graph_contract._require(totals["action_count"] == expected["action_count"], "audit action count differs")
    fold_payload = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "method": "tier_stratified_sha256_round_robin",
        "fold_count": 5,
        "assignments": _folds([row for row in jobs if row["dataset_split"] == "validation"]),
    }
    fold_path = prepared["output_root"] / ("pilot-folds.json" if pilot else "validation-folds.json")
    _write_json(fold_path, fold_payload)
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "pilot": bool(pilot),
        "spec_sha256": prepared["spec_sha256"],
        "dataset_manifest_sha256": prepared["spec"]["inputs"]["dataset_manifest_sha256"],
        "feature_build_sha256": prepared["spec"]["inputs"]["feature_build_sha256"],
        "materializer_sha256": prepared["materializer_sha256"],
        "contract_sha256": prepared["contract_sha256"],
        "assembly_count": len(inventory),
        **totals,
        "denominators": denominators,
        "component_assembly_counts": {"total": sum(row["dataset_split"] == "train" for row in jobs), **component_assemblies},
        "normalization": None if pilot else {"node": _normalization_finish(node_norm), "action": _normalization_finish(action_norm)},
        "validation_folds_path": fold_path.name,
        "validation_folds_sha256": production._sha256_file(fold_path),
        "inventory": inventory,
        "test_rows_loaded": 0,
    }
    output = prepared["output_root"] / ("pilot-audit.json" if pilot else "audit.json")
    _write_json(output, report)
    return report


def _parser():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(root / "configs" / "ranker_models.json"))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    materialize_parser = subparsers.add_parser("materialize")
    identity = materialize_parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--assembly-id")
    identity.add_argument("--job-index", type=int)
    materialize_parser.add_argument("--pilot", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    common = (args.spec, args.source_root, args.data_root, args.output_root)
    if args.command == "prepare":
        result = prepare(*common)
    elif args.command == "materialize":
        result = materialize(*common, args.assembly_id, args.job_index, args.pilot)
    else:
        result = audit(*common, args.pilot)
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
