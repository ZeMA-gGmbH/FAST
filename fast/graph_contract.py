"""Executable ranker graph-record contract and reference losses."""

import json
from pathlib import Path

import numpy as np

from fast.cad_features import ACTION_FEATURES, CANDIDATE_FEATURES, dataset_index


SPEC_SCHEMA_VERSION = "fast.model-spec.v1"
RECORD_SCHEMA_VERSION = "fast.graph-record.v1"
DEVELOPMENT_SPLITS = ("train", "validation")
DIRECTION_IDS = tuple(item[0] for item in dataset_index.DIRECTIONS)
DIRECTION_VECTORS = np.asarray([item[1] for item in dataset_index.DIRECTIONS], dtype=np.float32)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_hash(value, name):
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        "{} must be lowercase SHA-256".format(name),
    )


def load_spec(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        spec = json.load(stream)
    _require(spec.get("schema_version") == SPEC_SCHEMA_VERSION, "invalid ranker spec schema")
    _require(spec.get("status") == "frozen", "ranker spec is not frozen")
    record = spec.get("record", {})
    _require(record.get("schema_version") == RECORD_SCHEMA_VERSION, "record schema differs")
    _require(record.get("candidate_feature_count") == len(CANDIDATE_FEATURES), "candidate feature count differs")
    _require(record.get("action_feature_count") == len(ACTION_FEATURES), "action feature count differs")
    _require(tuple(record.get("directions", ())) == DIRECTION_IDS, "direction order differs")
    _require(tuple(record.get("development_splits", ())) == DEVELOPMENT_SPLITS, "development splits differ")
    _require(record.get("base_part") is None, "frozen base-part policy differs")
    return spec


def _array(value, shape, kind, name):
    array = np.asarray(value)
    _require(array.shape == shape, "{} shape differs".format(name))
    if kind == "bool":
        _require(array.dtype == np.dtype(bool), "{} must be boolean".format(name))
        return array
    if kind == "int":
        if array.size == 0:
            return array.astype(np.int64)
        _require(array.dtype.kind in "iu" and array.dtype != np.dtype(bool), "{} must be integer".format(name))
        return array.astype(np.int64, copy=False)
    _require(array.dtype.kind in "biuf" and np.isfinite(array).all(), "{} must be finite numeric".format(name))
    return array.astype(np.float32, copy=False)


def _logits(value, shape, name):
    array = np.asarray(value)
    _require(
        array.shape == shape and array.dtype.kind in "biuf" and np.isfinite(array).all(),
        "{} must be finite numeric with the expected shape".format(name),
    )
    return array.astype(np.float64, copy=False)


def _targets(payload, candidate_count):
    _require(isinstance(payload, dict) and set(payload) == {"R", "R_valid", "T", "T_valid", "Y", "Y_valid"}, "target fields differ")
    result = {
        "R": _array(payload["R"], (candidate_count, 6), "float", "R"),
        "R_valid": _array(payload["R_valid"], (candidate_count, 6), "bool", "R_valid"),
        "T": _array(payload["T"], (candidate_count,), "float", "T"),
        "T_valid": _array(payload["T_valid"], (candidate_count,), "bool", "T_valid"),
        "Y": _array(payload["Y"], (candidate_count, 6), "float", "Y"),
        "Y_valid": _array(payload["Y_valid"], (candidate_count, 6), "bool", "Y_valid"),
    }
    for name in ("R", "T", "Y"):
        values, valid = result[name], result[name + "_valid"]
        _require(np.isin(values, (0.0, 1.0)).all(), "{} targets must be binary placeholders".format(name))
        _require((values[~valid] == 0.0).all(), "{} null placeholders must be zero".format(name))

    t = result["T"][:, None]
    t_valid = result["T_valid"][:, None]
    resolved_false = (result["R_valid"] & (result["R"] == 0.0)) | (t_valid & (t == 0.0))
    resolved_true = result["R_valid"] & (result["R"] == 1.0) & t_valid & (t == 1.0)
    expected_valid = resolved_false | resolved_true
    expected_y = resolved_true.astype(np.float32)
    _require(np.array_equal(result["Y_valid"], expected_valid), "Y validity differs from R/T conjunction")
    _require(np.array_equal(result["Y"], expected_y), "Y values differ from R/T conjunction")
    return result


def validate_development_record(record):
    _require(isinstance(record, dict), "graph record must be an object")
    _require(record.get("dataset_split") in DEVELOPMENT_SPLITS, "development loader rejects test rows")
    required = {
        "schema_version", "query_id", "dataset_split", "source_split", "assembly_id",
        "tier", "state_id", "pose_id", "input_sha256", "feature_shard_sha256",
        "feature_extractor_sha256", "part_ids", "node_x", "edge_index",
        "candidate_index", "candidate_mask", "base_mask", "action_x", "targets",
    }
    _require(set(record) == required, "graph record fields differ")
    _require(record["schema_version"] == RECORD_SCHEMA_VERSION, "invalid graph record schema")
    _require(record["source_split"] == "training_assembly", "development source split differs")
    _require(record["tier"] in ("small", "medium", "large"), "invalid size tier")
    for name in ("query_id", "assembly_id", "state_id", "pose_id"):
        _require(isinstance(record[name], str) and record[name], "{} must be non-empty".format(name))
    for name in ("input_sha256", "feature_shard_sha256", "feature_extractor_sha256"):
        _require_hash(record[name], name)

    part_ids = record["part_ids"]
    _require(
        isinstance(part_ids, list)
        and 2 <= len(part_ids) <= 70
        and all(isinstance(part_id, str) and part_id for part_id in part_ids)
        and part_ids == sorted(set(part_ids)),
        "part IDs must be unique and sorted",
    )
    node_count = len(part_ids)
    node_x = _array(record["node_x"], (node_count, len(CANDIDATE_FEATURES)), "float", "node_x")
    edge_index = np.asarray(record["edge_index"])
    _require(edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index shape differs")
    edge_index = _array(edge_index, edge_index.shape, "int", "edge_index")
    _require(((0 <= edge_index) & (edge_index < node_count)).all(), "edge index is out of range")
    edges = [tuple(edge) for edge in edge_index.T.tolist()]
    _require(all(source != target for source, target in edges), "stored self-loop is forbidden")
    _require(edges == sorted(set(edges)), "edges must be sorted and duplicate-free")
    _require(all((target, source) in set(edges) for source, target in edges), "contact edges must be bidirectional")

    candidate_index = np.asarray(record["candidate_index"])
    _require(candidate_index.ndim == 1, "candidate_index shape differs")
    candidate_index = _array(candidate_index, candidate_index.shape, "int", "candidate_index")
    candidate_count = len(candidate_index)
    _require(1 <= candidate_count <= node_count, "candidate count is invalid")
    _require(len(set(candidate_index.tolist())) == candidate_count, "candidate indices must be unique")
    _require(((0 <= candidate_index) & (candidate_index < node_count)).all(), "candidate index is out of range")
    candidate_mask = _array(record["candidate_mask"], (node_count,), "bool", "candidate_mask")
    expected_mask = np.zeros(node_count, dtype=bool)
    expected_mask[candidate_index] = True
    _require(np.array_equal(candidate_mask, expected_mask), "candidate mask differs from candidate indices")
    base_mask = _array(record["base_mask"], (node_count,), "bool", "base_mask")
    _require(not base_mask.any(), "frozen base mask must be empty")

    action_x = _array(record["action_x"], (candidate_count, 6, len(ACTION_FEATURES)), "float", "action_x")
    _require(
        np.array_equal(action_x[:, :, :3], np.broadcast_to(DIRECTION_VECTORS, (candidate_count, 6, 3))),
        "action direction order or values differ",
    )
    targets = _targets(record["targets"], candidate_count)
    return {
        **{name: record[name] for name in (
            "schema_version", "query_id", "dataset_split", "source_split", "assembly_id",
            "tier", "state_id", "pose_id", "input_sha256", "feature_shard_sha256",
            "feature_extractor_sha256", "part_ids",
        )},
        "node_x": node_x,
        "edge_index": edge_index,
        "candidate_index": candidate_index,
        "candidate_mask": candidate_mask,
        "base_mask": base_mask,
        "action_x": action_x,
        "targets": targets,
    }


def batch_development_records(records):
    records = [validate_development_record(record) for record in records]
    _require(records, "cannot batch zero graph records")
    node_ptr = [0]
    candidate_ptr = [0]
    edges = []
    assemblies = {}
    context_assembly = []
    for context, record in enumerate(records):
        node_offset = node_ptr[-1]
        edges.append(record["edge_index"] + node_offset)
        node_ptr.append(node_offset + len(record["node_x"]))
        candidate_ptr.append(candidate_ptr[-1] + len(record["candidate_index"]))
        key = (record["source_split"], record["assembly_id"])
        context_assembly.append(assemblies.setdefault(key, len(assemblies)))

    concatenate = lambda name: np.concatenate([record[name] for record in records], axis=0)
    concatenate_target = lambda name: np.concatenate([record["targets"][name] for record in records], axis=0)
    return {
        "node_x": concatenate("node_x"),
        "edge_index": np.concatenate(edges, axis=1),
        "candidate_index": np.concatenate([
            record["candidate_index"] + node_ptr[index] for index, record in enumerate(records)
        ]),
        "candidate_mask": concatenate("candidate_mask"),
        "base_mask": concatenate("base_mask"),
        "action_x": concatenate("action_x"),
        "R": concatenate_target("R"),
        "R_valid": concatenate_target("R_valid"),
        "T": concatenate_target("T"),
        "T_valid": concatenate_target("T_valid"),
        "Y": concatenate_target("Y"),
        "Y_valid": concatenate_target("Y_valid"),
        "node_context": np.concatenate([
            np.full(len(record["node_x"]), index, dtype=np.int64) for index, record in enumerate(records)
        ]),
        "candidate_context": np.concatenate([
            np.full(len(record["candidate_index"]), index, dtype=np.int64) for index, record in enumerate(records)
        ]),
        "context_assembly": np.asarray(context_assembly, dtype=np.int64),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "candidate_ptr": np.asarray(candidate_ptr, dtype=np.int64),
        "assembly_keys": tuple(assemblies),
        "query_ids": tuple(record["query_id"] for record in records),
    }


def _binary_logit_loss(logits, targets):
    return np.logaddexp(0.0, logits) - targets * logits


def _joint_loss(r_logits, t_logits, targets):
    positive = np.logaddexp(0.0, -r_logits) + np.logaddexp(0.0, -t_logits)
    negative = (
        np.logaddexp(0.0, r_logits)
        + np.logaddexp(0.0, t_logits)
        - np.logaddexp(np.logaddexp(0.0, r_logits), t_logits)
    )
    return np.where(targets == 1.0, positive, negative)


def factorized_reference_loss(batch, r_logits, t_logits, component_assembly_counts):
    r_logits = _logits(r_logits, batch["R"].shape, "r_logits")
    t_logits = _logits(t_logits, batch["T"].shape, "t_logits")
    r_targets = batch["R"].astype(np.float64)
    t_targets = batch["T"].astype(np.float64)
    y_targets = batch["Y"].astype(np.float64)
    t_action_logits = t_logits[:, None]
    r_loss = _binary_logit_loss(r_logits, r_targets)
    t_loss = _binary_logit_loss(t_logits, t_targets)
    y_loss = _joint_loss(r_logits, t_action_logits, y_targets)
    log_y_probability = -np.logaddexp(0.0, -r_logits) - np.logaddexp(0.0, -t_action_logits)

    assembly_count = len(batch["assembly_keys"])
    candidate_assembly = batch["context_assembly"][batch["candidate_context"]]
    component_values = {name: np.full(assembly_count, np.nan) for name in ("rank", "R", "T", "Y")}
    denominators = {name: np.zeros(assembly_count, dtype=np.int64) for name in component_values}
    strict_y_valid = batch["R_valid"] & batch["T_valid"][:, None]

    for assembly in range(assembly_count):
        candidates = candidate_assembly == assembly
        r_mask = candidates[:, None] & batch["R_valid"]
        t_mask = candidates & batch["T_valid"]
        y_mask = candidates[:, None] & strict_y_valid
        for name, values, mask in (
            ("R", r_loss, r_mask), ("T", t_loss, t_mask), ("Y", y_loss, y_mask)
        ):
            denominators[name][assembly] = int(mask.sum())
            if mask.any():
                component_values[name][assembly] = float(values[mask].mean())

        ranking = []
        for context in np.flatnonzero(batch["context_assembly"] == assembly):
            context_candidates = batch["candidate_context"] == context
            if not batch["T_valid"][context_candidates].all() or not batch["R_valid"][context_candidates].all():
                continue
            context_targets = batch["Y"][context_candidates].reshape(-1)
            if not context_targets.any() or context_targets.all():
                continue
            scores = log_y_probability[context_candidates].reshape(-1)
            ranking.append(float(np.logaddexp.reduce(scores) - np.logaddexp.reduce(scores[context_targets == 1.0])))
        denominators["rank"][assembly] = len(ranking)
        if ranking:
            component_values["rank"][assembly] = float(np.mean(ranking))

    _require(
        isinstance(component_assembly_counts, dict)
        and set(component_assembly_counts) == {"total", "rank", "R", "T", "Y"},
        "component assembly counts differ",
    )
    total_assemblies = component_assembly_counts["total"]
    _require(type(total_assemblies) is int and total_assemblies >= assembly_count, "invalid total assembly count")
    per_assembly_total = np.zeros(assembly_count, dtype=np.float64)
    for name, weight in (("rank", 1.0), ("R", 1.0 / 3.0), ("T", 1.0 / 3.0), ("Y", 1.0 / 3.0)):
        defined_count = component_assembly_counts[name]
        _require(type(defined_count) is int and 0 < defined_count <= total_assemblies, "invalid {} assembly count".format(name))
        defined = np.isfinite(component_values[name])
        per_assembly_total[defined] += weight * total_assemblies / defined_count * component_values[name][defined]

    return {
        "loss": float(per_assembly_total.mean()),
        "components": {
            name: float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None
            for name, values in component_values.items()
        },
        "per_assembly": component_values,
        "denominators": denominators,
        "strict_y_valid_count": int(strict_y_valid.sum()),
    }
