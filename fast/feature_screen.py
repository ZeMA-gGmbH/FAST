"""Screen scalar feature groups without reading the frozen test split."""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fast import cad_features as feature_builder
from plan_sequence import production_sim_labels as production


REPORT_SCHEMA_VERSION = "fast.screen-report.v1"
AUDIT_SCHEMA_VERSION = "fast.screen-audit.v1"
SPLITS = {"train": 0, "validation": 1}
TIERS = {"small": 0, "medium": 1, "large": 2}
DIRECTION_IDS = [item[0] for item in feature_builder.dataset_index.DIRECTIONS]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _read_single(path):
    records = list(_read_jsonl(path))
    _require(len(records) == 1, "expected one JSON record: {}".format(path))
    return records[0]


def _write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(production._canonical_json(value) + "\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _screen_jobs(prepared):
    path = feature_builder._resolve_under(
        prepared["manifest_path"].parent.parent,
        prepared["config"]["screen_jobs"],
        "screen_jobs",
    )
    jobs = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            _require(len(fields) == 2, "invalid screen job line {}".format(line_number))
            jobs.append(tuple(fields))
    _require(len(jobs) == len(set(jobs)), "duplicate screen jobs")
    return path, jobs


def _feature_set_columns(feature_set, candidate_names, action_names):
    groups = set(feature_builder.FEATURE_GROUPS)
    if feature_set == "full":
        selected = groups
    elif feature_set.startswith("only:"):
        selected = {feature_set.split(":", 1)[1]}
    elif feature_set.startswith("without:"):
        selected = groups - {feature_set.split(":", 1)[1]}
    else:
        raise ValueError("invalid feature set: {}".format(feature_set))
    _require(selected <= groups, "unknown feature group")
    candidate_columns = [
        index
        for index, name in enumerate(candidate_names)
        if name.split("__", 1)[0] in selected
    ]
    action_columns = [
        index
        for index, name in enumerate(action_names)
        if name.split("__", 1)[0] in selected
    ]
    return candidate_columns, action_columns, sorted(selected)


def _counts_for_splits(build, field):
    return sum(
        scope[field]
        for key, scope in build["per_split_tier"].items()
        if key.split("/", 1)[0] in SPLITS
    )


def _load_data(prepared):
    build_path = prepared["output_root"] / "manifest.json"
    _require(build_path.is_file(), "feature audit is missing; submit it before screening")
    build = _read_single(build_path)
    _require(
        build.get("schema_version") == feature_builder.BUILD_SCHEMA_VERSION
        and build.get("status") == "passed"
        and build.get("feature_manifest_sha256") == prepared["manifest_sha256"]
        and build.get("extractor_sha256") == prepared["extractor_sha256"],
        "feature audit provenance differs",
    )
    candidate_count = _counts_for_splits(build, "candidate_count")
    action_count = _counts_for_splits(build, "action_count")
    candidate_names = build["candidate_features"]
    action_names = build["action_features"]
    candidate_x = np.empty((candidate_count, len(candidate_names)), dtype=np.float32)
    candidate_t = np.empty(candidate_count, dtype=np.int8)
    candidate_t_valid = np.empty(candidate_count, dtype=bool)
    candidate_split = np.empty(candidate_count, dtype=np.int8)
    candidate_tier = np.empty(candidate_count, dtype=np.int8)
    candidate_assembly = np.empty(candidate_count, dtype=np.int32)
    candidate_query = np.empty(candidate_count, dtype=np.int32)
    action_x = np.empty((action_count, len(action_names)), dtype=np.float32)
    action_candidate = np.empty(action_count, dtype=np.int32)
    action_direction = np.empty(action_count, dtype=np.int8)
    action_r = np.empty(action_count, dtype=np.int8)
    action_r_valid = np.empty(action_count, dtype=bool)
    action_y = np.empty(action_count, dtype=np.int8)
    action_y_valid = np.empty(action_count, dtype=bool)

    assemblies = {}
    queries = {}
    candidate_index = 0
    action_index = 0
    for shard in build["shards"]:
        if shard["dataset_split"] not in SPLITS:
            continue
        path = prepared["output_root"] / shard["path"]
        _require(production._sha256_file(path) == shard["sha256"], "feature shard hash differs")
        records = _read_jsonl(path)
        header = next(records)
        _require(header["record_type"] == "header", "invalid feature shard header")
        for record in records:
            if record["record_type"] == "summary":
                break
            split_id = SPLITS[record["dataset_split"]]
            tier_id = TIERS[record["tier"]]
            assembly_key = (record["source_split"], record["assembly_id"])
            query_key = record["query_id"]
            assembly_id = assemblies.setdefault(assembly_key, len(assemblies))
            query_id = queries.setdefault(query_key, len(queries))
            candidate_x[candidate_index] = [record["features"][name] for name in candidate_names]
            target = record["targets"]["T"]
            candidate_t[candidate_index] = int(target["value"] or False)
            candidate_t_valid[candidate_index] = target["valid"]
            candidate_split[candidate_index] = split_id
            candidate_tier[candidate_index] = tier_id
            candidate_assembly[candidate_index] = assembly_id
            candidate_query[candidate_index] = query_id
            for direction_index, action in enumerate(record["actions"]):
                _require(
                    action["direction_id"] == DIRECTION_IDS[direction_index],
                    "action direction order differs",
                )
                action_x[action_index] = [action["features"][name] for name in action_names]
                action_candidate[action_index] = candidate_index
                action_direction[action_index] = direction_index
                r_target = action["targets"]["R"]
                y_target = action["targets"]["Y"]
                action_r[action_index] = int(r_target["value"] or False)
                action_r_valid[action_index] = r_target["valid"]
                action_y[action_index] = int(y_target["value"] or False)
                action_y_valid[action_index] = y_target["valid"]
                action_index += 1
            candidate_index += 1
    _require(candidate_index == candidate_count, "loaded candidate count differs")
    _require(action_index == action_count, "loaded action count differs")
    return {
        "feature_build": build,
        "feature_build_sha256": production._sha256_file(build_path),
        "candidate_names": candidate_names,
        "action_names": action_names,
        "candidate_x": candidate_x,
        "candidate_t": candidate_t,
        "candidate_t_valid": candidate_t_valid,
        "candidate_split": candidate_split,
        "candidate_tier": candidate_tier,
        "candidate_assembly": candidate_assembly,
        "candidate_query": candidate_query,
        "action_x": action_x,
        "action_candidate": action_candidate,
        "action_direction": action_direction,
        "action_r": action_r,
        "action_r_valid": action_r_valid,
        "action_y": action_y,
        "action_y_valid": action_y_valid,
    }


def _assembly_weights(assembly_ids):
    counts = np.bincount(assembly_ids)
    weights = 1.0 / counts[assembly_ids]
    return weights * (len(weights) / weights.sum())


class _ConstantModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, values):
        positive = np.full(len(values), self.probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _fit_model(name, values, targets, weights):
    if values.shape[1] == 0 or len(np.unique(targets)) == 1:
        return _ConstantModel(np.average(targets, weights=weights))
    if name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, solver="lbfgs", random_state=0),
        )
    elif name == "histgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=0,
        )
    else:
        raise ValueError("unsupported fitted model: {}".format(name))
    model.fit(values, targets, **({"sample_weight": weights} if name == "histgb" else {"logisticregression__sample_weight": weights}))
    return model


def _matrix(data, action_mask, candidate_columns, action_columns):
    candidate_indices = data["action_candidate"][action_mask]
    blocks = []
    if candidate_columns:
        blocks.append(data["candidate_x"][candidate_indices][:, candidate_columns])
    if action_columns:
        blocks.append(data["action_x"][action_mask][:, action_columns])
    return np.column_stack(blocks) if blocks else np.empty((len(candidate_indices), 0))


def _candidate_matrix(data, mask, columns):
    return data["candidate_x"][mask][:, columns] if columns else np.empty((mask.sum(), 0))


def _probability(model, values):
    return model.predict_proba(values)[:, 1]


def _equal_mass_ece(targets, probabilities, bin_count=15):
    if not len(targets):
        return None
    ordered = np.argsort(probabilities, kind="stable")
    bins = np.array_split(ordered, min(bin_count, len(ordered)))
    return float(
        sum(
            len(indices)
            * abs(float(np.mean(probabilities[indices])) - float(np.mean(targets[indices])))
            for indices in bins
        )
        / len(targets)
    )


def _binary_metrics(targets, probabilities):
    if not len(targets):
        return None
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    result = {
        "count": int(len(targets)),
        "positive_fraction": float(np.mean(targets)),
        "average_precision": float(average_precision_score(targets, probabilities)),
        "brier": float(brier_score_loss(targets, probabilities)),
        "negative_log_likelihood": float(log_loss(targets, probabilities, labels=[0, 1])),
        "expected_calibration_error_15_equal_mass": _equal_mass_ece(
            targets, probabilities, 15
        ),
    }
    result["auroc"] = (
        float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) == 2 else None
    )
    return result


def _ranking_metrics(data, probabilities, tier_id=None):
    candidate_indices = data["action_candidate"]
    mask = data["candidate_split"][candidate_indices] == SPLITS["validation"]
    if tier_id is not None:
        mask &= data["candidate_tier"][candidate_indices] == tier_id
    action_indices = np.flatnonzero(mask)
    groups = {}
    for action_index in action_indices:
        query_id = data["candidate_query"][candidate_indices[action_index]]
        groups.setdefault(int(query_id), []).append(int(action_index))
    reciprocal_ranks = []
    success_at_1 = []
    success_at_3 = []
    recall_at_1 = []
    recall_at_3 = []
    all_negative = 0
    censored = 0
    for indices in groups.values():
        if not data["action_y_valid"][indices].all():
            censored += 1
            continue
        target = data["action_y"][indices]
        if not target.any():
            all_negative += 1
            continue
        ordered = np.asarray(indices)[np.argsort(-probabilities[indices], kind="stable")]
        ordered_target = data["action_y"][ordered]
        first = int(np.flatnonzero(ordered_target)[0]) + 1
        reciprocal_ranks.append(1.0 / first)
        success_at_1.append(float(first <= 1))
        success_at_3.append(float(first <= 3))
        positive_count = int(target.sum())
        recall_at_1.append(float(ordered_target[:1].sum() / positive_count))
        recall_at_3.append(float(ordered_target[:3].sum() / positive_count))
    return {
        "context_count": len(groups),
        "positive_context_count": len(reciprocal_ranks),
        "all_negative_context_count": all_negative,
        "censored_context_count": censored,
        "success_at_1": float(np.mean(success_at_1)) if success_at_1 else None,
        "success_at_3": float(np.mean(success_at_3)) if success_at_3 else None,
        "recall_at_1": float(np.mean(recall_at_1)) if recall_at_1 else None,
        "recall_at_3": float(np.mean(recall_at_3)) if recall_at_3 else None,
        "first_feasible_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
    }


def _evaluate(data, r_probability, t_probability, ranking_probability=None):
    candidate_indices = data["action_candidate"]
    y_probability = r_probability * t_probability[candidate_indices]
    if ranking_probability is None:
        ranking_probability = y_probability
    validation_candidates = data["candidate_split"] == SPLITS["validation"]
    validation_actions = data["candidate_split"][candidate_indices] == SPLITS["validation"]

    def metrics_for_tier(tier_id):
        candidate_tier = np.ones(len(validation_candidates), dtype=bool)
        action_tier = np.ones(len(validation_actions), dtype=bool)
        if tier_id is not None:
            candidate_tier = data["candidate_tier"] == tier_id
            action_tier = data["candidate_tier"][candidate_indices] == tier_id
        r_mask = validation_actions & action_tier & data["action_r_valid"]
        t_mask = validation_candidates & candidate_tier & data["candidate_t_valid"]
        y_mask = validation_actions & action_tier & data["action_y_valid"]
        return {
            "R": _binary_metrics(data["action_r"][r_mask], r_probability[r_mask]),
            "T": _binary_metrics(data["candidate_t"][t_mask], t_probability[t_mask]),
            "Y": _binary_metrics(data["action_y"][y_mask], y_probability[y_mask]),
            "ranking": _ranking_metrics(data, ranking_probability, tier_id),
        }

    return {
        "overall": metrics_for_tier(None),
        "per_tier": {name: metrics_for_tier(tier_id) for name, tier_id in TIERS.items()},
    }


def _fitted_probabilities(data, model_name, candidate_columns, action_columns):
    candidate_indices = data["action_candidate"]
    train_actions = (
        (data["candidate_split"][candidate_indices] == SPLITS["train"])
        & data["action_r_valid"]
    )
    train_candidates = (
        (data["candidate_split"] == SPLITS["train"]) & data["candidate_t_valid"]
    )
    r_model = _fit_model(
        model_name,
        _matrix(data, train_actions, candidate_columns, action_columns),
        data["action_r"][train_actions],
        _assembly_weights(data["candidate_assembly"][candidate_indices[train_actions]]),
    )
    t_model = _fit_model(
        model_name,
        _candidate_matrix(data, train_candidates, candidate_columns),
        data["candidate_t"][train_candidates],
        _assembly_weights(data["candidate_assembly"][train_candidates]),
    )
    all_actions = np.ones(len(data["action_x"]), dtype=bool)
    all_candidates = np.ones(len(data["candidate_x"]), dtype=bool)
    return (
        _probability(r_model, _matrix(data, all_actions, candidate_columns, action_columns)),
        _probability(t_model, _candidate_matrix(data, all_candidates, candidate_columns)),
    )


def _direction_prior(data):
    candidate_indices = data["action_candidate"]
    train_actions = (
        (data["candidate_split"][candidate_indices] == SPLITS["train"])
        & data["action_r_valid"]
    )
    return np.array(
        [
            np.mean(data["action_r"][train_actions & (data["action_direction"] == index)])
            for index in range(6)
        ]
    )


def _prior_probabilities(data):
    direction_probability = _direction_prior(data)
    train_t = (data["candidate_split"] == SPLITS["train"]) & data["candidate_t_valid"]
    t_probability = np.full(len(data["candidate_x"]), np.mean(data["candidate_t"][train_t]))
    return direction_probability[data["action_direction"]], t_probability


def _heur_out_probability(data):
    outside_name = "pose_support__candidate_outside_distance"
    outside = data["candidate_x"][:, data["candidate_names"].index(outside_name)]
    direction_tie_break = (6 - data["action_direction"]) * 1e-7
    return outside[data["action_candidate"]] + direction_tie_break


def _heur_out_direction_prior_probability(data):
    outside_name = "pose_support__candidate_outside_distance"
    outside = data["candidate_x"][:, data["candidate_names"].index(outside_name)]
    part_priority = np.empty(len(outside), dtype=np.float64)
    query_order = np.argsort(data["candidate_query"], kind="stable")
    boundaries = np.flatnonzero(
        np.diff(data["candidate_query"][query_order])
    ) + 1
    for candidates in np.split(query_order, boundaries):
        ordered = candidates[np.argsort(-outside[candidates], kind="stable")]
        part_priority[ordered] = np.arange(len(ordered), 0, -1)
    direction_probability = _direction_prior(data)[data["action_direction"]]
    return part_priority[data["action_candidate"]] + 0.5 * direction_probability


def _report_path(prepared, model_name, feature_set):
    safe_feature_set = feature_set.replace(":", "-")
    return prepared["output_root"] / "screening" / "{}__{}.json".format(
        model_name, safe_feature_set
    )


def screen(manifest_path, source_root, data_root, model_name, feature_set):
    prepared = feature_builder._read_config(manifest_path, source_root, data_root)
    _, jobs = _screen_jobs(prepared)
    _require((model_name, feature_set) in jobs, "screen job is not predeclared")
    data = _load_data(prepared)
    candidate_columns, action_columns, selected_groups = _feature_set_columns(
        feature_set, data["candidate_names"], data["action_names"]
    )
    if model_name in ("logistic", "histgb"):
        r_probability, t_probability = _fitted_probabilities(
            data, model_name, candidate_columns, action_columns
        )
        ranking_probability = None
        predictive = True
    elif model_name == "direction_prior":
        r_probability, t_probability = _prior_probabilities(data)
        ranking_probability = None
        predictive = True
    elif model_name == "heur_out":
        r_probability, t_probability = _prior_probabilities(data)
        ranking_probability = _heur_out_probability(data)
        predictive = False
    elif model_name == "heur_out_direction_prior":
        r_probability, t_probability = _prior_probabilities(data)
        ranking_probability = _heur_out_direction_prior_probability(data)
        predictive = False
    else:
        raise ValueError("unsupported model")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "model": model_name,
        "feature_set": feature_set,
        "selected_groups": selected_groups,
        "candidate_feature_count": len(candidate_columns),
        "action_feature_count": len(action_columns),
        "predictive_metrics_apply_to_model": predictive,
        "feature_build_sha256": data["feature_build_sha256"],
        "screener_sha256": production._sha256_file(Path(__file__)),
        "sklearn_version": sklearn.__version__,
        "test_rows_loaded": 0,
        "validation": _evaluate(data, r_probability, t_probability, ranking_probability),
    }
    _write_atomic(_report_path(prepared, model_name, feature_set), report)
    return report


def audit(manifest_path, source_root, data_root):
    prepared = feature_builder._read_config(manifest_path, source_root, data_root)
    _, jobs = _screen_jobs(prepared)
    feature_build_path = prepared["output_root"] / "manifest.json"
    feature_build_sha256 = production._sha256_file(feature_build_path)
    reports = []
    for model_name, feature_set in jobs:
        path = _report_path(prepared, model_name, feature_set)
        _require(path.is_file(), "screen report is missing: {}".format(path))
        report = _read_single(path)
        _require(
            report.get("schema_version") == REPORT_SCHEMA_VERSION
            and report.get("status") == "passed"
            and report.get("model") == model_name
            and report.get("feature_set") == feature_set
            and report.get("feature_build_sha256") == feature_build_sha256
            and report.get("screener_sha256") == production._sha256_file(Path(__file__))
            and report.get("test_rows_loaded") == 0,
            "screen report provenance differs: {}".format(path),
        )
        reports.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "path": path.relative_to(prepared["output_root"]).as_posix(),
                "sha256": production._sha256_file(path),
                "validation": report["validation"],
            }
        )
    result = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "feature_build_sha256": feature_build_sha256,
        "screener_sha256": production._sha256_file(Path(__file__)),
        "report_count": len(reports),
        "reports": reports,
    }
    _write_atomic(prepared["output_root"] / "screening" / "audit.json", result)
    return result


def _parser():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(root / "configs" / "features.json")
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--screen", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--model",
        choices=(
            "direction_prior",
            "heur_out",
            "heur_out_direction_prior",
            "logistic",
            "histgb",
        ),
    )
    parser.add_argument("--feature-set")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.audit:
        result = audit(args.manifest, args.source_root, args.data_root)
    else:
        _require(args.model is not None and args.feature_set is not None, "model and feature set are required")
        result = screen(
            args.manifest, args.source_root, args.data_root, args.model, args.feature_set
        )
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
