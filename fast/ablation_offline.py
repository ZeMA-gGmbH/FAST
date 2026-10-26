"""Component-ablation frozen-checkpoint reference gate and experiments A, B, and E (validation only).

Every number uses the three frozen ranker Graph Transformer checkpoints on the frozen
2,947 validation contexts. Rankings, denominators, and the 10,000-replicate tier-stratified
paired bootstrap follow the frozen ranker validation code and the frozen component-ablation spec.
"""

import argparse
import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
import torch

from fast import graph_contract, ranker_models, ranker_train, ranker_validate
from fast import pointnet_models
from fast import ablation_closure as closure
from fast import ablation_common as common


SCHEMA_VERSION = "fast.ablation-offline.v1"
SPEC_PATH = common.REPOSITORY_ROOT / "configs" / "component_ablation.json"
RANKER_SPEC_PATH = common.REPOSITORY_ROOT / "configs" / "ranker_models.json"
FROZEN_PRODUCER_COMMIT = "4096b52b56b0e7715f895b9745763b7dbcbe0ce5"
METRICS = ("offline_context_success_at_1", "offline_context_success_at_3", "first_feasible_mrr", "mean_first_feasible_rank")
ROWS = ("all",) + common.TIERS
THRESHOLD = 0.5
SEEDS = (0, 1, 2)


def load_spec():
    spec = common.read_json(SPEC_PATH)
    common.require(spec.get("status") == "frozen" and spec.get("run_id") == common.RUN_ID, "component-ablation spec is not frozen")
    return spec, common.sha256_file(SPEC_PATH)


def load_context(data_root):
    """Bind the verified closure, frozen specs, graph audit, and validation inventory."""
    spec, spec_sha256 = load_spec()
    root = common.run_root(data_root)
    verification = common.read_json(root / "transfer-verification.json")
    common.require(verification.get("status") == "passed" and verification.get("test_rows_loaded") == 0, "closure verification did not pass")
    common.require(verification["transfer_manifest_sha256"] == common.sha256_file(root / "transfer-manifest.json"), "transfer manifest changed after verification")
    data = common.closure_root(data_root) / "fast-data"
    ranker_spec = graph_contract.load_spec(RANKER_SPEC_PATH)
    common.require(common.sha256_file(RANKER_SPEC_PATH) == spec["frozen_inputs"]["ranker_model_spec"]["sha256"], "ranker spec hash differs")
    audit_path = data / closure.GRAPH_AUDIT
    common.require(common.sha256_file(audit_path) == closure.FROZEN_SHA256[closure.GRAPH_AUDIT], "graph audit hash differs")
    audit = common.read_json(audit_path)
    validation_rows = [row for row in audit["inventory"] if row["dataset_split"] == "validation"]
    common.require(len(validation_rows) == spec["population"]["validation_assembly_count"], "validation inventory differs")
    common.require(all(row["source_split"] == "training_assembly" for row in audit["inventory"]), "graph inventory has a non-development source")
    return {
        "spec": spec, "spec_sha256": spec_sha256, "data_root": Path(data_root).resolve(), "run_root": root, "data": data,
        "audit": audit, "audit_sha256": common.sha256_file(audit_path),
        "prepared": {"output_root": data / closure.GRAPH_ROOT, "audit": audit, "spec": ranker_spec},
        "validation_rows": validation_rows,
        "verification_sha256": common.sha256_file(root / "transfer-verification.json"),
        "manifest_sha256": verification["transfer_manifest_sha256"],
    }


def _checkpoint_path(context, run, architecture, seed, name):
    return context["data"] / run / "training" / architecture / "seed-{}".format(seed) / "attempt-1" / "checkpoints" / name


def _load_frozen(context, architecture, seed):
    if architecture == "graph_transformer":
        run, name, model = closure.RANKER_RUN, closure.FAST_CHECKPOINTS[seed], ranker_models.build_model("graph_transformer")
    else:
        run, name, model = closure.POINTNET_RUN, closure.CEBULLA_CHECKPOINTS[seed], pointnet_models.CebullaGCN()
    path = _checkpoint_path(context, run, architecture, seed, name)
    relative = path.relative_to(context["data"]).as_posix()
    expected = closure.FROZEN_SHA256[relative]
    common.require(common.sha256_file(path) == expected, "frozen checkpoint hash differs: " + relative)
    sidecar = common.read_json(str(path) + ".json")
    common.require(sidecar.get("checkpoint_sha256") == expected, "frozen checkpoint sidecar differs: " + relative)
    training_audit = common.read_json(context["data"] / run / "training-audit.json")
    audited = [row for row in training_audit["inventory"] if row["architecture"] == architecture and row["seed"] == seed]
    common.require(len(audited) == 1 and audited[0]["best_checkpoint_sha256"] == expected, "frozen checkpoint is not the audited selection")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    provenance = payload.get("provenance", {})
    common.require(
        payload.get("schema_version") == ranker_train.CHECKPOINT_SCHEMA_VERSION
        and provenance.get("architecture") == architecture and provenance.get("seed") == seed
        and provenance.get("test_rows_loaded") == 0
        and provenance.get("normalization_sha256") == ranker_train._canonical_hash(context["audit"]["normalization"]),
        "frozen checkpoint provenance differs: " + relative,
    )
    if architecture == "graph_transformer":
        common.require(provenance.get("implementation") == {"commit": FROZEN_PRODUCER_COMMIT, "dirty": False}
                       and provenance.get("materialization_audit_sha256") == context["audit_sha256"],
                       "frozen FAST producer differs")
    environment = provenance.get("environment", {})
    common.require(environment.get("deterministic_algorithms") is True and environment.get("tf32_matmul") is False,
                   "frozen checkpoint was not deterministic")
    model.load_state_dict(payload["model_state"])
    return model.eval(), {"path": relative, "sha256": expected, "epoch": payload.get("epoch")}


def compatibility_environment():
    import torch_geometric
    common.require(not torch.cuda.is_available(), "compatibility analyses must not use CUDA with the frozen CUDA-12.1 binary")
    return OrderedDict([
        ("python", "{}.{}.{}".format(*sys.version_info[:3])), ("torch", torch.__version__),
        ("torch_geometric", torch_geometric.__version__), ("numpy", np.__version__), ("device", "cpu"),
        ("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES")), ("torch_threads", torch.get_num_threads()),
        ("omp_num_threads", os.environ.get("OMP_NUM_THREADS")),
        ("deterministic_algorithms", torch.are_deterministic_algorithms_enabled()),
        ("lock", "environment/cu121/uv.lock"), ("lock_sha256", common.sha256_file(common.REPOSITORY_ROOT / "environment/cu121/uv.lock")),
    ])


def validation_records(context):
    """Yield validation records in the frozen predict() order, with the zero-test-row guard."""
    for batch_rows in ranker_train._assembly_batches(context["validation_rows"], 0, 0, shuffle=False):
        for row in batch_rows:
            for record in ranker_train._load_assembly(context["prepared"], row):
                common.require(record["dataset_split"] in graph_contract.DEVELOPMENT_SPLITS, "test row loaded")
                yield record


def predict_frozen(context, model):
    predictions, seconds = ranker_train.predict(model, context["prepared"], context["validation_rows"], torch.device("cpu"))
    records = list(validation_records(context))
    common.require([row["query_id"] for row in predictions] == [record["query_id"] for record in records], "prediction order differs")
    for prediction, record in zip(predictions, records):
        prediction["part_ids"] = [record["part_ids"][index] for index in record["candidate_index"]]
        prediction["state_id"], prediction["pose_id"] = record["state_id"], record["pose_id"]
        prediction["dataset_split"] = record["dataset_split"]
    return predictions, seconds


# ---------------------------------------------------------------- ranking variants


def part_ranks(part_ids):
    """Rank candidates by part-ID string, the planner's tie order (action_planner.scored_actions)."""
    order = sorted(range(len(part_ids)), key=lambda index: str(part_ids[index]))
    ranks = np.empty(len(part_ids), dtype=np.int64)
    ranks[order] = np.arange(len(part_ids))
    return ranks


def _score_order(scores, ranks):
    scores = np.asarray(scores).reshape(len(ranks), 6)
    part = np.repeat(ranks, 6)
    direction = np.tile(np.arange(6), len(ranks))
    return np.lexsort((direction, part, -scores.reshape(-1)))


def _part_order(part_scores, ranks, direction_order):
    parts = np.lexsort((ranks, -np.asarray(part_scores)))
    return np.asarray([int(part) * 6 + int(direction) for part in parts for direction in direction_order], dtype=np.int64)


def prior_direction_order(rates):
    return sorted(range(6), key=lambda index: (-float(rates[common.DIRECTIONS[index]]), index))


def variant_orders(prediction, prior_order):
    r, t, y = prediction["R_score"], prediction["T_score"], prediction["Y_score"]
    ranks = part_ranks(prediction["part_ids"])
    return OrderedDict([
        ("A0", _score_order(y, ranks)),
        ("A1", _score_order(r, ranks)),
        ("A2", _score_order(np.repeat(t[:, None], 6, axis=1), ranks)),
        ("B1", _part_order(y.max(axis=1), ranks, range(6))),
        ("B2", _part_order(y.max(axis=1), ranks, prior_order)),
    ])


def context_values(targets, order):
    ranked = np.asarray(targets).reshape(-1)[order]
    first = int(np.flatnonzero(ranked == 1)[0])
    return {
        "offline_context_success_at_1": float(ranked[0] == 1), "offline_context_success_at_3": float(ranked[:3].any()),
        "first_feasible_mrr": 1.0 / (first + 1), "mean_first_feasible_rank": float(first + 1),
    }


def eligible(prediction):
    return bool(prediction["fully_observed"]) and bool(prediction["Y"].any())


def ranking_tables(per_context):
    """per_context: list of (assembly_id, tier, eligible, values-or-None) for one variant and seed."""
    tables, assemblies = OrderedDict(), {}
    for row in ROWS:
        selected = [item for item in per_context if row == "all" or item[1] == row]
        values = [item[3] for item in selected if item[2]]
        tables[row] = OrderedDict([
            ("assembly_count", len({item[0] for item in selected})),
            ("stored_context_count", len(selected)),
            ("ranking_eligible_context_count", len(values)),
        ] + [(metric, float(np.mean([value[metric] for value in values])) if values else None) for metric in METRICS])
    for assembly_id, _tier, is_eligible, values in per_context:
        entry = assemblies.setdefault(assembly_id, {metric: [0.0, 0] for metric in METRICS})
        if is_eligible:
            for metric in METRICS:
                entry[metric][0] += values[metric]
                entry[metric][1] += 1
    return tables, assemblies


# ---------------------------------------------------------------- bootstrap


def load_plan(context):
    path = context["data"] / closure.RANKER_RUN / "bootstrap" / "resample-plan.json"
    expected = context["spec"]["statistics"]["plan"]["sha256"]
    common.require(common.sha256_file(path) == expected, "bootstrap plan hash differs")
    plan = common.read_json(path)
    common.require(plan["replicates"] == 10000 and plan["seed"] == 20260916, "bootstrap plan configuration differs")
    order = plan["assembly_order"]
    expected_order = sorted(({"assembly_id": row["assembly_id"], "tier": row["tier"]} for row in context["validation_rows"]),
                            key=lambda row: (row["tier"], row["assembly_id"]))
    common.require(order == expected_order, "bootstrap assembly order differs from validation inventory")
    samples = np.asarray(plan["sample_indices"], dtype=np.int64)
    common.require(samples.shape == (10000, len(order)), "bootstrap sample shape differs")
    tiers = np.asarray([row["tier"] for row in order])
    blocks, offset = OrderedDict(), 0
    for tier in common.TIERS:
        size = int((tiers == tier).sum())
        block = np.arange(offset, offset + size)
        common.require(np.all(tiers[samples[:, block]] == tier), "bootstrap block is not tier-stratified: " + tier)
        blocks[tier] = block
        offset += size
    common.require(offset == len(order), "bootstrap blocks do not cover the plan")
    verification = OrderedDict([
        ("sha256", expected), ("replicates", int(samples.shape[0])), ("assemblies", len(order)),
        ("tier_draws_per_replicate", OrderedDict((tier, int(len(block))) for tier, block in blocks.items())),
        ("every_tier_block_draws_only_its_tier", True),
    ])
    return order, samples, blocks, verification


def replicate_ratios(assemblies, metric, order, samples, columns):
    numerator = np.asarray([assemblies.get(row["assembly_id"], {}).get(metric, [0.0, 0])[0] for row in order], dtype=np.float64)
    denominator = np.asarray([assemblies.get(row["assembly_id"], {}).get(metric, [0.0, 0])[1] for row in order], dtype=np.float64)
    sampled = samples[:, columns]
    num, den = numerator[sampled].sum(axis=1), denominator[sampled].sum(axis=1)
    return np.divide(num, den, out=np.full(len(samples), np.nan), where=den != 0)


def paired_difference(variant_assemblies, control_assemblies, metric, order, samples, columns):
    """Seed-averaged paired replicate difference; inputs are per-seed assembly tables."""
    def averaged(tables):
        stacked = np.stack([replicate_ratios(table, metric, order, samples, columns) for table in tables])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(stacked, axis=0)

    difference = averaged(variant_assemblies) - averaged(control_assemblies)
    defined = difference[np.isfinite(difference)]
    return OrderedDict([
        ("replicate_mean", float(defined.mean()) if len(defined) else None),
        ("percentile_2_5", float(np.percentile(defined, 2.5)) if len(defined) else None),
        ("percentile_97_5", float(np.percentile(defined, 97.5)) if len(defined) else None),
        ("undefined_replicates", int((~np.isfinite(difference)).sum())),
    ])


def _columns(blocks, row):
    return np.concatenate(list(blocks.values())) if row == "all" else blocks[row]


def compare_variants(results, control, variants, order, samples, blocks):
    """results[variant][seed] = (tables, assemblies). Return point and interval differences by row."""
    output = OrderedDict()
    for variant in variants:
        rows = OrderedDict()
        for row in ROWS:
            metrics = OrderedDict()
            for metric in METRICS:
                variant_points = [results[variant][seed][0][row][metric] for seed in SEEDS]
                control_points = [results[control][seed][0][row][metric] for seed in SEEDS]
                metrics[metric] = OrderedDict([
                    ("point_difference", float(np.mean(variant_points) - np.mean(control_points))),
                    ("seed_differences", OrderedDict((str(seed), variant_points[index] - control_points[index]) for index, seed in enumerate(SEEDS))),
                    ("bootstrap", paired_difference(
                        [results[variant][seed][1] for seed in SEEDS], [results[control][seed][1] for seed in SEEDS],
                        metric, order, samples, _columns(blocks, row))),
                ])
            rows[row] = metrics
        output["{}_minus_{}".format(variant, control)] = rows
    return output


def variant_summary(results, variant):
    rows = OrderedDict()
    for row in ROWS:
        seed_tables = [results[variant][seed][0][row] for seed in SEEDS]
        rows[row] = OrderedDict(
            [(name, seed_tables[0][name]) for name in ("assembly_count", "stored_context_count", "ranking_eligible_context_count")]
            + [(metric, OrderedDict([("seed_mean", float(np.mean([table[metric] for table in seed_tables]))),
                                     ("seeds", OrderedDict((str(seed), table[metric]) for seed, table in zip(SEEDS, seed_tables)))]))
               for metric in METRICS]
        )
    return rows


# ---------------------------------------------------------------- prediction quality


def threshold_counts(values, valid, probabilities):
    values, valid, probabilities = (np.asarray(item).reshape(-1) for item in (values, valid, probabilities))
    predicted = probabilities >= THRESHOLD
    positive = values == 1.0
    counts = OrderedDict([
        ("TP", int((valid & predicted & positive).sum())), ("TN", int((valid & ~predicted & ~positive).sum())),
        ("FP", int((valid & predicted & ~positive).sum())), ("FN", int((valid & ~predicted & positive).sum())),
        ("unresolved", int((~valid).sum())), ("probability_exactly_threshold", int((probabilities == THRESHOLD).sum())),
    ])
    counts["total"] = int(len(values))
    return counts


def quality_rates(counts):
    total = counts["total"]
    resolved = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    share = lambda value, base: 100.0 * value / base if base else None
    return OrderedDict([
        ("TP_percent", share(counts["TP"], total)), ("TN_percent", share(counts["TN"], total)),
        ("FP_percent", share(counts["FP"], total)), ("FN_percent", share(counts["FN"], total)),
        ("unresolved_percent", share(counts["unresolved"], total)),
        ("accuracy_percent", share(counts["TP"] + counts["TN"], resolved)),
        ("precision_percent", share(counts["TP"], counts["TP"] + counts["FP"])),
        ("recall_percent", share(counts["TP"], counts["TP"] + counts["FN"])),
    ])


def _target_arrays(prediction, target):
    if target == "T":
        return prediction["T"], prediction["T_valid"], prediction["T_score"]
    return prediction[target], prediction[target + "_valid"], prediction[target + "_score"]


def prediction_quality(predictions):
    output = OrderedDict()
    for target in ("R", "T", "Y"):
        rows = OrderedDict()
        for row in ROWS:
            selected = [item for item in predictions if row == "all" or item["tier"] == row]
            arrays = [_target_arrays(item, target) for item in selected]
            counts = threshold_counts(*(np.concatenate([np.asarray(array[index]).reshape(-1) for array in arrays]) for index in range(3)))
            rows[row] = OrderedDict([("assembly_count", len({item["assembly_id"] for item in selected})),
                                     ("counts", counts), ("rates", quality_rates(counts))])
        threshold_free = ranker_validate._metric_tables(ranker_validate._target_entries(predictions, target))
        for row in ROWS:
            metrics = threshold_free["overall"] if row == "all" else threshold_free["tiers"][row]
            rows[row]["average_precision"] = metrics["average_precision"]
            rows[row]["auroc"] = metrics["auroc"]
            common.require(metrics["null_count"] == rows[row]["counts"]["unresolved"], "unresolved count differs from frozen metric table")
        output[target] = rows
    return output


def _mean_quality(per_seed):
    output = OrderedDict()
    for target, rows in per_seed["0"].items():
        output[target] = OrderedDict()
        for row in rows:
            seeds = [per_seed[str(seed)][target][row] for seed in SEEDS]
            output[target][row] = OrderedDict(
                [("assembly_count", seeds[0]["assembly_count"]), ("label_total", seeds[0]["counts"]["total"]),
                 ("unresolved", seeds[0]["counts"]["unresolved"])]
                + [(name, float(np.mean([item["rates"][name] for item in seeds])) if all(item["rates"][name] is not None for item in seeds) else None)
                   for name in seeds[0]["rates"]]
                + [(name, float(np.mean([item[name] for item in seeds])) if all(item[name] is not None for item in seeds) else None)
                   for name in ("average_precision", "auroc")]
            )
    return output


# ---------------------------------------------------------------- reference gate


def _normalised(value):
    import json
    return json.loads(common.canonical_json(value))


def reference_reproduction(context, predictions_by_seed, checkpoints):
    gate = context["spec"]["reference_gate"]
    per_seed, identical = OrderedDict(), True
    for seed in SEEDS:
        predictions = predictions_by_seed[seed]
        ranking, _ = ranker_validate._ranking_tables(predictions)
        raw = {target: ranker_validate._metric_tables(ranker_validate._target_entries(predictions, target)) for target in ("R", "T", "Y")}
        frozen = common.read_json(context["data"] / closure.RANKER_RUN / "validation" / "graph_transformer" / "seed-{}.json".format(seed))
        common.require(frozen["checkpoint_sha256"] == checkpoints[seed]["sha256"], "frozen report checkpoint differs")
        seed_identical = _normalised(ranking) == frozen["ranking"] and all(
            _normalised(raw[target]) == frozen["targets"][target]["raw"] for target in raw)
        identical = identical and seed_identical
        per_seed[str(seed)] = OrderedDict([
            ("offline_context_success_at_1", ranking["overall"]["offline_context_success_at_1"]),
            ("first_feasible_mrr", ranking["overall"]["first_feasible_mrr"]),
            ("Y_average_precision", raw["Y"]["overall"]["average_precision"]),
            ("frozen_offline_context_success_at_1", frozen["ranking"]["overall"]["offline_context_success_at_1"]),
            ("frozen_first_feasible_mrr", frozen["ranking"]["overall"]["first_feasible_mrr"]),
            ("frozen_Y_average_precision", frozen["targets"]["Y"]["raw"]["overall"]["average_precision"]),
            ("ranking_denominators", ranking["overall"]["denominators"]),
            ("identical_to_frozen_report", seed_identical),
        ])
    means = OrderedDict((name, float(np.mean([per_seed[str(seed)][name] for seed in SEEDS])))
                        for name in gate["three_seed_means_rounded_5"])
    checks = OrderedDict((name, round(means[name], 5) == expected) for name, expected in gate["three_seed_means_rounded_5"].items())
    denominators = per_seed["0"]["ranking_denominators"]
    population = context["spec"]["population"]
    count_checks = OrderedDict([
        ("validation_context_count", denominators["total"] == population["validation_context_count"]),
        ("ranking_eligible_context_count", denominators["positive"] == population["ranking_eligible_context_count"]),
    ])
    return OrderedDict([
        ("status", "passed" if all(checks.values()) and all(count_checks.values()) else "failed"),
        ("rule", gate["rule"]), ("expected_rounded_5", gate["three_seed_means_rounded_5"]),
        ("three_seed_means", means), ("rounded_5_checks", checks), ("population_count_checks", count_checks),
        ("per_seed", per_seed), ("byte_identical_metric_tables", identical),
        ("reproduction_kind", "identical per-seed metric tables" if identical else "numerically reproduced at five decimals only"),
        ("checkpoints", OrderedDict((str(seed), checkpoints[seed]) for seed in SEEDS)),
    ])


# ---------------------------------------------------------------- entry point


def _provenance(context, environment):
    return OrderedDict([
        ("run_id", common.RUN_ID), ("spec_sha256", context["spec_sha256"]),
        ("implementation", common.implementation_identity(require_clean=True)),
        ("environment", environment), ("transfer_manifest_sha256", context["manifest_sha256"]),
        ("transfer_verification_sha256", context["verification_sha256"]), ("graph_audit_sha256", context["audit_sha256"]),
        ("created_at", common.utc_now()), ("post_hoc_exploratory", True), ("validation_only", True), ("test_rows_loaded", 0),
    ])


def run(data_root):
    ranker_train.configure_determinism(0)
    context = load_context(data_root)
    environment = compatibility_environment()
    root = context["run_root"]
    for name in ("reference-reproduction.json", "score-ablation.json", "granularity-ablation.json", "prediction-quality.json"):
        common.require(not (root / name).exists(), "offline output already exists: " + name)

    predictions_by_seed, checkpoints, inference_seconds = {}, {}, {}
    for seed in SEEDS:
        model, checkpoints[seed] = _load_frozen(context, "graph_transformer", seed)
        predictions_by_seed[seed], inference_seconds[str(seed)] = predict_frozen(context, model)
    test_rows = sum(1 for seed in SEEDS for row in predictions_by_seed[seed] if row.get("dataset_split") == "test")
    common.require(test_rows == 0, "test rows loaded")

    reference = reference_reproduction(context, predictions_by_seed, checkpoints)
    provenance = _provenance(context, environment)
    reference_report = OrderedDict([("schema_version", SCHEMA_VERSION), ("kind", "reference_reproduction")] + list(provenance.items())
                                   + list(reference.items()) + [("inference_seconds", inference_seconds)])
    reference_sha256 = common.write_json_once(root / "reference-reproduction.json", reference_report)
    common.require(reference["status"] == "passed", "frozen reference did not reproduce; A, B, and E are aborted")

    prior_path = context["data"] / closure.POINTNET_RUN / "planner" / "direction-prior.json"
    common.require(common.sha256_file(prior_path) == context["spec"]["experiments"]["B"]["direction_prior_sha256"], "direction prior hash differs")
    prior = common.read_json(prior_path)
    common.require(tuple(prior["directions"]) == common.DIRECTIONS == graph_contract.DIRECTION_IDS, "direction order differs")
    prior_order = prior_direction_order(prior["rates"])

    results = OrderedDict((variant, {}) for variant in ("A0", "A1", "A2", "B1", "B2"))
    for seed in SEEDS:
        per_variant = OrderedDict((variant, []) for variant in results)
        for prediction in predictions_by_seed[seed]:
            is_eligible = eligible(prediction)
            orders = variant_orders(prediction, prior_order)
            for variant, order in orders.items():
                values = context_values(prediction["Y"], order) if is_eligible else None
                per_variant[variant].append((prediction["assembly_id"], prediction["tier"], is_eligible, values))
        for variant, rows in per_variant.items():
            results[variant][seed] = ranking_tables(rows)
    reference_success = [reference["per_seed"][str(seed)]["offline_context_success_at_1"] for seed in SEEDS]
    a0_success = [results["A0"][seed][0]["all"]["offline_context_success_at_1"] for seed in SEEDS]
    tie_rule = OrderedDict([
        ("rule", context["spec"]["ranking_contract"]["sort"]),
        ("a0_contexts_where_part_id_tie_rule_changes_frozen_stored_order", OrderedDict(
            (str(seed), sum(int(not np.array_equal(variant_orders(row, prior_order)["A0"], np.argsort(-row["Y_score"].reshape(-1), kind="stable")))
                            for row in predictions_by_seed[seed])) for seed in SEEDS)),
        ("a0_success_at_1_equals_frozen_ranking", a0_success == reference_success),
    ])

    order, samples, blocks, plan_verification = load_plan(context)
    shared = list(provenance.items()) + [("reference_reproduction_sha256", reference_sha256), ("bootstrap_plan", plan_verification),
                                         ("checkpoints", reference["checkpoints"]), ("tie_rule_check", tie_rule)]
    score = OrderedDict([("schema_version", SCHEMA_VERSION), ("kind", "score_ablation"), ("experiment", context["spec"]["experiments"]["A"])]
                        + shared + [
        ("variants", OrderedDict((variant, variant_summary(results, variant)) for variant in ("A0", "A1", "A2"))),
        ("paired_differences", compare_variants(results, "A0", ("A1", "A2"), order, samples, blocks)),
    ])
    granularity = OrderedDict([("schema_version", SCHEMA_VERSION), ("kind", "granularity_ablation"), ("experiment", context["spec"]["experiments"]["B"])]
                              + shared + [
        ("direction_prior_order", [common.DIRECTIONS[index] for index in prior_order]),
        ("variants", OrderedDict((variant, variant_summary(results, variant)) for variant in ("A0", "B1", "B2"))),
        ("paired_differences", compare_variants(results, "A0", ("B1", "B2"), order, samples, blocks)),
    ])
    per_seed_quality = OrderedDict((str(seed), prediction_quality(predictions_by_seed[seed])) for seed in SEEDS)
    quality = OrderedDict([("schema_version", SCHEMA_VERSION), ("kind", "prediction_quality"), ("experiment", context["spec"]["experiments"]["E"])]
                          + shared + [
        ("threshold", "predicted positive iff uncalibrated probability >= 0.5"),
        ("null_label_name", "unresolved/censored"), ("seed_mean", _mean_quality(per_seed_quality)), ("per_seed", per_seed_quality),
    ])
    hashes = OrderedDict([("reference-reproduction.json", reference_sha256)])
    hashes["score-ablation.json"] = common.write_json_once(root / "score-ablation.json", score)
    hashes["granularity-ablation.json"] = common.write_json_once(root / "granularity-ablation.json", granularity)
    hashes["prediction-quality.json"] = common.write_json_once(root / "prediction-quality.json", quality)
    return OrderedDict([("status", "passed"), ("outputs", hashes), ("test_rows_loaded", 0)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)
    print(common.canonical_json(run(args.data_root)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
