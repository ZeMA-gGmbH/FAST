"""Component-ablation Experiment D analysis: same-runtime D0/D1/D2 comparison on the frozen validation contexts.

Run only after complete matched rounds. Each selected checkpoint is re-evaluated on cuda:0 in the
training runtime, ranked with the frozen component-ablation tie rule, and compared with the same-seed D0
control. Average-precision intervals use an exact sample-weighted form of sklearn's average
precision, with each replicate's assembly multiplicities from the frozen tier-stratified plan.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
import warnings
from collections import OrderedDict

import numpy as np
import torch

from fast import ranker_train, ranker_validate
from fast import ablation_closure as closure
from fast import ablation_common as common
from fast import ablation_masks as masks
from fast import ablation_offline as offline
from fast import ablation_runner as runner
from fast import ablation_train as trainer


SCHEMA_VERSION = "fast.feature-ablation.v1"
AP_TARGETS = ("R", "Y")
CHUNK = 250


def weighted_average_precision(targets, scores, weights):
    """sklearn.metrics.average_precision_score with sample_weight, vectorized over weight rows."""
    targets, scores = np.asarray(targets, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered_scores, ordered_targets = scores[order], targets[order]
    ends = np.r_[np.where(np.diff(ordered_scores))[0], len(ordered_scores) - 1]
    ordered_weights = weights[:, order]
    true_positive = np.cumsum(ordered_weights * ordered_targets, axis=1)[:, ends]
    false_positive = np.cumsum(ordered_weights * (1.0 - ordered_targets), axis=1)[:, ends]
    predicted = true_positive + false_positive
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
    total = true_positive[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = true_positive / total
    previous = np.concatenate([np.zeros((len(weights), 1)), recall[:, :-1]], axis=1)
    values = np.sum((recall - previous) * precision, axis=1)
    return np.where(total[:, 0] > 0, values, np.nan)


def _entries(predictions, target):
    values, scores, assemblies, tiers = [], [], [], []
    for row in predictions:
        valid = np.asarray(row[target + "_valid"]).reshape(-1)
        values.append(np.asarray(row[target]).reshape(-1)[valid])
        scores.append(np.asarray(row[target + "_score"]).reshape(-1)[valid])
        assemblies += [row["assembly_id"]] * int(valid.sum())
        tiers += [row["tier"]] * int(valid.sum())
    return np.concatenate(values), np.concatenate(scores), np.asarray(assemblies), np.asarray(tiers)


def ap_replicates(predictions, target, order, samples, columns, tier):
    values, scores, assemblies, tiers = _entries(predictions, target)
    if tier != "all":
        keep = tiers == tier
        values, scores, assemblies = values[keep], scores[keep], assemblies[keep]
    position = {row["assembly_id"]: index for index, row in enumerate(order)}
    entry_assembly = np.asarray([position[value] for value in assemblies])
    output = np.empty(len(samples))
    for start in range(0, len(samples), CHUNK):
        sampled = samples[start:start + CHUNK][:, columns]
        multiplicity = np.zeros((len(sampled), len(order)))
        for column in range(sampled.shape[1]):
            np.add.at(multiplicity, (np.arange(len(sampled)), sampled[:, column]), 1.0)
        output[start:start + CHUNK] = weighted_average_precision(values, scores, multiplicity[:, entry_assembly])
    return output, float(weighted_average_precision(values, scores, np.ones(len(values)))[0])


def _interval(difference):
    defined = difference[np.isfinite(difference)]
    return OrderedDict([("replicate_mean", float(defined.mean()) if len(defined) else None),
                        ("percentile_2_5", float(np.percentile(defined, 2.5)) if len(defined) else None),
                        ("percentile_97_5", float(np.percentile(defined, 97.5)) if len(defined) else None),
                        ("undefined_replicates", int((~np.isfinite(difference)).sum()))])


def completed_seeds(root, registry):
    seeds = []
    for seed in (0, 1, 2):
        rows = [row for row in registry["matrix"] if row["seed"] == seed]
        if all(runner.completed(root, row) for row in rows):
            seeds.append(seed)
    common.require(seeds and seeds == list(range(len(seeds))), "D analysis needs complete matched rounds starting at seed 0")
    return seeds


def evaluate(data, root, row, device):
    directory = runner.completed(root, row)
    result = common.read_json(directory / "result.json")
    provenance = result["provenance"]
    common.require(provenance["variant"] == row["variant"] and provenance["seed"] == row["seed"], "result identity differs")
    checkpoint = directory / result["best_checkpoint"]
    common.require(common.sha256_file(checkpoint) == result["best_checkpoint_sha256"], "selected checkpoint hash differs")
    spec = data.context["spec"]
    model = masks.build_model(spec, row["variant"]).to(device)
    trainer.load_checkpoint(checkpoint, model, None, provenance)
    indices, _ = masks.action_mask(spec, row["variant"])
    predictions, _ = trainer.predict(model, data, data.validation_rows, indices, device)
    selection = ranker_train._selection_metrics(predictions)
    recorded = {key: value for key, value in result["best"]["selection"].items() if key != "epoch"}
    return predictions, OrderedDict([
        ("result_sha256", common.sha256_file(directory / "result.json")), ("selected_checkpoint_sha256", result["best_checkpoint_sha256"]),
        ("selected_epoch", result["best"]["epoch"]), ("implementation_commit", provenance["implementation"]["commit"]),
        ("selection_metrics_reproduced", selection == recorded), ("parameter_count", provenance["parameter_count"]),
        ("input_width", provenance["input_width"]),
    ])


def run(data_root):
    ranker_train.configure_determinism(0)
    root = common.run_root(data_root)
    common.require(not (root / "feature-ablation.json").exists(), "feature-ablation output already exists")
    registry = runner.load_registry(root)
    seeds = completed_seeds(root, registry)
    data = trainer.TrainingData(data_root)
    context, spec = data.context, data.context["spec"]
    device = torch.device("cuda", 0)
    environment = trainer.training_environment()
    results, runs, ap_rows = OrderedDict(), OrderedDict(), OrderedDict()
    order, samples, blocks, plan_verification = offline.load_plan(context)
    for variant in masks.VARIANTS:
        results[variant], ap_rows[variant] = {}, {}
        for seed in seeds:
            row = next(item for item in registry["matrix"] if item["variant"] == variant and item["seed"] == seed)
            predictions, runs[row["run_name"]] = evaluate(data, root, row, device)
            per_context = []
            for prediction in predictions:
                is_eligible = offline.eligible(prediction)
                ranked = offline.variant_orders(prediction, list(range(6)))["A0"]
                per_context.append((prediction["assembly_id"], prediction["tier"], is_eligible,
                                    offline.context_values(prediction["Y"], ranked) if is_eligible else None))
            results[variant][seed] = offline.ranking_tables(per_context)
            tables = {target: ranker_validate._metric_tables(ranker_validate._target_entries(predictions, target)) for target in AP_TARGETS}
            ap_rows[variant][seed] = {}
            for tier in offline.ROWS:
                columns = offline._columns(blocks, tier)
                for target in AP_TARGETS:
                    replicates, point = ap_replicates(predictions, target, order, samples, columns, tier)
                    frozen_point = (tables[target]["overall"] if tier == "all" else tables[target]["tiers"][tier])["average_precision"]
                    common.require(frozen_point is None or abs(point - frozen_point) < 1e-12, "weighted AP differs from sklearn")
                    ap_rows[variant][seed][(tier, target)] = (point, replicates)
    summary = OrderedDict()
    for variant in masks.VARIANTS:
        summary[variant] = OrderedDict()
        for tier in offline.ROWS:
            first = results[variant][seeds[0]][0][tier]
            entry = OrderedDict((name, first[name]) for name in ("assembly_count", "stored_context_count", "ranking_eligible_context_count"))
            for metric in offline.METRICS:
                values = OrderedDict((str(seed), results[variant][seed][0][tier][metric]) for seed in seeds)
                entry[metric] = OrderedDict([("seed_mean", float(np.mean(list(values.values())))), ("seeds", values)])
            for target in AP_TARGETS:
                values = OrderedDict((str(seed), ap_rows[variant][seed][(tier, target)][0]) for seed in seeds)
                entry[target + "_average_precision"] = OrderedDict([("seed_mean", float(np.mean(list(values.values())))), ("seeds", values)])
            summary[variant][tier] = entry
    differences = OrderedDict()
    for variant in ("D1", "D2"):
        rows = OrderedDict()
        for tier in offline.ROWS:
            columns = offline._columns(blocks, tier)
            metrics = OrderedDict()
            for metric in offline.METRICS:
                metrics[metric] = OrderedDict([
                    ("point_difference", summary[variant][tier][metric]["seed_mean"] - summary["D0"][tier][metric]["seed_mean"]),
                    ("seed_differences", OrderedDict((str(seed), results[variant][seed][0][tier][metric] - results["D0"][seed][0][tier][metric]) for seed in seeds)),
                    ("bootstrap", offline.paired_difference([results[variant][seed][1] for seed in seeds], [results["D0"][seed][1] for seed in seeds],
                                                            metric, order, samples, columns)),
                ])
            for target in AP_TARGETS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    variant_replicates = np.nanmean(np.stack([ap_rows[variant][seed][(tier, target)][1] for seed in seeds]), axis=0)
                    control_replicates = np.nanmean(np.stack([ap_rows["D0"][seed][(tier, target)][1] for seed in seeds]), axis=0)
                metrics[target + "_average_precision"] = OrderedDict([
                    ("point_difference", summary[variant][tier][target + "_average_precision"]["seed_mean"] - summary["D0"][tier][target + "_average_precision"]["seed_mean"]),
                    ("seed_differences", OrderedDict((str(seed), ap_rows[variant][seed][(tier, target)][0] - ap_rows["D0"][seed][(tier, target)][0]) for seed in seeds)),
                    ("bootstrap", _interval(variant_replicates - control_replicates)),
                ])
            rows[tier] = metrics
        differences[variant + "_minus_D0"] = rows
    reference = common.read_json(root / "reference-reproduction.json")
    report = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("kind", "feature_ablation"), ("experiment", spec["experiments"]["D"]),
        ("run_id", common.RUN_ID), ("spec_sha256", context["spec_sha256"]), ("implementation", common.implementation_identity(require_clean=True)),
        ("environment", environment), ("created_at", common.utc_now()), ("post_hoc_exploratory", True), ("validation_only", True),
        ("seeds_completed", seeds), ("single_seed_exploratory", seeds == [0]),
        ("bootstrap_plan", plan_verification), ("runs", runs), ("variants", summary), ("paired_differences", differences),
        ("frozen_fast_context_only", OrderedDict([
            ("note", "frozen A100 checkpoints; compatibility reference and historical seed spread only, not the causal control"),
            ("per_seed", {seed: {key: values[key] for key in ("offline_context_success_at_1", "first_feasible_mrr", "Y_average_precision")}
                          for seed, values in reference["per_seed"].items()}),
        ])),
        ("test_rows_loaded", 0),
    ])
    digest = common.write_json_once(root / "feature-ablation.json", report)
    return {"status": "passed", "feature-ablation.json": digest, "seeds_completed": seeds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)
    print(common.canonical_json(run(args.data_root)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
