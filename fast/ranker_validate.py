"""Run and audit ranker validation, calibration, and bootstrap reports."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score

from fast import graph_contract, graph_materialize, ranker_models, ranker_train
from plan_sequence import production_sim_labels as production


REPORT_SCHEMA_VERSION = "fast.ranker-validation-report.v1"
CALIBRATION_SCHEMA_VERSION = "fast.ranker-calibration.v1"
BOOTSTRAP_SCHEMA_VERSION = "fast.ranker-bootstrap.v1"
AUDIT_SCHEMA_VERSION = "fast.ranker-validation-audit.v1"


def _write_json(path, value):
    graph_materialize._write_json(path, value)


def _read_json(path):
    return graph_materialize._read_json(path)


def _logit_y(prediction):
    if prediction["Y_logit"] is not None:
        return prediction["Y_logit"]
    log_probability = -np.logaddexp(0.0, -prediction["R_logit"]) - np.logaddexp(
        0.0, -prediction["T_logit"][:, None]
    )
    return log_probability - np.log(-np.expm1(log_probability))


def _target_entries(predictions, target):
    entries = []
    for prediction in predictions:
        if target == "R" and prediction["R_logit"] is None:
            continue
        if target == "T" and prediction["T_logit"] is None:
            continue
        logits = prediction[target + "_logit"] if target != "Y" else _logit_y(prediction)
        values, valid = prediction[target], prediction[target + "_valid"]
        if target == "T":
            directions = np.full(values.shape, -1, dtype=np.int64)
        else:
            directions = np.broadcast_to(np.arange(6), values.shape)
        for value, is_valid, logit, direction in zip(
            values.reshape(-1), valid.reshape(-1), logits.reshape(-1), directions.reshape(-1)
        ):
            entries.append({
                "assembly_id": prediction["assembly_id"],
                "tier": prediction["tier"],
                "direction": int(direction),
                "target": float(value),
                "valid": bool(is_valid),
                "logit": float(logit),
            })
    return entries


def _ece(targets, probabilities, bins=15):
    if not len(targets):
        return None
    order = np.argsort(probabilities, kind="stable")
    result = 0.0
    for indices in np.array_split(order, min(bins, len(order))):
        result += len(indices) / len(order) * abs(float(targets[indices].mean() - probabilities[indices].mean()))
    return result


def _predictive_metrics(entries, temperature=1.0):
    valid_entries = [row for row in entries if row["valid"]]
    targets = np.asarray([row["target"] for row in valid_entries], dtype=np.float64)
    logits = np.asarray([row["logit"] for row in valid_entries], dtype=np.float64) / temperature
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    both_classes = len(np.unique(targets)) == 2
    return {
        "valid_count": len(valid_entries),
        "null_count": len(entries) - len(valid_entries),
        "positive_count": int(targets.sum()),
        "positive_fraction": float(targets.mean()) if len(targets) else None,
        "assembly_count": len({row["assembly_id"] for row in valid_entries}),
        "average_precision": float(average_precision_score(targets, probabilities)) if targets.any() else None,
        "auroc": float(roc_auc_score(targets, probabilities)) if both_classes else None,
        "brier": float(np.square(probabilities - targets).mean()) if len(targets) else None,
        "negative_log_likelihood": float(np.logaddexp(0.0, logits).mean() - (targets * logits).mean()) if len(targets) else None,
        "ece_15_equal_mass": _ece(targets, probabilities),
    }


def _metric_tables(entries, temperatures=None):
    temperature_for = (lambda row: temperatures[row["assembly_id"]]) if temperatures else (lambda row: 1.0)

    def metrics(rows):
        if not temperatures:
            return _predictive_metrics(rows)
        adjusted = [{**row, "logit": row["logit"] / temperature_for(row)} for row in rows]
        return _predictive_metrics(adjusted)

    result = {
        "overall": metrics(entries),
        "tiers": {tier: metrics([row for row in entries if row["tier"] == tier]) for tier in ("small", "medium", "large")},
    }
    if entries and entries[0]["direction"] >= 0:
        result["directions"] = {
            graph_contract.DIRECTION_IDS[index]: metrics([row for row in entries if row["direction"] == index])
            for index in range(6)
        }
    return result


def _temperature(entries):
    valid = [row for row in entries if row["valid"]]
    graph_contract._require(valid, "temperature target has no valid labels")
    targets = np.asarray([row["target"] for row in valid])
    logits = np.asarray([row["logit"] for row in valid])

    def objective(log_temperature):
        scaled = logits / math.exp(log_temperature)
        return float(np.logaddexp(0.0, scaled).mean() - (targets * scaled).mean())

    fit = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded", options={"xatol": 1e-10})
    graph_contract._require(fit.success and math.isfinite(fit.x), "temperature fit failed")
    return math.exp(float(fit.x))


def _cross_fit(entries, fold_assignments):
    fold_temperatures = {}
    assembly_temperatures = {}
    for fold in range(5):
        training = [
            row for row in entries
            if row["valid"] and fold_assignments["training_assembly:" + row["assembly_id"]] != fold
        ]
        temperature = _temperature(training)
        fold_temperatures[str(fold)] = temperature
        for assembly_id, assigned in fold_assignments.items():
            if assigned == fold:
                assembly_temperatures[assembly_id.split(":", 1)[1]] = temperature
    graph_contract._require(set(assembly_temperatures) == {row["assembly_id"] for row in entries}, "cross-fit fold coverage differs")
    return fold_temperatures, assembly_temperatures, _temperature(entries)


def _ranking_metrics(predictions, direct_y=False):
    categories = {name: 0 for name in ("total", "positive", "all_negative", "all_positive", "censored")}
    success1, success3, reciprocal, recall1, recall3 = [], [], [], [], []
    by_assembly = {}
    for row in predictions:
        categories["total"] += 1
        observed = bool(row["Y_valid"].all()) if direct_y else row["fully_observed"]
        targets = row["Y"].reshape(-1)
        if not observed:
            categories["censored"] += 1
            continue
        if not targets.any():
            categories["all_negative"] += 1
            continue
        categories["positive"] += 1
        categories["all_positive"] += int(targets.all())
        order = np.argsort(-row["Y_score"].reshape(-1), kind="stable")
        ranked = targets[order]
        values = (
            float(ranked[0] == 1),
            float(ranked[:3].any()),
            1.0 / (int(np.flatnonzero(ranked == 1)[0]) + 1),
            float(ranked[:1].sum() / targets.sum()),
            float(ranked[:3].sum() / targets.sum()),
        )
        for output, value in zip((success1, success3, reciprocal, recall1, recall3), values):
            output.append(value)
        assembly = by_assembly.setdefault(row["assembly_id"], {"numerator": 0, "denominator": 0})
        assembly["numerator"] += values[0]
        assembly["denominator"] += 1
    return {
        "offline_context_success_at_1": float(np.mean(success1)) if success1 else None,
        "offline_context_success_at_3": float(np.mean(success3)) if success3 else None,
        "first_feasible_mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "recall_at_1": float(np.mean(recall1)) if recall1 else None,
        "recall_at_3": float(np.mean(recall3)) if recall3 else None,
        "denominators": categories,
    }, by_assembly


def _ranking_tables(predictions, direct_y=False):
    overall, by_assembly = _ranking_metrics(predictions, direct_y)
    return {
        "overall": overall,
        "tiers": {
            tier: _ranking_metrics([row for row in predictions if row["tier"] == tier], direct_y)[0]
            for tier in ("small", "medium", "large")
        },
    }, by_assembly


def _load_training(prepared):
    audit_path = prepared["run_root"] / "training-audit.json"
    graph_contract._require(audit_path.is_file(), "training audit is missing")
    audit = _read_json(audit_path)
    graph_contract._require(audit.get("status") == "passed" and audit.get("successful_job_count") == 15, "training audit did not pass")
    graph_contract._require(audit.get("test_rows_loaded") == 0, "training audit loaded test rows")
    return audit_path, audit


def _load_selected_model(prepared, row, device):
    architecture, seed = row["architecture"], row["seed"]
    model = ranker_models.build_model(architecture).to(device)
    result = _read_json(prepared["run_root"] / row["result_path"])
    provenance = result["provenance"]
    ranker_train._validate_provenance(prepared, provenance, architecture, seed)
    checkpoint_path = prepared["run_root"] / row["best_checkpoint"]
    _, sidecar = ranker_train.load_checkpoint(checkpoint_path, model, None, provenance, restore_rng=False)
    graph_contract._require(sidecar["checkpoint_sha256"] == row["best_checkpoint_sha256"], "audited checkpoint hash differs")
    return model


def _bootstrap_plan(validation_rows, replicates, seed):
    assembly_order = sorted(
        ({"assembly_id": row["assembly_id"], "tier": row["tier"]} for row in validation_rows),
        key=lambda row: (row["tier"], row["assembly_id"]),
    )
    rng = np.random.RandomState(seed)
    samples = np.empty((replicates, len(assembly_order)), dtype=np.int64)
    offset = 0
    for tier in ("small", "medium", "large"):
        indices = np.asarray([index for index, row in enumerate(assembly_order) if row["tier"] == tier])
        samples[:, offset:offset + len(indices)] = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        offset += len(indices)
    return assembly_order, samples


def _bootstrap_architecture(architecture, seed_assemblies, assembly_order, samples):
    replicates = []
    point_estimates = []
    for seed in (0, 1, 2):
        values = seed_assemblies[(architecture, seed)]
        numerator = np.asarray([values.get(row["assembly_id"], {}).get("numerator", 0) for row in assembly_order])
        denominator = np.asarray([values.get(row["assembly_id"], {}).get("denominator", 0) for row in assembly_order])
        point_estimates.append(float(numerator.sum() / denominator.sum()) if denominator.sum() else np.nan)
        sampled_denominator = denominator[samples].sum(axis=1)
        sampled_numerator = numerator[samples].sum(axis=1)
        replicates.append(np.divide(sampled_numerator, sampled_denominator, out=np.full(len(samples), np.nan), where=sampled_denominator != 0))
    averaged = np.nanmean(np.stack(replicates), axis=0)
    defined = averaged[np.isfinite(averaged)]
    return {
        "seed_point_estimates": {str(seed): point_estimates[seed] for seed in (0, 1, 2)},
        "seed_mean": float(np.nanmean(point_estimates)),
        "percentile_2_5": float(np.percentile(defined, 2.5)) if len(defined) else None,
        "percentile_97_5": float(np.percentile(defined, 97.5)) if len(defined) else None,
        "undefined_replicates": int((~np.isfinite(averaged)).sum()),
    }, averaged


def validate(spec_path, source_root, data_root, graph_root=None, run_root=None):
    ranker_train.configure_determinism(0)
    prepared = ranker_train._prepare(spec_path, source_root, data_root, graph_root, run_root)
    training_audit_path, training_audit = _load_training(prepared)
    validation_rows = ranker_train._inventory(prepared, "validation")
    fold_path = prepared["output_root"] / prepared["audit"]["validation_folds_path"]
    graph_contract._require(production._sha256_file(fold_path) == prepared["audit"]["validation_folds_sha256"], "validation fold hash differs")
    folds = _read_json(fold_path)["assignments"]
    device = torch.device("cpu")
    reports, calibration_reports, assembly_values = [], [], {}

    for inventory_row in training_audit["inventory"]:
        architecture, seed = inventory_row["architecture"], inventory_row["seed"]
        model = _load_selected_model(prepared, inventory_row, device)
        predictions, inference_seconds = ranker_train.predict(model, prepared, validation_rows, device)
        direct_y = architecture == ranker_models.DIRECT_Y_ARCHITECTURE
        ranking, by_assembly = _ranking_tables(predictions, direct_y)
        assembly_values[(architecture, seed)] = by_assembly
        target_reports, calibrators = {}, {}
        targets = ("Y",) if direct_y else ("R", "T", "Y")
        for target in targets:
            entries = _target_entries(predictions, target)
            fold_temperatures, assembly_temperatures, final_temperature = _cross_fit(entries, folds)
            graph_contract._require(all(value > 0 for value in fold_temperatures.values()) and final_temperature > 0, "temperature must be positive")
            target_reports[target] = {
                "raw": _metric_tables(entries),
                "cross_fitted": _metric_tables(entries, assembly_temperatures),
            }
            calibrators[target] = {
                "fold_temperatures": fold_temperatures,
                "final_all_validation_temperature": final_temperature,
            }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "architecture": architecture,
            "seed": seed,
            "checkpoint_sha256": inventory_row["best_checkpoint_sha256"],
            "parameter_count": inventory_row["parameter_count"],
            "ranking": ranking,
            "targets": target_reports,
            "inference_seconds": inference_seconds,
            "test_rows_loaded": 0,
        }
        report_path = prepared["run_root"] / "validation" / architecture / "seed-{}.json".format(seed)
        _write_json(report_path, report)
        calibration = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "architecture": architecture,
            "seed": seed,
            "method": "positive_scalar_temperature_five_fold_cross_fit",
            "validation_folds_sha256": prepared["audit"]["validation_folds_sha256"],
            "targets": calibrators,
            "Y_order_preserved": True,
            "test_rows_loaded": 0,
        }
        calibration_path = prepared["run_root"] / "calibration" / architecture / "seed-{}.json".format(seed)
        _write_json(calibration_path, calibration)
        reports.append({
            "architecture": architecture, "seed": seed,
            "path": report_path.relative_to(prepared["run_root"]).as_posix(),
            "sha256": production._sha256_file(report_path),
        })
        calibration_reports.append({
            "architecture": architecture, "seed": seed,
            "path": calibration_path.relative_to(prepared["run_root"]).as_posix(),
            "sha256": production._sha256_file(calibration_path),
        })

    primary = {}
    for architecture in ranker_models.PRIMARY_ARCHITECTURES:
        rows = [_read_json(prepared["run_root"] / row["path"]) for row in reports if row["architecture"] == architecture]
        primary[architecture] = {
            "offline_context_success_at_1": float(np.mean([row["ranking"]["overall"]["offline_context_success_at_1"] for row in rows])),
            "first_feasible_mrr": float(np.mean([row["ranking"]["overall"]["first_feasible_mrr"] for row in rows])),
            "Y_average_precision": float(np.mean([row["targets"]["Y"]["raw"]["overall"]["average_precision"] for row in rows])),
            "parameter_count": rows[0]["parameter_count"],
            "inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
        }
    selected = max(primary, key=lambda name: (
        primary[name]["offline_context_success_at_1"],
        primary[name]["first_feasible_mrr"],
        primary[name]["Y_average_precision"],
        -primary[name]["parameter_count"],
        -primary[name]["inference_seconds"],
    ))

    assembly_order, samples = _bootstrap_plan(validation_rows, prepared["spec"]["uncertainty"]["replicates"], prepared["spec"]["uncertainty"]["seed"])
    plan = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "method": "tier_stratified_assembly_with_replacement",
        "seed": prepared["spec"]["uncertainty"]["seed"],
        "replicates": len(samples),
        "assembly_order": assembly_order,
        "sample_indices": samples.tolist(),
    }
    plan_path = prepared["run_root"] / "bootstrap" / "resample-plan.json"
    _write_json(plan_path, plan)
    plan_sha256 = production._sha256_file(plan_path)
    bootstrap_metrics, replicates = {}, {}
    for architecture in ranker_models.PRIMARY_ARCHITECTURES:
        bootstrap_metrics[architecture], replicates[architecture] = _bootstrap_architecture(
            architecture, assembly_values, assembly_order, samples
        )
        bootstrap_metrics[architecture]["resample_plan_sha256"] = plan_sha256
    comparisons = {}
    for architecture in ranker_models.PRIMARY_ARCHITECTURES:
        if architecture == selected:
            continue
        difference = replicates[selected] - replicates[architecture]
        defined = difference[np.isfinite(difference)]
        comparisons[selected + "_minus_" + architecture] = {
            "mean": float(defined.mean()) if len(defined) else None,
            "percentile_2_5": float(np.percentile(defined, 2.5)) if len(defined) else None,
            "percentile_97_5": float(np.percentile(defined, 97.5)) if len(defined) else None,
            "undefined_replicates": int((~np.isfinite(difference)).sum()),
            "resample_plan_sha256": plan_sha256,
        }
    bootstrap_report = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "metric": "offline_context_success_at_1",
        "replicates": len(samples),
        "seed": prepared["spec"]["uncertainty"]["seed"],
        "resample_plan_sha256": plan_sha256,
        "architectures": bootstrap_metrics,
        "paired_comparisons": comparisons,
        "test_rows_loaded": 0,
    }
    bootstrap_path = prepared["run_root"] / "bootstrap" / "report.json"
    _write_json(bootstrap_path, bootstrap_report)
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "spec_sha256": prepared["spec_sha256"],
        "training_audit_sha256": production._sha256_file(training_audit_path),
        "validation_folds_sha256": prepared["audit"]["validation_folds_sha256"],
        "reports": reports,
        "calibration_reports": calibration_reports,
        "bootstrap_plan_sha256": plan_sha256,
        "bootstrap_report_sha256": production._sha256_file(bootstrap_path),
        "primary_architecture_means": primary,
        "selected_architecture": selected,
        "direct_y_excluded_from_selection": True,
        "claim_boundary": "offline observed-context ranking and masked prediction only; not planning success or simulator bypass",
        "test_rows_loaded": 0,
    }
    _write_json(prepared["run_root"] / "validation" / "summary.json", summary)
    return summary


def audit(spec_path, source_root, data_root, graph_root=None, run_root=None):
    prepared = ranker_train._prepare(spec_path, source_root, data_root, graph_root, run_root)
    training_path, _ = _load_training(prepared)
    summary_path = prepared["run_root"] / "validation" / "summary.json"
    summary = _read_json(summary_path)
    graph_contract._require(summary.get("status") == "passed" and summary.get("test_rows_loaded") == 0, "validation summary did not pass")
    graph_contract._require(summary.get("training_audit_sha256") == production._sha256_file(training_path), "training audit hash differs")
    graph_contract._require(len(summary.get("reports", [])) == 15 and len(summary.get("calibration_reports", [])) == 15, "validation report matrix differs")
    for row in summary["reports"] + summary["calibration_reports"]:
        path = prepared["run_root"] / row["path"]
        graph_contract._require(production._sha256_file(path) == row["sha256"], "validation artifact hash differs")
        payload = _read_json(path)
        graph_contract._require(payload.get("test_rows_loaded") == 0, "validation artifact loaded test rows")
        if payload.get("schema_version") == CALIBRATION_SCHEMA_VERSION:
            for target in payload["targets"].values():
                graph_contract._require(
                    target["final_all_validation_temperature"] > 0
                    and all(value > 0 for value in target["fold_temperatures"].values()),
                    "non-positive calibrator",
                )
    plan_path = prepared["run_root"] / "bootstrap" / "resample-plan.json"
    bootstrap_path = prepared["run_root"] / "bootstrap" / "report.json"
    graph_contract._require(production._sha256_file(plan_path) == summary["bootstrap_plan_sha256"], "bootstrap plan hash differs")
    graph_contract._require(production._sha256_file(bootstrap_path) == summary["bootstrap_report_sha256"], "bootstrap report hash differs")
    bootstrap = _read_json(bootstrap_path)
    graph_contract._require(bootstrap["replicates"] == 10000 and bootstrap["seed"] == 20260916, "bootstrap configuration differs")
    graph_contract._require(
        all(row["resample_plan_sha256"] == summary["bootstrap_plan_sha256"] for row in bootstrap["architectures"].values()),
        "bootstrap plan was not reused",
    )
    graph_contract._require(summary["selected_architecture"] in ranker_models.PRIMARY_ARCHITECTURES and summary["direct_y_excluded_from_selection"] is True, "selection boundary differs")
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "spec_sha256": prepared["spec_sha256"],
        "validation_summary_sha256": production._sha256_file(summary_path),
        "training_audit_sha256": production._sha256_file(training_path),
        "report_count": 15,
        "calibration_report_count": 15,
        "bootstrap_replicates": 10000,
        "selected_architecture": summary["selected_architecture"],
        "claim_boundary_verified": True,
        "test_rows_loaded": 0,
    }
    _write_json(prepared["run_root"] / "validation-audit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--spec", default=str(root / "configs" / "ranker_models.json"))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--graph-root")
    parser.add_argument("--run-root")
    parser.add_argument("command", choices=("validate", "audit"))
    args = parser.parse_args(argv)
    common = (args.spec, args.source_root, args.data_root, args.graph_root, args.run_root)
    result = validate(*common) if args.command == "validate" else audit(*common)
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
