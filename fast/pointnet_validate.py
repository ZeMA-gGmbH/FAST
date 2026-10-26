"""Validate PointNet models and make the frozen representation decision."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from fast import graph_contract, ranker_models, ranker_train, ranker_validate, pointnet_models, pointnet_train
from plan_sequence import production_sim_labels as production


def _read_json(path):
    return ranker_train._read_json(path)


def _write_json(path, value):
    ranker_train._write_json(path, value)


def advancement_decision(resource_audits_pass, lower_endpoint):
    advance = bool(resource_audits_pass and lower_endpoint is not None and lower_endpoint > 0.0)
    return {
        "selected_representation": "pointnet_graph_transformer" if advance else "graph_transformer_control",
        "pointnet_advanced": advance,
        "reason": "all_resource_audits_pass_and_difference_percentile_2_5_gt_0" if advance else "frozen_fallback_rule",
    }


def _prediction_hash(predictions):
    digest = hashlib.sha256()
    for row in predictions:
        for name in ("assembly_id", "tier", "query_id"):
            digest.update(row[name].encode("utf-8") + b"\0")
        for name in ("R", "R_valid", "T", "T_valid", "Y", "Y_valid", "Y_score", "R_score", "R_logit", "T_score", "T_logit"):
            value = row.get(name)
            if value is None:
                digest.update(name.encode("ascii") + b":none\n")
            else:
                array = np.ascontiguousarray(value)
                digest.update(name.encode("ascii") + b":" + array.dtype.str.encode("ascii") + b":" + str(array.shape).encode("ascii") + b"\0")
                digest.update(array.tobytes())
    return digest.hexdigest()


def _load_pointnet_model(prepared, row, device, implementation):
    architecture, seed = row["architecture"], row["seed"]
    model = pointnet_train._model(architecture).to(device)
    result = _read_json(prepared["run_root"] / row["result_path"])
    provenance = result["provenance"]
    pointnet_train._validate_gpu_provenance(prepared, provenance, architecture, seed, implementation)
    checkpoint = prepared["run_root"] / row["best_checkpoint"]
    _, sidecar = ranker_train.load_checkpoint(checkpoint, model, None, provenance, restore_rng=False)
    graph_contract._require(sidecar["checkpoint_sha256"] == row["best_checkpoint_sha256"], "PointNet checkpoint hash differs")
    return model


def _control_prepared(prepared, source_root, data_root, graph_root):
    spec = prepared["spec"]
    ranker_spec = prepared["repository_root"] / spec["inputs"]["ranker_spec"]["path"]
    run_root = prepared["data_root"] / spec["artifacts"]["ranker_training"]
    return ranker_train._prepare(ranker_spec, source_root, data_root, graph_root, run_root, require_clean=False)


def _load_control_model(control, row, expected_hash, device):
    graph_contract._require(row["architecture"] == "graph_transformer" and row["best_checkpoint_sha256"] == expected_hash, "control checkpoint inventory differs")
    model = ranker_models.build_model("graph_transformer").to(device)
    result = _read_json(control["run_root"] / row["result_path"])
    provenance = result["provenance"]
    graph_contract._require(
        provenance.get("architecture") == "graph_transformer"
        and provenance.get("seed") == row["seed"]
        and provenance.get("parameter_count") == 90142
        and provenance.get("spec_sha256") == control["spec_sha256"]
        and provenance.get("materialization_audit_sha256") == control["audit_sha256"]
        and provenance.get("test_rows_loaded") == 0,
        "control checkpoint provenance differs",
    )
    checkpoint = control["run_root"] / row["best_checkpoint"]
    _, sidecar = ranker_train.load_checkpoint(checkpoint, model, None, provenance, restore_rng=False)
    graph_contract._require(sidecar["checkpoint_sha256"] == expected_hash, "control checkpoint hash differs")
    return model


def _calibrate(predictions, folds):
    reports, temperatures = {}, {}
    for target in ("R", "T", "Y"):
        entries = ranker_validate._target_entries(predictions, target)
        fold_temperatures, assembly_temperatures, final = ranker_validate._cross_fit(entries, folds)
        reports[target] = {"raw": ranker_validate._metric_tables(entries), "cross_fitted": ranker_validate._metric_tables(entries, assembly_temperatures)}
        temperatures[target] = {"fold_temperatures": fold_temperatures, "final_all_validation_temperature": final}
    return reports, temperatures


def _training_audit(prepared):
    path = prepared["run_root"] / "training-audit.json"
    audit = _read_json(path)
    expected = 6 if audit.get("pointnet_resource_eligible") is True else 3
    graph_contract._require(
        audit.get("status") == "passed"
        and audit.get("successful_job_count") == expected
        and audit.get("artifact_implementation") == pointnet_train._training_implementation(prepared)
        and audit.get("test_rows_loaded") == 0,
        "PointNet training audit did not pass",
    )
    return path, audit


def _resource_gate(prepared, training_audit):
    pilot_path = pointnet_train._pilot_path(prepared, training_audit["artifact_implementation"])
    pilot = _read_json(pilot_path)
    limits = prepared["spec"]["resources"]["gpu"]
    pointnet = [row for row in training_audit["inventory"] if row["architecture"] == "pointnet_graph_transformer"]
    pass_gate = (
        pilot.get("status") == "passed" and pilot.get("eligible") is True and pilot.get("test_rows_loaded") == 0
        and len(pointnet) == 3
        and all(row["observations"]["training_seconds"] <= limits["wall_hours"] * 3600 for row in pointnet)
        and all(row["observations"]["peak_gpu_bytes"] <= 40 * 1024 ** 3 for row in pointnet)
        and all(row["observations"]["peak_host_kib"] <= limits["memory_gib"] * 1024 ** 2 for row in pointnet)
        and all(row["observations"]["checkpoint_bytes"] <= limits["disk_gib"] * 1024 ** 3 for row in pointnet)
    )
    return pass_gate, {"gpu_pilot_sha256": production._sha256_file(pilot_path), "gpu_pilot_eligible": pilot.get("eligible"), "training_limits_passed": pass_gate}


def validate(spec_path, source_root, data_root, graph_root=None, point_root=None, run_root=None):
    ranker_train.configure_determinism(0)
    prepared = pointnet_train._prepare(spec_path, source_root, data_root, graph_root, point_root, run_root)
    training_path, training_audit = _training_audit(prepared)
    validation_rows = pointnet_train._inventory(prepared, "validation")
    fold_path = prepared["output_root"] / prepared["audit"]["validation_folds_path"]
    graph_contract._require(production._sha256_file(fold_path) == prepared["audit"]["validation_folds_sha256"], "validation fold hash differs")
    folds = _read_json(fold_path)["assignments"]
    device = torch.device("cpu")
    reports, calibrations, pointnet_assemblies = [], [], {}
    cebulla_reports = []
    for row in training_audit["inventory"]:
        architecture, seed = row["architecture"], row["seed"]
        model = _load_pointnet_model(prepared, row, device, training_audit["artifact_implementation"])
        if architecture == "pointnet_graph_transformer":
            predictions, inference = pointnet_train._pointnet_predictions(model, prepared, validation_rows, device)
            ranking, by_assembly = ranker_validate._ranking_tables(predictions)
            pointnet_assemblies[(architecture, seed)] = by_assembly
            targets, temperatures = _calibrate(predictions, folds)
            report = {"schema_version": "fast.pointnet-validation-report.v1", "architecture": architecture, "seed": seed,
                      "checkpoint_sha256": row["best_checkpoint_sha256"], "parameter_count": row["parameter_count"],
                      "prediction_sha256": _prediction_hash(predictions), "ranking": ranking, "targets": targets,
                      "inference_seconds": inference, "test_rows_loaded": 0}
            path = prepared["run_root"] / "validation" / architecture / "seed-{}.json".format(seed)
            _write_json(path, report)
            calibration = {"schema_version": "fast.pointnet-calibration.v1", "architecture": architecture, "seed": seed,
                           "method": "positive_scalar_temperature_five_fold_cross_fit", "validation_folds_sha256": prepared["audit"]["validation_folds_sha256"],
                           "targets": temperatures, "Y_order_preserved": True, "test_rows_loaded": 0}
            calibration_path = prepared["run_root"] / "calibration" / architecture / "seed-{}.json".format(seed)
            _write_json(calibration_path, calibration)
            reports.append({"architecture": architecture, "seed": seed, "path": path.relative_to(prepared["run_root"]).as_posix(), "sha256": production._sha256_file(path)})
            calibrations.append({"architecture": architecture, "seed": seed, "path": calibration_path.relative_to(prepared["run_root"]).as_posix(), "sha256": production._sha256_file(calibration_path)})
        else:
            predictions, inference = pointnet_train._cebulla_predictions(model, prepared, validation_rows, device)
            metrics = pointnet_train._cebulla_selection(predictions)
            report = {"schema_version": "fast.cebulla-validation.v1", "architecture": architecture, "seed": seed,
                      "checkpoint_sha256": row["best_checkpoint_sha256"], "parameter_count": row["parameter_count"],
                      "selection": metrics, "inference_seconds": inference, "test_rows_loaded": 0}
            path = prepared["run_root"] / "validation" / architecture / "seed-{}.json".format(seed)
            _write_json(path, report)
            cebulla_reports.append({"architecture": architecture, "seed": seed, "path": path.relative_to(prepared["run_root"]).as_posix(), "sha256": production._sha256_file(path)})

    control = _control_prepared(prepared, source_root, data_root, graph_root)
    control_training_path = control["run_root"] / "training-audit.json"
    graph_contract._require(production._sha256_file(control_training_path) == prepared["spec"]["inputs"]["ranker_training_audit_sha256"], "control training audit hash differs")
    control_training = _read_json(control_training_path)
    control_rows = [row for row in control_training["inventory"] if row["architecture"] == "graph_transformer"]
    control_assemblies, control_hashes = {}, {}
    for row in control_rows:
        seed = row["seed"]
        expected_hash = prepared["spec"]["control"]["checkpoints"][str(seed)]
        model = _load_control_model(control, row, expected_hash, device)
        predictions, _ = ranker_train.predict(model, control, ranker_train._inventory(control, "validation"), device)
        _, by_assembly = ranker_validate._ranking_tables(predictions)
        control_assemblies[("graph_transformer_control", seed)] = by_assembly
        control_hashes[str(seed)] = _prediction_hash(predictions)

    plan_path = control["run_root"] / "bootstrap" / "resample-plan.json"
    graph_contract._require(production._sha256_file(plan_path) == prepared["spec"]["inputs"]["ranker_bootstrap_plan_sha256"], "bootstrap plan hash differs")
    plan = _read_json(plan_path)
    assembly_order, samples = plan["assembly_order"], np.asarray(plan["sample_indices"], dtype=np.int64)
    control_metric, control_replicates = ranker_validate._bootstrap_architecture("graph_transformer_control", control_assemblies, assembly_order, samples)
    graph_contract._require(
        abs(control_metric["seed_mean"] - prepared["spec"]["control"]["mean_validation_success_at_1"]) < 1e-12,
        "control validation result was not reproduced",
    )
    resources_pass, resource_evidence = _resource_gate(prepared, training_audit)
    if resources_pass:
        pointnet_metric, pointnet_replicates = ranker_validate._bootstrap_architecture("pointnet_graph_transformer", pointnet_assemblies, assembly_order, samples)
        difference = pointnet_replicates - control_replicates
        defined = difference[np.isfinite(difference)]
        comparison = {"mean": float(defined.mean()), "percentile_2_5": float(np.percentile(defined, 2.5)),
                      "percentile_97_5": float(np.percentile(defined, 97.5)), "undefined_replicates": int((~np.isfinite(difference)).sum()),
                      "resample_plan_sha256": production._sha256_file(plan_path)}
    else:
        pointnet_metric = None
        comparison = {"mean": None, "percentile_2_5": None, "percentile_97_5": None,
                      "undefined_replicates": len(samples), "resample_plan_sha256": production._sha256_file(plan_path)}
    decision = advancement_decision(resources_pass, comparison["percentile_2_5"])
    decision.update({"comparison": comparison, "resource_evidence": resource_evidence, "frozen_before_planner_runs": True, "test_rows_loaded": 0})
    decision_path = prepared["run_root"] / "representation-decision.json"
    _write_json(decision_path, decision)
    summary = {
        "schema_version": "fast.pointnet-validation-summary.v1", "status": "passed", "spec_sha256": prepared["spec_sha256"],
        "training_audit_sha256": production._sha256_file(training_path), "control_training_audit_sha256": production._sha256_file(control_training_path),
        "control_prediction_sha256": control_hashes, "reports": reports, "calibration_reports": calibrations, "cebulla_reports": cebulla_reports,
        "bootstrap": {"pointnet": pointnet_metric, "control": control_metric, "comparison": comparison, "plan_sha256": production._sha256_file(plan_path)},
        "representation_decision_path": decision_path.relative_to(prepared["run_root"]).as_posix(),
        "representation_decision_sha256": production._sha256_file(decision_path),
        "claim_boundary": "development offline representation selection only; no planner or test claim", "test_rows_loaded": 0,
    }
    _write_json(prepared["run_root"] / "validation" / "summary.json", summary)
    return summary


def audit(spec_path, source_root, data_root, graph_root=None, point_root=None, run_root=None):
    prepared = pointnet_train._prepare(spec_path, source_root, data_root, graph_root, point_root, run_root)
    training_path, _ = _training_audit(prepared)
    summary_path = prepared["run_root"] / "validation" / "summary.json"
    summary = _read_json(summary_path)
    graph_contract._require(summary.get("status") == "passed" and summary.get("test_rows_loaded") == 0, "validation summary did not pass")
    graph_contract._require(summary.get("training_audit_sha256") == production._sha256_file(training_path), "training audit hash differs")
    expected_pointnet = 3 if summary.get("bootstrap", {}).get("pointnet") is not None else 0
    graph_contract._require(len(summary.get("reports", [])) == expected_pointnet and len(summary.get("calibration_reports", [])) == expected_pointnet and len(summary.get("cebulla_reports", [])) == 3, "validation artifact matrix differs")
    for item in summary["reports"] + summary["calibration_reports"] + summary["cebulla_reports"]:
        path = prepared["run_root"] / item["path"]
        graph_contract._require(production._sha256_file(path) == item["sha256"] and _read_json(path).get("test_rows_loaded") == 0, "validation artifact differs")
    decision_path = prepared["run_root"] / summary["representation_decision_path"]
    graph_contract._require(production._sha256_file(decision_path) == summary["representation_decision_sha256"], "representation decision hash differs")
    decision = _read_json(decision_path)
    expected = advancement_decision(decision["resource_evidence"]["training_limits_passed"], decision["comparison"]["percentile_2_5"])
    graph_contract._require(all(decision.get(name) == value for name, value in expected.items()), "advancement rule was not applied")
    report = {"schema_version": "fast.pointnet-validation-audit.v1", "status": "passed", "spec_sha256": prepared["spec_sha256"],
              "validation_summary_sha256": production._sha256_file(summary_path), "representation_decision_sha256": production._sha256_file(decision_path),
              "selected_representation": decision["selected_representation"], "claim_boundary_verified": True, "test_rows_loaded": 0}
    _write_json(prepared["run_root"] / "validation-audit.json", report)
    return report


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(root / "configs" / "pointnet.json"))
    parser.add_argument("--source-root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--graph-root"); parser.add_argument("--point-root"); parser.add_argument("--run-root")
    parser.add_argument("command", choices=("validate", "audit"))
    args = parser.parse_args(argv)
    common = (args.spec, args.source_root, args.data_root, args.graph_root, args.point_root, args.run_root)
    result = validate(*common) if args.command == "validate" else audit(*common)
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
