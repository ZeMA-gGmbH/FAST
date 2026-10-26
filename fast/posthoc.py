"""Read-only post-hoc efficiency and factor-value analysis of a completed planning study.

It reports matched-success resource thresholds with paired tier-stratified bootstrap intervals,
aggregate resource yield, and the validation-only feature-group synthesis. All results are
exploratory; the paired planning-success test of the planning-study audit stays confirmatory.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from fast import complete_case


METHODS = ("asap_random", "asap_heur_out", "asap_released_gatv2", "cebulla_gcn",
           "heur_out_direction_prior", "fast")
BASELINES = METHODS[:-1]
TARGETS = (0.1, 0.2, 0.3)
GROUPS = ("child_stability", "direction", "directional_clearance", "geometry",
          "pose_support", "state", "topology")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260916
# Inputs under the data root, in the layout written by the pipeline commands.
INPUTS = {
    "feature_screen_audit": "features/scalar-features-v3/screening/audit.json",
    "ranker_validation_summary": "runs/ranker-training-v2/validation/summary.json",
    "pointnet_validation_summary": "runs/pointnet-v1/validation/summary.json",
    "pointnet_representation_decision": "runs/pointnet-v1/representation-decision.json",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                          encoding="utf-8")


def _verify_hash(path, expected, label):
    _require(Path(path).is_file() and _sha256(path) == expected, "{} differs".format(label))


def _repository_identity(root):
    root = Path(root)
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=str(root), text=True).strip()
    dirty = bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=str(root), text=True).strip())
    _require(not dirty, "implementation repository is dirty")
    return {"commit": commit, "dirty": False}


def _read_jobs(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    _require(len(rows) == 6399 and [int(row["process"]) for row in rows] == list(range(6399)),
             "planning-study matrix differs")
    return rows


def _trace_path(root, job):
    return (Path(root) / "traces" / "primary" / job["method"]
            / "seed-{}".format(job["seed"]) / (job["assembly_id"] + ".jsonl"))


def _resource_event(actions, summary, part_count):
    denominator = int(part_count) - 1
    _require(denominator > 0, "invalid normalized-resource denominator")
    solution = summary.get("solution_action_validation")
    if solution is None:
        return None, None
    _require(isinstance(solution, int) and 0 < solution <= len(actions), "solution action differs")
    rollouts = sum(row["redmax_rollouts"] + row["stability_restarts"]
                   for row in actions[:solution])
    return solution / denominator, rollouts / denominator


def _read_primary_trace(trace_root, job, registry_sha256, redmax_sha256):
    trace = _trace_path(trace_root, job)
    sidecar = Path(str(trace) + ".sha256")
    _require(trace.is_file() and sidecar.is_file(), "planning-study trace/digest pair is missing")
    digest = sidecar.read_text(encoding="ascii").split()[0]
    _require(digest == _sha256(trace), "planning-study trace hash differs")
    with trace.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    _require(len(rows) >= 2, "planning-study trace is empty")
    header, actions, summary = rows[0], rows[1:-1], rows[-1]
    _require(
        header.get("record_type") == "header"
        and header.get("dataset_scope") == "locked_test"
        and header.get("registry_sha256") == registry_sha256
        and header.get("redmax_binary_sha256") == redmax_sha256
        and header.get("test_rows_loaded") == 1
        and header.get("method") == job["method"]
        and header.get("seed") == int(job["seed"])
        and header.get("assembly_id") == job["assembly_id"]
        and header.get("input_sha256") == job["input_sha256"]
        and summary.get("record_type") == "summary"
        and summary.get("complete") is True
        and summary.get("method") == job["method"]
        and summary.get("seed") == int(job["seed"])
        and summary.get("assembly_id") == job["assembly_id"]
        and summary.get("test_rows_loaded") == 1
        and summary.get("mandatory_redmax_validation") is True
        and summary.get("unvalidated_leaf_shortcut") is False
        and len(summary.get("root_parts", [])) == int(job["part_count"])
        and summary.get("action_validation_count") == len(actions)
        and summary.get("redmax_rollout_count") == sum(row["redmax_rollouts"] for row in actions)
        and summary.get("stability_restart_count") == sum(row["stability_restarts"] for row in actions)
        and [row.get("action_validation") for row in actions] == list(range(1, len(actions) + 1))
        and all(row.get("record_type") == "action" and row["redmax_rollouts"] >= 0
                and row["stability_restarts"] >= 0 for row in actions),
        "planning-study primary trace contract differs",
    )
    validations, rollouts = _resource_event(actions, summary, job["part_count"])
    denominator = int(job["part_count"]) - 1
    _require(
        set(summary.get("budgets", {})) == {"1", "2", "4", "8"}
        and all(summary["budgets"][str(budget)].get("absolute_action_validations") == budget * denominator
                and summary["budgets"][str(budget)].get("success")
                == (validations is not None and validations <= budget)
                for budget in (1, 2, 4, 8)),
        "planning-study success/resource contract differs",
    )
    return {
        "method": job["method"], "seed": int(job["seed"]), "assembly_id": job["assembly_id"],
        "tier": job["tier"], "validations": validations, "rollouts": rollouts,
        "action_validation_count": len(actions),
        "physics_rollout_count": summary["redmax_rollout_count"] + summary["stability_restart_count"],
        "wall_seconds": float(summary["wall_seconds"]), "trace_sha256": digest,
    }


def _empirical_threshold(events, target):
    values = np.asarray(events, dtype=np.float64).reshape(-1)
    rank = int(math.ceil(target * len(values))) - 1
    threshold = float(np.partition(values, rank)[rank])
    return None if not np.isfinite(threshold) else threshold


def _resample_indices(tiers, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    tiers = np.asarray(tiers)
    samples = np.empty((replicates, len(tiers)), dtype=np.int64)
    rng, offset = np.random.RandomState(seed), 0
    for tier in ("small", "medium", "large"):
        indices = np.flatnonzero(tiers == tier)
        _require(len(indices) > 0, "bootstrap tier is empty")
        samples[:, offset:offset + len(indices)] = rng.choice(
            indices, (replicates, len(indices)), replace=True
        )
        offset += len(indices)
    _require(offset == len(tiers), "bootstrap tier inventory differs")
    return samples


def _bootstrap_thresholds(events, samples, targets=TARGETS):
    sampled = np.asarray(events, dtype=np.float64)[samples].reshape(len(samples), -1)
    ranks = [int(math.ceil(target * sampled.shape[1])) - 1 for target in targets]
    thresholds = np.partition(sampled, ranks, axis=1)[:, ranks]
    thresholds[~np.isfinite(thresholds)] = np.nan
    return thresholds


def _matched_success(records, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    assemblies = sorted({(row["tier"], row["assembly_id"]) for row in records})
    _require(len(assemblies) > 0, "matched-success assembly set is empty")
    assembly_index = {assembly_id: index for index, (_, assembly_id) in enumerate(assemblies)}
    samples = _resample_indices([tier for tier, _ in assemblies], replicates, seed)
    result = {}
    for resource in ("validations", "rollouts"):
        arrays = {method: np.full((len(assemblies), 3), np.inf) for method in METHODS}
        seen = set()
        for row in records:
            key = (row["method"], row["assembly_id"], row["seed"])
            _require(key not in seen and row["seed"] in (0, 1, 2), "matched-success row differs")
            seen.add(key)
            value = row[resource]
            arrays[row["method"]][assembly_index[row["assembly_id"]], row["seed"]] = (
                np.inf if value is None else value
            )
        _require(len(seen) == len(METHODS) * 3 * len(assemblies), "matched-success population is incomplete")
        method_thresholds = {
            method: {str(target): _empirical_threshold(values, target) for target in TARGETS}
            for method, values in arrays.items()
        }
        bootstraps = {method: _bootstrap_thresholds(values, samples) for method, values in arrays.items()}
        comparisons = {}
        for baseline in BASELINES:
            comparisons[baseline] = {}
            for column, target in enumerate(TARGETS):
                fast, control = bootstraps["fast"][:, column], bootstraps[baseline][:, column]
                defined = np.isfinite(fast) & np.isfinite(control)
                differences = fast[defined] - control[defined]
                comparisons[baseline][str(target)] = {
                    "fast_threshold": method_thresholds["fast"][str(target)],
                    "baseline_threshold": method_thresholds[baseline][str(target)],
                    "fast_minus_baseline": (None if method_thresholds["fast"][str(target)] is None
                                             or method_thresholds[baseline][str(target)] is None
                                             else method_thresholds["fast"][str(target)]
                                             - method_thresholds[baseline][str(target)]),
                    "percentile_2_5": (float(np.percentile(differences, 2.5)) if len(differences) else None),
                    "percentile_97_5": (float(np.percentile(differences, 97.5)) if len(differences) else None),
                    "undefined_replicates": int(len(defined) - np.count_nonzero(defined)),
                    "fast_undefined_replicates": int(np.count_nonzero(~np.isfinite(fast))),
                    "baseline_undefined_replicates": int(np.count_nonzero(~np.isfinite(control))),
                }
        result[resource] = {"method_thresholds": method_thresholds, "comparisons": comparisons}
    return result


def _metrics(overall):
    return {
        "R_average_precision": overall["R"]["average_precision"],
        "T_average_precision": overall["T"]["average_precision"],
        "Y_average_precision": overall["Y"]["average_precision"],
        "success_at_1": overall["ranking"]["success_at_1"],
        "first_feasible_mrr": overall["ranking"]["first_feasible_mrr"],
    }


def _feature_synthesis(audit, screen_root):
    _require(audit.get("status") == "passed" and audit.get("report_count") == 33,
             "scalar-feature screen audit differs")
    reports = {}
    for row in audit["reports"]:
        _verify_hash(Path(screen_root) / row["path"], row["sha256"], "scalar-feature screen report")
        reports[(row["model"], row["feature_set"])] = row["validation"]["overall"]
    result = {}
    for model in ("histgb", "logistic"):
        full = _metrics(reports[(model, "full")])
        groups = {}
        for group in GROUPS:
            without = _metrics(reports[(model, "without:" + group)])
            groups[group] = {
                "full_minus_without_group": {key: full[key] - without[key] for key in full},
                "only_group": _metrics(reports[(model, "only:" + group)]),
            }
        result[model] = {"full": full, "groups": groups}
    return result, len(audit["reports"])


def _verify_referenced_reports(summary, root):
    for row in summary.get("reports", []):
        _verify_hash(Path(root) / row["path"], row["sha256"], "referenced validation report")


def _component_context(ranker_summary, ranker_root, pointnet_summary, pointnet_decision, decision_sha256):
    bootstrap_path = Path(ranker_root) / "bootstrap" / "report.json"
    _verify_hash(bootstrap_path, ranker_summary["bootstrap_report_sha256"], "ranker bootstrap report")
    _verify_hash(Path(ranker_root) / "bootstrap" / "resample-plan.json",
                 ranker_summary["bootstrap_plan_sha256"], "ranker bootstrap plan")
    bootstrap = _read_json(bootstrap_path)
    comparisons = bootstrap["paired_comparisons"]
    _require(all(row["percentile_2_5"] <= 0 <= row["percentile_97_5"]
                 for row in comparisons.values()), "ranker architecture interval no longer includes zero")
    denominators = {}
    report_rows = {(row["architecture"], row["seed"]): row for row in ranker_summary["reports"]}
    for architecture in ("graph_transformer", "graph_transformer_direct_y"):
        denominators[architecture] = {}
        for seed in (0, 1, 2):
            row = report_rows[(architecture, seed)]
            report = _read_json(Path(ranker_root) / row["path"])
            denominators[architecture][str(seed)] = report["ranking"]["overall"]["denominators"]
    _require(
        ranker_summary.get("status") == pointnet_summary.get("status") == "passed"
        and ranker_summary.get("test_rows_loaded") == pointnet_summary.get("test_rows_loaded") == 0
        and pointnet_summary.get("representation_decision_sha256") == decision_sha256
        and pointnet_decision.get("pointnet_advanced") is False
        and pointnet_decision.get("selected_representation") == "graph_transformer_control",
        "ranker/PointNet component evidence differs",
    )
    return {
        "architecture": {
            "status": "unresolved",
            "metric": bootstrap["metric"],
            "paired_graph_transformer_comparisons": comparisons,
            "reason": "all three paired 95% intervals include zero",
        },
        "factorization": {
            "status": "unresolved",
            "ranking_denominators_by_seed": denominators,
            "reason": "factorized and direct-Y diagnostics use different ranking denominators",
        },
        "pointnet": {
            "status": "did_not_advance",
            "comparison": pointnet_summary["bootstrap"]["comparison"],
            "selected_representation": pointnet_decision["selected_representation"],
        },
        "study": {
            "status": "end_to_end_performance_only",
            "causal_component_attribution": False,
        },
    }


def _descriptive_accounting(aggregate, records):
    wall = {method: [] for method in METHODS}
    for row in records:
        wall[row["method"]].append(row["wall_seconds"] / 60.0)
    result = {}
    for method in METHODS:
        successes = sum(aggregate["success"][method][str(seed)]["all"]["8"]["success"]
                        for seed in (0, 1, 2))
        denominator = sum(aggregate["success"][method][str(seed)]["all"]["8"]["total"]
                          for seed in (0, 1, 2))
        accounting = aggregate["accounting"][method]
        values = np.asarray(wall[method])
        method_records = [row for row in records if row["method"] == method]
        _require(
            len(values) == denominator
            and successes == sum(row["validations"] is not None and row["validations"] <= 8
                                 for row in method_records)
            and accounting["action_validations"] == sum(row["action_validation_count"]
                                                         for row in method_records)
            and accounting["physics_rollouts"] == sum(row["physics_rollout_count"]
                                                       for row in method_records)
            and np.isclose(accounting["wall_seconds"], sum(row["wall_seconds"]
                                                           for row in method_records)),
            "aggregate resource-yield accounting differs",
        )
        result[method] = {
            "budget_8_successes": successes, "denominator": denominator,
            "budget_8_success_rate": successes / denominator,
            "total_validations": accounting["action_validations"],
            "validations_per_budget_8_success": accounting["action_validations"] / successes,
            "total_physics_rollouts": accounting["physics_rollouts"],
            "physics_rollouts_per_budget_8_success": accounting["physics_rollouts"] / successes,
            "total_wall_minutes": accounting["wall_seconds"] / 60.0,
            "wall_minutes_per_budget_8_success": accounting["wall_seconds"] / 60.0 / successes,
            "wall_minutes_per_run": {
                "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                "median": float(np.median(values)), "percentile_25": float(np.percentile(values, 25)),
                "percentile_75": float(np.percentile(values, 75)),
            },
        }
    return result


def _format_number(value):
    return "undefined" if value is None else "{:.6f}".format(value)


def _interpretation(summary):
    lines = [
        "# Post-Hoc Efficiency and Factor-Value Analysis", "",
        "All results are post-hoc exploratory. The paired planning-success test remains the confirmatory "
        "result. The analysis retains {} of {} predefined assemblies after the assembly-wide exclusion of "
        "assemblies with repeatable simulator execution failures; no outcomes are imputed.".format(
            summary["complete_case"]["analyzed_assemblies"], summary["complete_case"]["predefined_assemblies"]), "",
    ]
    labels = {"validations": "Normalized action validations", "rollouts": "Normalized physics rollouts"}
    for resource, label in labels.items():
        lines += ["## {} at matched success".format(label), "",
                  "| Target | Baseline | FAST | Baseline | FAST - baseline | 95% percentile interval | Undefined replicates |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for target in TARGETS:
            for baseline in BASELINES:
                row = summary["matched_success"][resource]["comparisons"][baseline][str(target)]
                interval = "[{}, {}]".format(_format_number(row["percentile_2_5"]),
                                               _format_number(row["percentile_97_5"]))
                lines.append("| {:.0%} | {} | {} | {} | {} | {} | {} |".format(
                    target, baseline, _format_number(row["fast_threshold"]),
                    _format_number(row["baseline_threshold"]),
                    _format_number(row["fast_minus_baseline"]), interval,
                    row["undefined_replicates"],
                ))
        lines += ["", "Negative differences favor FAST. Intervals are exploratory uncertainty summaries, "
                  "not significance tests.", ""]
        comparisons = [
            (baseline, target, summary["matched_success"][resource]["comparisons"][baseline][str(target)])
            for target in TARGETS for baseline in BASELINES
        ]
        negative = sum(row["fast_minus_baseline"] is not None and row["fast_minus_baseline"] < 0
                       for _, _, row in comparisons)
        below_zero = sum(row["percentile_97_5"] is not None and row["percentile_97_5"] < 0
                         for _, _, row in comparisons)
        unresolved = ["{} at {:.0%}".format(baseline, target) for baseline, target, row in comparisons
                      if row["percentile_97_5"] is not None and row["percentile_97_5"] >= 0]
        undefined = max(row["undefined_replicates"] for _, _, row in comparisons)
        lines += [
            "FAST has the lower point threshold in {} of {} comparisons; {} percentile intervals lie "
            "strictly below zero. Intervals touching or including zero: {}. At the 30% target, some "
            "baseline resamples do not reach the target (maximum undefined replicates: {}).".format(
                negative, len(comparisons), below_zero, ", ".join(unresolved) or "none", undefined
            ), "",
        ]
    lines += ["## Aggregate resource yield", "",
              "These quantities include resources spent on unsuccessful runs; they are not conditional means among successful runs.", "",
              "| Method | Budget-8 successes / runs | Validations / success | Rollouts / success | Wall minutes / success | Mean wall minutes / run |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        row = summary["aggregate_resource_yield"][method]
        lines.append("| {} | {} / {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
            method, row["budget_8_successes"], row["denominator"],
            row["validations_per_budget_8_success"], row["physics_rollouts_per_budget_8_success"],
            row["wall_minutes_per_budget_8_success"], row["wall_minutes_per_run"]["mean"],
        ))
    lines += [
        "", "Wall minutes per success are descriptive: they mix method success yield with uncontrolled "
        "machine and simulation costs.",
    ]
    for model in ("histgb", "logistic"):
        lines += ["", "## {} feature-group synthesis".format(
            "HistGradientBoosting" if model == "histgb" else "Logistic-regression robustness"
        ), "", "All values are validation-only descriptive evidence.", "",
                  "| Group | ΔR AP | ΔT AP | ΔY AP | ΔSuccess@1 | ΔMRR | Only R AP | Only T AP | Only Y AP | Only Success@1 | Only MRR |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in GROUPS:
            row = summary["factor_value"][model]["groups"][group]
            delta, only = row["full_minus_without_group"], row["only_group"]
            keys = ("R_average_precision", "T_average_precision", "Y_average_precision",
                    "success_at_1", "first_feasible_mrr")
            lines.append("| {} | {} |".format(group, " | ".join(
                "{:.6f}".format(values[key]) for values in (delta, only) for key in keys
            )))
        largest = []
        for key in ("R_average_precision", "T_average_precision", "Y_average_precision",
                    "success_at_1", "first_feasible_mrr"):
            group = max(GROUPS, key=lambda candidate: summary["factor_value"][model]["groups"][candidate]
                        ["full_minus_without_group"][key])
            value = summary["factor_value"][model]["groups"][group]["full_minus_without_group"][key]
            largest.append("{}: {} ({:+.6f})".format(key, group, value))
        lines += ["", "Largest leave-one-group-out decrease by metric (not a universal ranking): {}.".format(
            "; ".join(largest)
        )]
    lines += [
        "", "## Interpretation boundary", "",
        "The matched-success estimates support only exploratory efficiency comparisons; their percentile "
        "intervals are not confirmatory tests. Wall time is descriptive because cluster contention and "
        "heterogeneous simulation costs were uncontrolled. The legacy counter is excluded because it has no "
        "FAST arm and is not commensurate with action validations.", "",
        "Feature-group changes measure predictive and ranking value under the audited validation models. They "
        "are not causal importance or physical-mechanism evidence. Graph Transformer superiority and the "
        "factorization contribution remain unresolved, the direct-Y ranking denominator mismatch was not "
        "repaired, and PointNet did not improve on the scalar control under the frozen advancement rule. "
        "The planning study establishes end-to-end FAST planner performance but cannot identify a single causal component.", "",
        "No claim extends beyond the frozen six-direction, G=2 ASAP/RedMax testbed.", "",
    ]
    return "\n".join(lines)


def analyze(data_root, run_root, registry_path, implementation_root, output_root, excluded=(), aggregate_path=None):
    data_root, run_root = Path(data_root).resolve(), Path(run_root).resolve()
    implementation_root, output_root = Path(implementation_root).resolve(), Path(output_root).resolve()
    _require(not output_root.exists(), "post-hoc output root cannot be overwritten")
    identity = _repository_identity(implementation_root)
    paths = {name: data_root / relative for name, relative in INPUTS.items()}
    paths.update(study_registry=Path(registry_path).resolve(), study_matrix=run_root / "jobs.tsv",
                 study_aggregate=Path(aggregate_path or run_root / "aggregate" / "summary.json").resolve())
    inputs = {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}
    values = {name: _read_json(path) for name, path in paths.items() if path.suffix == ".json"}
    registry, aggregate = values["study_registry"], values["study_aggregate"]
    jobs = _read_jobs(paths["study_matrix"])
    excluded = frozenset(excluded)
    retained, assemblies = complete_case.retained_jobs(jobs, excluded, METHODS, METHODS[:3])
    primary = [row for row in retained if row["counter"] == "action_validation"]
    _require(
        registry.get("status") == "locked" and aggregate.get("status") == "passed"
        and aggregate.get("registry_sha256") == inputs["study_registry"]["sha256"]
        and aggregate.get("test_assembly_count") == len(assemblies)
        and aggregate.get("valid_primary_trace_count") == len(primary)
        and aggregate.get("job_count") == len(retained),
        "planning-study aggregate does not match the complete-case population",
    )
    records = [_read_primary_trace(run_root, job, inputs["study_registry"]["sha256"], registry["redmax"]["binary_sha256"])
               for job in primary]
    matched = _matched_success(records)
    feature_root = data_root / "features" / "scalar-features-v3"
    factor_value, feature_report_count = _feature_synthesis(values["feature_screen_audit"], feature_root)
    ranker_root = data_root / "runs" / "ranker-training-v2"
    pointnet_root = data_root / "runs" / "pointnet-v1"
    _verify_referenced_reports(values["ranker_validation_summary"], ranker_root)
    _verify_referenced_reports(values["pointnet_validation_summary"], pointnet_root)
    component_context = _component_context(
        values["ranker_validation_summary"], ranker_root,
        values["pointnet_validation_summary"], values["pointnet_representation_decision"],
        inputs["pointnet_representation_decision"]["sha256"],
    )
    tier_counts = {tier: sum(row[0] == tier for row in sorted({(r["tier"], r["assembly_id"]) for r in records}))
                   for tier in ("small", "medium", "large")}
    summary = {
        "schema_version": "fast.posthoc-summary.v1", "status": "passed",
        "analysis_class": "post_hoc_exploratory",
        "implementation": identity,
        "complete_case": {
            "predefined_assemblies": len(assemblies) + len(excluded), "excluded_assemblies": len(excluded),
            "analyzed_assemblies": len(assemblies), "primary_rows": len(primary),
            "exclusion_reason": "repeatable simulator execution failures",
            "outcome_imputation": False, "tier_counts": tier_counts,
        },
        "matched_success": {
            "experimental_unit": "assembly", "seeds_carried_together": [0, 1, 2],
            "targets": list(TARGETS), "empirical_curve": "step_function_no_smoothing_or_extrapolation",
            "censoring": "unsolved runs remain right-censored at observed resource ceilings",
            "bootstrap": "paired tier-stratified assembly resampling",
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "interval": "95% percentile", "hypothesis_tests": False,
            **matched,
        },
        "aggregate_resource_yield": _descriptive_accounting(aggregate, records),
        "factor_value": {
            "scope": "validation_only_descriptive", "causal_importance": False,
            **factor_value, "component_context": component_context,
        },
        "legacy_counter_included": False,
        "matched_wall_time_inference": False,
        "claim_boundary": "exploratory; the paired planning-success test remains the confirmatory result",
        "per_assembly_test_payload_export": False,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".posthoc-", dir=str(output_root.parent)) as temporary:
        staging = Path(temporary) / output_root.name
        staging.mkdir()
        summary_path = staging / "summary.json"
        interpretation_path = staging / "interpretation.md"
        _write_json(summary_path, summary)
        interpretation_path.write_text(_interpretation(summary), encoding="utf-8")
        audit = {
            "schema_version": "fast.posthoc-audit-manifest.v1", "status": "passed",
            "implementation": identity, "inputs": inputs,
            "transitive_verification": {
                "feature_report_hashes": feature_report_count,
                "ranker_validation_report_hashes": len(values["ranker_validation_summary"]["reports"]),
                "pointnet_validation_report_hashes": len(values["pointnet_validation_summary"]["reports"]),
                "ranker_bootstrap_report_sha256": values["ranker_validation_summary"]["bootstrap_report_sha256"],
                "ranker_bootstrap_plan_sha256": values["ranker_validation_summary"]["bootstrap_plan_sha256"],
                "primary_trace_digest_pairs": len(records),
                "primary_trace_digest_inventory_sha256": hashlib.sha256(
                    json.dumps(sorted(row["trace_sha256"] for row in records), separators=(",", ":"))
                    .encode("ascii")
                ).hexdigest(),
            },
            "outputs": {
                "summary.json": {"sha256": _sha256(summary_path), "bytes": summary_path.stat().st_size},
                "interpretation.md": {"sha256": _sha256(interpretation_path),
                                      "bytes": interpretation_path.stat().st_size},
            },
            "prohibited_processes_invoked": [], "test_payload_export": False,
        }
        audit_path = staging / "audit-manifest.json"
        _write_json(audit_path, audit)
        Path(str(audit_path) + ".sha256").write_text(
            "{}  {}\n".format(_sha256(audit_path), audit_path.name), encoding="ascii"
        )
        os.replace(str(staging), str(output_root))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True, help="planning-study run root (setup.json, jobs.tsv, traces/)")
    parser.add_argument("--registry", required=True, help="locked planning-study registry")
    parser.add_argument("--aggregate", help="audit summary; default: <run-root>/aggregate/summary.json")
    parser.add_argument("--exclude-assembly", action="append", default=[],
                        help="assembly excluded from every method after a failed retry (repeatable)")
    parser.add_argument("--implementation-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = analyze(args.data_root, args.run_root, args.registry, args.implementation_root, args.output_root,
                     args.exclude_assembly, args.aggregate)
    print(json.dumps({"status": result["status"], "output_root": str(Path(args.output_root).resolve())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
