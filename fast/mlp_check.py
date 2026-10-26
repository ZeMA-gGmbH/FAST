"""Post-hoc seed-0 MLP planning check on the complete-case assemblies of a planning study.

Reuses the locked planning-study registry, planner, and test-access guards; only the ranker
architecture changes (Graph Transformer -> MLP seed 0). The result is one-seed sanity evidence;
the six-method planning study stays the confirmatory result.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

from fast import graph_contract, action_planner, complete_case
from fast import planner_integration as integration
from fast import planning_study
from plan_sequence import production_sim_labels as production


REGISTRY_SCHEMA = "fast.mlp-check-registry.v1"
AUDIT_SCHEMA = "fast.mlp-check-registry-audit.v1"
TRACE_AUDIT_SCHEMA = "fast.mlp-check-trace-audit.v1"
ANALYSIS_SCHEMA = "fast.mlp-check-paired-analysis.v1"
METHOD = integration.MLP_CHECK_METHOD
TRAINING_RUN = "runs/ranker-training-v2"
OUTPUT_ROOT = "runs/mlp-check-v1"
TIERS = ("small", "medium", "large")
MATRIX_FIELDS = ("process", "method", "seed", "assembly_id", "tier", "part_count", "input_sha256")
CONTRACT = {
    "methods": [METHOD], "seeds": [0], "h_max": 1, "grippers_total": 2,
    "directions": list(action_planner.DIRECTIONS), "budgets": [1, 2, 4, 8],
    "max_action_validations": "8*(part_count-1)", "max_stable_poses": 3, "stable_pose_probability": 0.9,
    "removal_timeout_seconds": 60, "stability_timeout_seconds": 900, "wall_timeout_seconds": 7200,
    "redmax_validation_required": True, "unvalidated_two_part_leaf": False, "legacy_counter_enabled": False,
    "ranking": "factorized Y=sigma(R)*sigma(T); frozen calibration is order-preserving (Y_order_preserved)",
}
FAILURE_POLICY = {
    "imputation": False,
    "valid_trace": ("complete trace bound to the registry, matrix process, assembly input, MLP checkpoint and "
                    "calibration, with mandatory RedMax validation; a planner wall-limit trace is a valid unsolved run"),
    "technical_failure": "no valid trace after a crash, a nonzero exit, or a scheduler removal",
    "retry": "at most one outcome-blind retry per technical failure, from the unchanged matrix row",
    "pairing": ("assembly-complete: an assembly whose MLP row has no valid trace after its retry is excluded from "
                "both FAST and MLP; the excluded count is disclosed and nothing is imputed"),
}
BUDGETS = (1, 2, 4, 8)

_require = graph_contract._require
_read_json = planning_study._read_json


def _mlp_model(data_root):
    """Seed-0 MLP checkpoint, calibration, and validation report from the ranker training run."""
    root = Path(data_root) / TRAINING_RUN
    audit = _read_json(root / "training-audit.json")
    row = integration._checkpoint_rows(audit, "mlp")[0]
    checkpoint = root / row["best_checkpoint"]
    _require(production._sha256_file(checkpoint) == row["best_checkpoint_sha256"], "MLP checkpoint differs")
    files = {"checkpoint": checkpoint, "sidecar": Path(str(checkpoint) + ".json"),
             "calibration": root / "calibration" / "mlp" / "seed-0.json",
             "validation": root / "validation" / "mlp" / "seed-0.json"}
    model = {"seed": 0, "parameter_count": row["parameter_count"], "normalization_sha256": audit["normalization_sha256"]}
    for name, path in files.items():
        model[name] = path.relative_to(data_root).as_posix()
        model[name + "_sha256"] = production._sha256_file(path)
    return model


def _matrix_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MATRIX_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for process, row in enumerate(rows):
        writer.writerow({"process": process, "method": METHOD, "seed": 0, "assembly_id": row["assembly_id"],
                         "tier": row["tier"], "part_count": row["part_count"], "input_sha256": row["input_sha256"]})
    return stream.getvalue()


def _head(repository_root):
    git = lambda *args: subprocess.run(["git", "-C", str(repository_root)] + list(args), check=True,
                                       stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    _require(git("status", "--porcelain", "--untracked-files=normal") == "", "implementation repository is dirty")
    return git("rev-parse", "HEAD")


def build(repository_root, data_root, run_root, study_registry, excluded=()):
    data_root = Path(data_root)
    source = _read_json(study_registry)
    rows = complete_case.population(run_root, excluded)
    matrix = _matrix_text(rows)
    registry = copy.deepcopy(source)
    registry.update({
        "schema_version": REGISTRY_SCHEMA, "status": "locked",
        "implementation": {"commit": _head(repository_root), "dirty": False},
        "source_study_registry_sha256": production._sha256_file(study_registry),
        "models": {"architecture": "mlp", "factorization": source["models"]["factorization"],
                   "seeds": [_mlp_model(data_root)]},
        "seeds": [0], "planner_contract": CONTRACT,
        "population": {"assembly_count": len(rows), "tiers": {tier: sum(row["tier"] == tier for row in rows) for tier in TIERS},
                       "excluded_assemblies_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode("utf-8")).hexdigest()},
        "matrix": {"sha256": hashlib.sha256(matrix.encode("utf-8")).hexdigest(), "row_count": len(rows)},
        "output_root": OUTPUT_ROOT,
        "trace_schema": {"version": action_planner.TRACE_SCHEMA_VERSION,
                         "implementation_sha256": source["trace_schema"]["implementation_sha256"],
                         "dataset_scope": "locked_test"},
        "aggregate_outputs": {"per_assembly_test_payload_export": False, "root": OUTPUT_ROOT + "/aggregate"},
        "claim_boundary": "post-hoc one-seed sanity evidence; the six-method planning study remains the confirmatory result",
    })
    return registry, matrix


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def check_contract(registry):
    """Reject any registry that is not exactly the seed-0 MLP planning check."""
    model = registry.get("models", {}).get("seeds", [{}])[0]
    _require(registry.get("schema_version") == REGISTRY_SCHEMA and registry.get("status") == "locked"
             and registry.get("models", {}).get("architecture") == "mlp"
             and model.get("seed") == 0 and _is_sha256(model.get("checkpoint_sha256"))
             and _is_sha256(model.get("calibration_sha256"))
             and registry.get("planner_contract") == CONTRACT
             and registry.get("population", {}).get("assembly_count") == registry.get("matrix", {}).get("row_count")
             and registry.get("output_root") == OUTPUT_ROOT
             and registry.get("aggregate_outputs", {}).get("per_assembly_test_payload_export") is False,
             "registry is not the MLP planning-check contract")
    return registry


def prepare(repository_root, data_root, run_root, study_registry, output_dir, excluded=()):
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), "MLP check registry cannot be overwritten")
    registry, matrix = build(repository_root, data_root, run_root, study_registry, excluded)
    check_contract(registry)
    output_dir.mkdir(parents=True)
    (output_dir / "matrix.tsv").write_text(matrix, encoding="utf-8")
    from fast.graph_materialize import _write_json
    _write_json(output_dir / "registry.json", registry)
    _write_json(output_dir / "registry-audit.json",
                verify(repository_root, data_root, run_root, study_registry, output_dir, excluded))
    return {name: production._sha256_file(output_dir / name) for name in ("registry.json", "matrix.tsv", "registry-audit.json")}


def verify(repository_root, data_root, run_root, study_registry, registry_dir, excluded=()):
    """Independent replay: rebuild from the inputs and compare byte-for-byte."""
    registry_dir = Path(registry_dir)
    registry, matrix = build(repository_root, data_root, run_root, study_registry, excluded)
    _require(_read_json(registry_dir / "registry.json") == registry
             and (registry_dir / "matrix.tsv").read_text(encoding="utf-8") == matrix, "MLP check registry replay differs")
    return {"schema_version": AUDIT_SCHEMA, "status": "passed",
            "registry_sha256": production._sha256_file(registry_dir / "registry.json"),
            "matrix_sha256": production._sha256_file(registry_dir / "matrix.tsv"),
            "rows": registry["matrix"]["row_count"], "test_trace_count": 0, "outcomes_accessed": False}


def _matrix_rows(registry_dir, registry):
    matrix_path = Path(registry_dir) / "matrix.tsv"
    _require(production._sha256_file(matrix_path) == registry["matrix"]["sha256"], "MLP check matrix differs")
    with matrix_path.open("r", encoding="utf-8", newline="") as stream:
        return {int(row["process"]): row for row in csv.DictReader(stream, delimiter="\t")}


def evaluate(spec_path, data_root, source_root, asset_folder, registry_dir, registry_sha256,
             released_checkpoint, direction_prior, process):
    registry_path = Path(registry_dir) / "registry.json"
    check_contract(planning_study._locked(registry_path, registry_sha256))  # registry before any test access
    registry = _read_json(registry_path)
    rows = _matrix_rows(registry_dir, registry)
    _require(process in rows, "MLP process differs")
    row = rows[process]
    assembly_dir = Path(source_root) / "test_assembly" / row["assembly_id"]
    _require(production._hash_assembly(assembly_dir) == row["input_sha256"], "MLP check assembly input differs")
    model = registry["models"]["seeds"][0]
    result = integration.run_planner(
        spec_path, data_root, asset_folder, assembly_dir, _mlp_trace(data_root, row["assembly_id"]), METHOD, 0,
        released_checkpoint=released_checkpoint, direction_prior_path=direction_prior,
        registry=registry_path, registry_sha256=registry_sha256, development=False,
        trace_context={"architecture": "mlp", "checkpoint_sha256": model["checkpoint_sha256"],
                       "calibration_sha256": model["calibration_sha256"], "planner_contract": CONTRACT,
                       "matrix_process": process, "output_root": OUTPUT_ROOT},
    )
    _require(result["summary"]["test_rows_loaded"] == 1, "MLP check trace test count differs")
    return {"status": "trace_written", "trace_sha256": result["trace_sha256"]}


def _summary_lines(trace):
    lines = Path(trace).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:-1]], json.loads(lines[-1])


def _mlp_trace(data_root, assembly_id):
    return Path(data_root) / OUTPUT_ROOT / "traces" / "primary" / METHOD / "seed-0" / (assembly_id + ".jsonl")


def _valid_mlp_trace(data_root, row, registry_sha256, model):
    """Technical contract only; success values are read by nobody here."""
    trace = _mlp_trace(data_root, row["assembly_id"])
    if not trace.is_file():
        return None
    digest = Path(str(trace) + ".sha256").read_text(encoding="ascii").split()[0]
    _require(digest == production._sha256_file(trace), "MLP trace digest differs")
    header, actions, summary = _summary_lines(trace)
    parts = int(row["part_count"])
    _require(
        header["dataset_scope"] == "locked_test" and header["registry_sha256"] == registry_sha256
        and header["method"] == METHOD and header["seed"] == 0 and header["assembly_id"] == row["assembly_id"]
        and header["input_sha256"] == row["input_sha256"] and header["architecture"] == "mlp"
        and header["checkpoint_sha256"] == model["checkpoint_sha256"]
        and header["calibration_sha256"] == model["calibration_sha256"]
        and header["matrix_process"] == int(row["process"]) and header["test_rows_loaded"] == summary["test_rows_loaded"] == 1
        and summary["complete"] is True and summary["mandatory_redmax_validation"] is True
        and summary["unvalidated_leaf_shortcut"] is False and summary["action_validation_count"] == len(actions)
        and all(summary["budgets"][str(b)]["absolute_action_validations"] == b * (parts - 1) for b in BUDGETS)
        and all(not a["accepted_transition"] or (a["R"]["label"] is True and a["T"]["label"] is True) for a in actions),
        "MLP trace technical contract differs",
    )
    return digest


def audit_traces(data_root, registry_dir, registry_sha256, output_path):
    """Technical trace accounting; rows without a valid trace are technical failures. No outcomes."""
    registry = check_contract(_read_json(Path(registry_dir) / "registry.json"))
    _require(production._sha256_file(Path(registry_dir) / "registry.json") == registry_sha256, "MLP registry differs")
    model = registry["models"]["seeds"][0]
    records = [{"process": process, "trace_sha256": _valid_mlp_trace(data_root, row, registry_sha256, model)}
               for process, row in sorted(_matrix_rows(registry_dir, registry).items())]
    for record in records:
        record["valid_trace"] = record["trace_sha256"] is not None
    report = {
        "schema_version": TRACE_AUDIT_SCHEMA, "status": "complete", "registry_sha256": registry_sha256,
        "failure_policy": FAILURE_POLICY, "rows": len(records),
        "valid_traces": sum(r["valid_trace"] for r in records),
        "technical_failure_processes": [r["process"] for r in records if not r["valid_trace"]],
        "outcomes_accessed": False, "records": records,
    }
    output_path = Path(output_path)
    _require(not output_path.exists(), "MLP trace audit cannot be overwritten")
    from fast.graph_materialize import _write_json
    _write_json(output_path, report)
    return {"trace_audit_sha256": production._sha256_file(output_path), "valid_traces": report["valid_traces"],
            "technical_failures": len(report["technical_failure_processes"])}


def paired_intervals(fast, mlp, tiers, seed=integration.BOOTSTRAP_SEED, replicates=integration.BOOTSTRAP_REPLICATES):
    """Paired tier-stratified assembly resampling, intervals only: FAST minus MLP per budget."""
    import numpy as np

    assemblies = sorted(fast)
    _require(set(fast) == set(mlp) == set(tiers), "paired assemblies differ")
    rng = np.random.RandomState(seed)
    samples = np.empty((replicates, len(assemblies)), dtype=np.int64)
    offset = 0
    for tier in ("small", "medium", "large"):
        indices = np.asarray([i for i, a in enumerate(assemblies) if tiers[a] == tier], dtype=np.int64)
        if len(indices):
            samples[:, offset:offset + len(indices)] = rng.choice(indices, (replicates, len(indices)), replace=True)
        offset += len(indices)
    result = {}
    for budget in BUDGETS:
        f = np.asarray([float(fast[a][str(budget)]) for a in assemblies])
        m = np.asarray([float(mlp[a][str(budget)]) for a in assemblies])
        differences = f - m
        replicate_means = differences[samples].mean(axis=1)
        result[str(budget)] = {
            "n_paired": len(assemblies), "fast_success": int(f.sum()), "mlp_success": int(m.sum()),
            "fast_rate": float(f.mean()), "mlp_rate": float(m.mean()), "fast_minus_mlp": float(differences.mean()),
            "percentile_2_5": float(np.percentile(replicate_means, 2.5)),
            "percentile_97_5": float(np.percentile(replicate_means, 97.5)),
        }
    return result


def analyze(data_root, run_root, registry_dir, registry_sha256, trace_audit_path, trace_audit_sha256, output_path):
    """Aggregate-only paired FAST seed 0 vs MLP seed 0 result; run once, after the trace audit."""
    data_root = Path(data_root)
    _require(production._sha256_file(trace_audit_path) == trace_audit_sha256, "MLP trace audit differs")
    audit = _read_json(trace_audit_path)
    registry = _read_json(Path(registry_dir) / "registry.json")
    rows = _matrix_rows(registry_dir, registry)
    _require(audit["registry_sha256"] == registry_sha256 and audit["rows"] == len(rows), "MLP trace audit binding differs")
    fast, mlp, tiers, descriptive = {}, {}, {}, {"fast": [], "mlp": []}
    for record in audit["records"]:
        if not record["valid_trace"]:
            continue  # assembly-complete pairing: excluded for both arms
        row = rows[record["process"]]
        assembly_id = row["assembly_id"]
        mlp_path = _mlp_trace(data_root, assembly_id)
        fast_path = Path(run_root) / "traces" / "primary" / "fast" / "seed-0" / (assembly_id + ".jsonl")
        fast_digest = Path(str(fast_path) + ".sha256").read_text(encoding="ascii").split()[0]
        _require(production._sha256_file(mlp_path) == record["trace_sha256"]
                 and production._sha256_file(fast_path) == fast_digest, "paired trace digest differs")
        for arm, path, store in (("mlp", mlp_path, mlp), ("fast", fast_path, fast)):
            summary = _summary_lines(path)[2]
            store[assembly_id] = {str(b): summary["budgets"][str(b)]["success"] is True for b in BUDGETS}
            descriptive[arm].append(summary)
        tiers[assembly_id] = row["tier"]
    def describe(summaries):
        return {"action_validations": sum(s["action_validation_count"] for s in summaries),
                "physics_rollouts": sum(s["redmax_rollout_count"] + s["stability_restart_count"] for s in summaries),
                "wall_seconds": sum(s["wall_seconds"] for s in summaries),
                "wall_timeouts": sum(s["stop_reason"] == "wall_timeout" for s in summaries)}
    report = {
        "schema_version": ANALYSIS_SCHEMA, "registry_sha256": registry_sha256,
        "trace_audit_sha256": trace_audit_sha256,
        "pairing": FAILURE_POLICY["pairing"], "n_paired": len(fast), "excluded_assemblies": len(rows) - len(fast),
        "paired_tier_counts": {tier: sum(value == tier for value in tiers.values()) for tier in TIERS},
        "bootstrap": {"scheme": "paired tier-stratified assembly resampling", "seed": integration.BOOTSTRAP_SEED,
                      "replicates": integration.BOOTSTRAP_REPLICATES, "interval": "percentile 95%", "p_values": False},
        "budgets": paired_intervals(fast, mlp, tiers),
        "descriptive": {arm: describe(values) for arm, values in descriptive.items()},
        "claim_boundary": "post-hoc one-seed sanity evidence; no equivalence or general encoder claim; "
                          "the six-method planning study stays confirmatory",
    }
    output_path = Path(output_path)
    _require(not output_path.exists(), "MLP paired analysis cannot be overwritten")
    from fast.graph_materialize import _write_json
    _write_json(output_path, report)
    return {"analysis_sha256": production._sha256_file(output_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        sub = commands.add_parser(name)
        for flag in ("repository-root", "data-root", "run-root", "study-registry"):
            sub.add_argument("--" + flag, required=True)
        sub.add_argument("--output-dir" if name == "prepare" else "--registry-dir", required=True)
        sub.add_argument("--exclude-assembly", action="append", default=[])
    sub = commands.add_parser("audit-traces")
    for flag in ("data-root", "registry-dir", "registry-sha256", "output"):
        sub.add_argument("--" + flag, required=True)
    sub = commands.add_parser("analyze")
    for flag in ("data-root", "run-root", "registry-dir", "registry-sha256", "trace-audit", "trace-audit-sha256", "output"):
        sub.add_argument("--" + flag, required=True)
    sub = commands.add_parser("evaluate")
    for flag in ("spec", "data-root", "source-root", "asset-folder", "registry-dir", "registry-sha256",
                 "released-checkpoint", "direction-prior"):
        sub.add_argument("--" + flag, required=True)
    sub.add_argument("--process", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.repository_root, args.data_root, args.run_root, args.study_registry, args.output_dir,
                         args.exclude_assembly)
    elif args.command == "verify":
        result = verify(args.repository_root, args.data_root, args.run_root, args.study_registry, args.registry_dir,
                        args.exclude_assembly)
    elif args.command == "audit-traces":
        result = audit_traces(args.data_root, args.registry_dir, args.registry_sha256, args.output)
    elif args.command == "analyze":
        result = analyze(args.data_root, args.run_root, args.registry_dir, args.registry_sha256, args.trace_audit,
                         args.trace_audit_sha256, args.output)
    else:
        result = evaluate(args.spec, args.data_root, args.source_root, args.asset_folder, args.registry_dir,
                          args.registry_sha256, args.released_checkpoint, args.direction_prior, args.process)
    print(production._canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
