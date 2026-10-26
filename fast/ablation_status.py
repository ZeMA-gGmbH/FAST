"""Outcome-free component-ablation round status: completeness, runtime, resources, telemetry, failures, projection.

No validation metric, selected epoch, or comparison is read into this report, so a stop decision
after a round can depend only on time, resources, and machine availability.
"""

import argparse
import sys
from collections import OrderedDict

from fast import ablation_common as common
from fast import ablation_runner as runner


SCHEMA_VERSION = "fast.ablation-round-status.v1"


def _attempt_record(root, directory):
    epochs = common.read_jsonl(directory / "epochs.jsonl") if (directory / "epochs.jsonl").exists() else []
    result = common.read_json(directory / "result.json") if (directory / "result.json").exists() else None
    failed = common.read_json(directory / "failed.json") if (directory / "failed.json").exists() else None
    wandb = common.read_json(directory / "wandb.json") if (directory / "wandb.json").exists() else None
    resources = OrderedDict([
        ("epochs_logged", len(epochs)),
        ("mean_epoch_seconds", sum(row["epoch_seconds"] for row in epochs) / len(epochs) if epochs else None),
        ("peak_cuda_allocated_bytes", max((row["peak_cuda_allocated_bytes"] for row in epochs), default=None)),
        ("peak_cuda_reserved_bytes", max((row["peak_cuda_reserved_bytes"] for row in epochs), default=None)),
        ("peak_host_kib", max((row["peak_host_kib"] for row in epochs), default=None)),
    ])
    return OrderedDict([
        ("attempt_root", directory.relative_to(root).as_posix()),
        ("status", "passed" if result and result.get("status") == "passed" else ("failed" if failed else "running_or_incomplete")),
        ("implementation_commit", result["provenance"]["implementation"]["commit"] if result else None),
        ("training_wall_seconds", result["observations"]["training_wall_seconds"] if result else None),
        ("result_sha256", common.sha256_file(directory / "result.json") if result else None),
        ("selected_checkpoint_sha256", result["best_checkpoint_sha256"] if result else None),
        ("epoch_log_sha256", common.sha256_file(directory / "epochs.jsonl") if epochs else None),
        ("resources", resources),
        ("failure", OrderedDict((key, failed[key]) for key in ("exit_code", "wall_seconds")) if failed else None),
        ("wandb", OrderedDict((key, wandb.get(key)) for key in ("name", "mode", "run_id", "url", "status", "logged_epochs", "offline_sync_required")) if wandb else None),
    ])


def round_status(data_root, round_name):
    root = common.run_root(data_root)
    registry = runner.load_registry(root)
    common.require(round_name in runner.SEED_ROUNDS, "unknown round")
    rows = [row for row in registry["matrix"] if row["round"] == round_name]
    jobs = []
    for row in rows:
        attempts = [_attempt_record(root, directory) for directory in runner.attempt_dirs(root, row)]
        jobs.append(OrderedDict([("run_name", row["run_name"]), ("identity_sha256", row["identity_sha256"]),
                                 ("completed", runner.completed(root, row) is not None), ("attempts", attempts)]))
    ledger = common.read_jsonl(root / "attempts.jsonl") if (root / "attempts.jsonl").exists() else []
    ends = {(row["run_name"], row["attempt"]): row for row in ledger if row["event"] == "end"}
    starts = {(row["run_name"], row["attempt"]): row for row in ledger if row["event"] == "start"}
    for job in jobs:
        for attempt in job["attempts"]:
            key = (job["run_name"], int(attempt["attempt_root"].rsplit("-", 1)[1]))
            attempt["started_at"] = starts.get(key, {}).get("started_at")
            attempt["ended_at"] = ends.get(key, {}).get("ended_at")
            attempt["runner_wall_seconds"] = ends.get(key, {}).get("wall_seconds")
    passed = [attempt for job in jobs for attempt in job["attempts"] if attempt["status"] == "passed"]
    epoch_seconds = [attempt["resources"]["mean_epoch_seconds"] for attempt in passed if attempt["resources"]["mean_epoch_seconds"]]
    decision = common.read_json(root / "concurrency-decision.json") if (root / "concurrency-decision.json").exists() else None
    per_job_hours = max(attempt["runner_wall_seconds"] or 0 for attempt in passed) / 3600.0 if passed else None
    return OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("run_id", common.RUN_ID), ("round", round_name), ("reported_at", common.utc_now()),
        ("contains_validation_outcomes", False), ("complete", all(job["completed"] for job in jobs)), ("jobs", jobs),
        ("concurrency_decision", OrderedDict((key, decision[key]) for key in ("allowed_concurrent_jobs", "measurements")) if decision else None),
        ("measured_mean_epoch_seconds", epoch_seconds),
        ("projection_remaining_rounds", OrderedDict([
            ("rounds_remaining", sum(1 for name in runner.SEED_ROUNDS if name > round_name)),
            ("jobs_per_round", 3), ("measured_longest_job_hours_this_round", per_job_hours),
        ])),
        ("test_rows_loaded", 0),
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--round", default="seed-0", choices=runner.SEED_ROUNDS)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    report = round_status(args.data_root, args.round)
    if args.publish:
        common.require(report["complete"], "round is incomplete")
        digest = common.write_json_once(common.run_root(args.data_root) / "{}-status.json".format(args.round), report)
        report = {"published": "{}-status.json".format(args.round), "sha256": digest}
    print(common.canonical_json(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
