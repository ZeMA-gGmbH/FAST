"""Separate-process W&B mirror for component-ablation training telemetry (never authoritative).

The trainer never imports W&B. This mirror tails the trainer's local epoch log and logs scalar
metrics only. It disables code, git, requirements, console, and metadata upload, logs no
artifacts, and falls back to offline mode on any online failure. Its failure cannot abort or
alter training; the local JSON, checkpoints, and hashes remain the source of record.
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

from fast import ablation_common as common


PROJECT = "fast-component-ablation-2026"
GROUP = common.RUN_ID
MODES = ("online", "offline", "disabled")
POLL_SECONDS = 10.0


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def scalar_config(config):
    """Only names, hashes, and scalars; no paths to data, labels, or checkpoints."""
    provenance = config["provenance"]
    return OrderedDict([
        ("run_id", provenance["run_id"]), ("variant", config["variant"]), ("seed", config["seed"]),
        ("implementation_commit", provenance["implementation"]["commit"]),
        ("environment_lock_sha256", provenance["environment"]["lock_sha256"]),
        ("torch", provenance["environment"]["torch"]), ("device", provenance["environment"]["device"]),
        ("dataset_manifest_sha256", provenance["dataset_manifest_sha256"]), ("graph_audit_sha256", provenance["graph_audit_sha256"]),
        ("normalization_sha256", provenance["normalization_sha256"]), ("sliced_normalization_sha256", provenance["sliced_normalization_sha256"]),
        ("spec_sha256", provenance["spec_sha256"]), ("action_feature_names", provenance["action_feature_names"]),
        ("input_width", provenance["input_width"]), ("parameter_count", provenance["parameter_count"]),
        ("epochs", config["epochs"]),
    ])


def epoch_metrics(row):
    selection = row["validation_selection"]
    metrics = OrderedDict([
        ("epoch", row["epoch"]), ("loss/mean_batch", row["mean_batch_loss"]),
        ("validation/success_at_1", selection["offline_context_success_at_1"]),
        ("validation/first_feasible_mrr", selection["first_feasible_mrr"]),
        ("validation/R_average_precision", row["R_average_precision"]),
        ("validation/Y_average_precision", selection["Y_average_precision"]),
        ("time/epoch_seconds", row["epoch_seconds"]), ("time/validation_inference_seconds", row["validation_inference_seconds"]),
        ("resource/peak_host_kib", row["peak_host_kib"]), ("resource/peak_cuda_allocated_bytes", row["peak_cuda_allocated_bytes"]),
        ("selection/best_epoch_so_far", row["best_epoch_so_far"]),
    ])
    for name, value in row["loss_components"].items():
        if value is not None:
            metrics["loss/" + name] = value
    return metrics


def _settings(wandb, mode):
    return wandb.Settings(
        mode=mode, save_code=False, disable_code=True, disable_git=True, disable_job_creation=True,
        x_disable_meta=True, x_save_requirements=False, console="off", allow_offline_artifacts=False,
        init_timeout=90, silent=True,
    )


def _init(wandb, mode, name, config, directory):
    kwargs = dict(project=PROJECT, group=GROUP, name=name, job_type="training", config=dict(config), dir=str(directory),
                  save_code=False)
    if mode == "online":
        try:
            return wandb.init(settings=_settings(wandb, "online"), **kwargs), "online", None
        except Exception as error:
            fallback = "{}: {}".format(type(error).__name__, error)[:500]
            return wandb.init(settings=_settings(wandb, "offline"), **kwargs), "offline", fallback
    return wandb.init(settings=_settings(wandb, "offline"), **kwargs), "offline", None


def mirror(attempt_root, run_name, mode, trainer_pid=None, poll_seconds=POLL_SECONDS, max_wait_seconds=None):
    attempt_root = Path(attempt_root)
    common.require(mode in MODES, "telemetry mode differs")
    record = OrderedDict([("project", PROJECT), ("group", GROUP), ("name", run_name), ("requested_mode", mode),
                          ("authoritative", False), ("uploads_disabled", ["code", "git", "requirements", "console", "metadata", "artifacts"])])
    output = attempt_root / "wandb.json"
    if mode == "disabled":
        record.update([("mode", "disabled"), ("run_id", None), ("url", None), ("status", "disabled"), ("logged_epochs", 0)])
        common.write_json_once(output, record)
        return record
    started = time.time()
    config_path = attempt_root / "telemetry-config.json"
    while not config_path.exists():
        if (trainer_pid and not _alive(trainer_pid)) or (max_wait_seconds and time.time() - started > max_wait_seconds):
            record.update([("mode", None), ("run_id", None), ("url", None), ("status", "trainer_never_started"), ("logged_epochs", 0)])
            common.write_json_once(output, record)
            return record
        time.sleep(min(poll_seconds, 2.0))
    config = common.read_json(config_path)
    errors, logged = [], 0
    run, actual_mode = None, None
    try:
        import wandb
        run, actual_mode, fallback = _init(wandb, mode, run_name, scalar_config(config), attempt_root)
        if fallback:
            errors.append({"stage": "online_init", "error": fallback})
    except Exception as error:
        errors.append({"stage": "init", "error": "{}: {}".format(type(error).__name__, error)[:500]})
    epoch_log = attempt_root / "epochs.jsonl"
    seen = 0
    status = "running"
    while True:
        rows = common.read_jsonl(epoch_log) if epoch_log.exists() else []
        for row in rows[seen:]:
            if run is not None:
                try:
                    run.log(dict(epoch_metrics(row)), step=int(row["epoch"]))
                    logged += 1
                except Exception as error:
                    errors.append({"stage": "log", "epoch": row.get("epoch"), "error": str(error)[:300]})
        seen = len(rows)
        result = attempt_root / "result.json"
        if result.exists():
            payload = common.read_json(result)
            if run is not None:
                try:
                    run.summary["selected_epoch"] = payload["best"]["epoch"]
                    run.summary["selected_checkpoint_sha256"] = payload["best_checkpoint_sha256"]
                    run.summary["final_status"] = payload["status"]
                except Exception as error:
                    errors.append({"stage": "summary", "error": str(error)[:300]})
            status = "completed"
            break
        if trainer_pid and not _alive(trainer_pid):
            status = "trainer_exited_without_result"
            break
        if max_wait_seconds and time.time() - started > max_wait_seconds:
            status = "mirror_timeout"
            break
        time.sleep(poll_seconds)
    run_id = url = None
    if run is not None:
        run_id, url = run.id, getattr(run, "url", None)
        try:
            run.finish(exit_code=0 if status == "completed" else 1)
        except Exception as error:
            errors.append({"stage": "finish", "error": str(error)[:300]})
    record.update([("mode", actual_mode), ("run_id", run_id), ("url", url if actual_mode == "online" else None),
                   ("status", status), ("logged_epochs", logged), ("errors", errors),
                   ("offline_sync_required", actual_mode == "offline")])
    common.write_json_once(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=MODES, default="online")
    parser.add_argument("--trainer-pid", type=int)
    args = parser.parse_args(argv)
    print(common.canonical_json(mirror(args.attempt_root, args.run_name, args.mode, args.trainer_pid)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
