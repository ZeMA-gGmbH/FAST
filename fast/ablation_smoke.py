"""Component-ablation D smoke checks: determinism, resume, frozen-initialization parity, telemetry isolation,
resource measurement, and the measured concurrency decision. Smoke runs live outside the registry."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from fast import ranker_models, ranker_train
from fast import ablation_common as common
from fast import ablation_masks as masks
from fast import ablation_runner as runner
from fast import ablation_train as trainer


SCHEMA_VERSION = "fast.ablation-smoke.v1"
VOLATILE_EPOCH_FIELDS = ("epoch_seconds", "training_seconds", "validation_inference_seconds", "peak_host_kib",
                         "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")


def _pilot_trace(spec, variant, seed, batch, counts, steps, device, initial=None):
    ranker_train.configure_determinism(seed)
    model = masks.build_model(spec, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    if initial is not None:
        model.load_state_dict(copy.deepcopy(initial["model"]))
        optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
        ranker_train._restore_rng(copy.deepcopy(initial["rng"]))
    else:
        initial = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
                   "rng": copy.deepcopy(ranker_train._rng_state())}
    trace = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = ranker_models.model_loss(batch, model(batch), counts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        trace.append(float(loss.detach()))
    final = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
             "rng": copy.deepcopy(ranker_train._rng_state())}
    return trace, ranker_train._state_digest(model), initial, final


def pilot_determinism(data, device):
    spec = data.context["spec"]
    rows = data.train_rows[:8]
    records = data.batch_records(rows)
    output = OrderedDict()
    for variant in masks.VARIANTS:
        indices, _ = masks.action_mask(spec, variant)
        batch = masks.masked_torch_batch(records, data.normalization, indices, device)
        first, first_digest, initial, _ = _pilot_trace(spec, variant, 0, batch, data.counts, 2, device)
        second, second_digest, _, _ = _pilot_trace(spec, variant, 0, batch, data.counts, 2, device, initial)
        one, _, initial, interrupted = _pilot_trace(spec, variant, 0, batch, data.counts, 1, device)
        resumed, resumed_digest, _, _ = _pilot_trace(spec, variant, 0, batch, data.counts, 1, device, interrupted)
        straight, straight_digest, _, _ = _pilot_trace(spec, variant, 0, batch, data.counts, 2, device, initial)
        output[variant] = OrderedDict([
            ("fixed_training_assemblies", [row["assembly_id"] for row in rows]), ("loss_trace", first),
            ("parameter_state_sha256", first_digest), ("repeat_identical", first == second and first_digest == second_digest),
            ("resume_identical", one + resumed == straight and resumed_digest == straight_digest),
            ("input_width", len(indices)), ("parameter_count", masks.parameter_count(masks.build_model(spec, variant))),
        ])
    return output


def initialization_parity(spec):
    ranker_train.configure_determinism(0)
    frozen = ranker_models.build_model("graph_transformer").state_dict()
    ranker_train.configure_determinism(0)
    masked = masks.build_model(spec, "D0").state_dict()
    return list(frozen) == list(masked) and all(torch.equal(frozen[name], masked[name]) for name in frozen)


def _scientific_epochs(path):
    return [{key: value for key, value in row.items() if key not in VOLATILE_EPOCH_FIELDS} for row in common.read_jsonl(path)]


def telemetry_equivalence(data_root, data, smoke_root, python):
    disabled_root, enabled_root = smoke_root / "telemetry-disabled", smoke_root / "telemetry-offline"
    disabled = trainer.train(data_root, "D0", 0, disabled_root, epochs=1, data=data, run_name="D0-seed-0-smoke-disabled")
    from fast import ablation_telemetry
    ablation_telemetry.mirror(disabled_root, "D0-seed-0-smoke-disabled", "disabled")
    enabled_root.mkdir(parents=True, exist_ok=False)
    with (enabled_root / "telemetry.log").open("w") as log:
        mirror = subprocess.Popen([python, "-m", "fast.ablation_telemetry", "--attempt-root", str(enabled_root),
                                   "--run-name", "D0-seed-0-smoke-offline", "--mode", "offline", "--trainer-pid", str(os.getpid())],
                                  cwd=str(common.REPOSITORY_ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES="", WANDB_MODE="offline"),
                                  stdout=log, stderr=subprocess.STDOUT)
    enabled = trainer.train(data_root, "D0", 0, enabled_root, epochs=1, data=data, run_name="D0-seed-0-smoke-offline")
    mirror.wait(timeout=600)
    wandb_record = common.read_json(enabled_root / "wandb.json")
    checks = OrderedDict([
        ("epoch_scientific_fields_identical", _scientific_epochs(disabled_root / "epochs.jsonl") == _scientific_epochs(enabled_root / "epochs.jsonl")),
        ("checkpoint_sha256_identical", disabled["best_checkpoint_sha256"] == enabled["best_checkpoint_sha256"]),
        ("final_model_state_identical", disabled["final_model_state_sha256"] == enabled["final_model_state_sha256"]),
        ("mirror_completed_offline", wandb_record.get("status") == "completed" and wandb_record.get("mode") == "offline" and wandb_record.get("logged_epochs") == 1),
    ])
    return checks, disabled, enabled, wandb_record


def smoke(data_root):
    root = common.run_root(data_root)
    output = root / "smoke" / "d0-seed-0-smoke.json"
    common.require(not output.exists(), "smoke report already exists")
    runner.require_gate(root)
    implementation = common.implementation_identity(require_clean=True)
    device = torch.device("cuda", 0)
    data = trainer.TrainingData(data_root)
    spec = data.context["spec"]
    pilot = pilot_determinism(data, device)
    parity = initialization_parity(spec)
    checks, disabled, enabled, wandb_record = telemetry_equivalence(data_root, data, root / "smoke", sys.executable)
    epoch = common.read_jsonl(root / "smoke" / "telemetry-disabled" / "epochs.jsonl")[0]
    epochs = spec["experiments"]["D"]["protocol"]["epochs"]
    report = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("run_id", common.RUN_ID), ("implementation", implementation),
        ("environment", trainer.training_environment()), ("created_at", common.utc_now()),
        ("pilot_determinism", pilot), ("d0_initialization_equals_frozen_graph_transformer", parity),
        ("telemetry_equivalence", checks), ("telemetry_record", wandb_record),
        ("resource_probe", OrderedDict([
            ("variant", "D0"), ("seed", 0), ("full_training_epochs_measured", 2),
            ("epoch_seconds", [disabled["observations"]["mean_epoch_seconds"], enabled["observations"]["mean_epoch_seconds"]]),
            ("training_seconds_one_epoch", epoch["training_seconds"]), ("validation_inference_seconds_one_epoch", epoch["validation_inference_seconds"]),
            ("peak_cuda_allocated_bytes", max(disabled["observations"]["peak_cuda_allocated_bytes"], enabled["observations"]["peak_cuda_allocated_bytes"])),
            ("peak_cuda_reserved_bytes", max(disabled["observations"]["peak_cuda_reserved_bytes"], enabled["observations"]["peak_cuda_reserved_bytes"])),
            ("peak_host_kib_process", enabled["observations"]["peak_host_kib"]),
            ("projected_hours_100_epochs", float(np.mean([disabled["observations"]["mean_epoch_seconds"], enabled["observations"]["mean_epoch_seconds"]])) * epochs / 3600.0),
        ])),
        ("smoke_runs_are_not_registry_attempts", True), ("test_rows_loaded", 0),
    ])
    passed = (all(row["repeat_identical"] and row["resume_identical"] for row in pilot.values()) and parity and all(checks.values()))
    report["status"] = "passed" if passed else "failed"
    common.write_json_once(output, report)
    return {"status": report["status"], "smoke": str(output), "resource_probe": report["resource_probe"], "telemetry_equivalence": checks}


def concurrency(data_root, probe_jobs=("D1", "D2")):
    """After D0-seed-0: apply the frozen resource rule and compare measured concurrent epoch throughput."""
    root = common.run_root(data_root)
    output = root / "concurrency-decision.json"
    common.require(not output.exists(), "concurrency decision already exists")
    registry = runner.load_registry(root)
    d0 = runner.completed(root, next(row for row in registry["matrix"] if row["run_name"] == "D0-seed-0"))
    common.require(d0 is not None, "D0-seed-0 has not completed")
    result = common.read_json(d0 / "result.json")
    observed = result["observations"]
    environment = trainer.training_environment()
    station = runner.workstation()
    vram_limit = 0.80 * min(32e9, environment["device_total_memory_bytes"])
    ram_limit = 0.75 * station["mem_total_kib"]
    cpu_limit = (station["nproc"] - 2) // 4
    resource_allowed = max([count for count in (1, 2, 3)
                            if count * observed["peak_cuda_reserved_bytes"] < vram_limit
                            and count * observed["peak_host_kib"] < ram_limit and count <= cpu_limit] or [1])
    probe_root = root / "smoke" / "concurrency-probe"
    common.require(not probe_root.exists(), "concurrency probe already exists")
    processes, started = [], time.time()
    for variant in probe_jobs[:resource_allowed]:
        attempt = probe_root / variant
        attempt.mkdir(parents=True)
        with (attempt / "stdout.log").open("w") as stdout, (attempt / "stderr.log").open("w") as stderr:
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "fast.ablation_train", "--data-root", str(data_root), "--variant", variant, "--seed", "0",
                 "--attempt-root", str(attempt), "--run-name", variant + "-seed-0-concurrency-probe", "--epochs", "1"],
                cwd=str(common.REPOSITORY_ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), stdout=stdout, stderr=stderr))
    codes = [process.wait() for process in processes]
    common.require(all(code == 0 for code in codes), "concurrency probe failed")
    probes = [common.read_jsonl(probe_root / variant / "epochs.jsonl")[0] for variant in probe_jobs[:resource_allowed]]
    single_epoch = observed["mean_epoch_seconds"]
    concurrent_epoch = max(row["epoch_seconds"] for row in probes)
    throughput_gain = (len(probes) / concurrent_epoch) / (1.0 / single_epoch)
    allowed = len(probes) if len(probes) > 1 and throughput_gain > 1.05 else 1
    decision = OrderedDict([
        ("schema_version", SCHEMA_VERSION), ("run_id", common.RUN_ID), ("decided_at", common.utc_now()),
        ("rule", registry["concurrency_rule"]),
        ("measurements", OrderedDict([
            ("d0_seed0_peak_cuda_reserved_bytes", observed["peak_cuda_reserved_bytes"]), ("d0_seed0_peak_host_kib", observed["peak_host_kib"]),
            ("d0_seed0_mean_epoch_seconds", single_epoch), ("vram_limit_bytes", vram_limit), ("host_ram_limit_kib", ram_limit),
            ("wsl_visible_logical_processors", station["nproc"]), ("max_jobs_by_cpu", cpu_limit),
            ("probe_jobs", list(probe_jobs[:resource_allowed])), ("probe_epoch_seconds", [row["epoch_seconds"] for row in probes]),
            ("probe_peak_cuda_reserved_bytes", [row["peak_cuda_reserved_bytes"] for row in probes]),
            ("probe_peak_host_kib", [row["peak_host_kib"] for row in probes]),
            ("aggregate_throughput_gain_vs_single", throughput_gain),
        ])),
        ("resource_allowed_concurrent_jobs", resource_allowed), ("allowed_concurrent_jobs", allowed),
        ("probe_runs_are_not_registry_attempts", True), ("test_rows_loaded", 0),
    ])
    common.write_json_once(output, decision)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("command", choices=("smoke", "concurrency"))
    args = parser.parse_args(argv)
    result = smoke(args.data_root) if args.command == "smoke" else concurrency(args.data_root)
    print(common.canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
