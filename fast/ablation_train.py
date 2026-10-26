"""Component-ablation same-runtime D0/D1/D2 retraining on the RTX 5090 (deterministic, validation-only).

The loop mirrors ranker_train.train exactly: seeding, deterministic flags, model construction
order, AdamW, clipping, whole-assembly batches in numpy.RandomState(seed*1000003+epoch) order,
the frozen factorized loss, per-epoch validation, and the (Success@1, MRR, Y AP, -epoch)
checkpoint-selection key. Only the action-feature mask and the runtime differ. Shards are
hash-checked once and cached in memory; this changes speed, not arithmetic. The process never
imports W&B: it appends an authoritative local epoch log that a separate mirror process may tail.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import argparse
import hashlib
import resource
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from fast import graph_contract, ranker_models, ranker_train
from fast import ablation_closure as closure
from fast import ablation_common as common
from fast import ablation_masks as masks
from fast import ablation_offline as offline


RUN_SCHEMA_VERSION = "fast.ablation-training-run.v1"
CHECKPOINT_SCHEMA_VERSION = "fast.ablation-checkpoint.v1"
EPOCH_SCHEMA_VERSION = "fast.ablation-epoch.v1"
LOCK_PATH = "environment/cu128/uv.lock"
STATIC_ENVIRONMENT = ("python", "torch", "torch_cuda", "cudnn", "torch_geometric", "numpy", "cuda_runtime_package",
                      "cudnn_package", "device", "capability", "deterministic_algorithms", "cudnn_benchmark",
                      "cudnn_deterministic", "tf32_matmul", "tf32_cudnn", "cublas_workspace_config", "lock", "lock_sha256")


def _driver_version():
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, timeout=30).stdout
        return output.strip().splitlines()[0]
    except Exception:
        return None


def training_environment():
    import torch_geometric
    common.require(torch.cuda.is_available(), "component-ablation retraining requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    return OrderedDict([
        ("python", "{}.{}.{}".format(*sys.version_info[:3])), ("torch", torch.__version__), ("torch_cuda", torch.version.cuda),
        ("cudnn", torch.backends.cudnn.version()), ("torch_geometric", torch_geometric.__version__), ("numpy", np.__version__),
        ("cuda_runtime_package", metadata.version("nvidia-cuda-runtime-cu12")), ("cudnn_package", metadata.version("nvidia-cudnn-cu12")),
        ("device", torch.cuda.get_device_name(0)), ("capability", list(torch.cuda.get_device_capability(0))),
        ("device_total_memory_bytes", int(properties.total_memory)), ("driver_version", _driver_version()),
        ("arch_list", torch.cuda.get_arch_list()),
        ("deterministic_algorithms", torch.are_deterministic_algorithms_enabled()), ("cudnn_benchmark", torch.backends.cudnn.benchmark),
        ("cudnn_deterministic", torch.backends.cudnn.deterministic), ("tf32_matmul", torch.backends.cuda.matmul.allow_tf32),
        ("tf32_cudnn", torch.backends.cudnn.allow_tf32), ("cublas_workspace_config", os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
        ("omp_num_threads", os.environ.get("OMP_NUM_THREADS")), ("torch_threads", torch.get_num_threads()),
        ("lock", LOCK_PATH), ("lock_sha256", common.sha256_file(common.REPOSITORY_ROOT / LOCK_PATH)),
    ])


def _require_deterministic(environment):
    common.require(environment["deterministic_algorithms"] is True and environment["cudnn_benchmark"] is False
                   and environment["cudnn_deterministic"] is True and environment["tf32_matmul"] is False
                   and environment["tf32_cudnn"] is False and environment["cublas_workspace_config"] == ":4096:8",
                   "deterministic training settings differ")
    common.require(list(environment["capability"]) == [12, 0] and "sm_120" in environment["arch_list"], "RTX 5090 sm_120 runtime differs")


class TrainingData:
    """Hash-checked, validated development records cached once per process."""

    def __init__(self, data_root):
        self.context = offline.load_context(data_root)
        audit = self.context["audit"]
        self.normalization = audit["normalization"]
        self.train_rows = [row for row in audit["inventory"] if row["dataset_split"] == "train"]
        self.validation_rows = self.context["validation_rows"]
        population = self.context["spec"]["population"]
        common.require(len(self.train_rows) == population["train_assembly_count"], "train inventory differs")
        self.records = {}
        for row in self.train_rows + self.validation_rows:
            records = ranker_train._load_assembly(self.context["prepared"], row)
            common.require(all(record["dataset_split"] in graph_contract.DEVELOPMENT_SPLITS for record in records), "test row loaded")
            self.records[row["assembly_id"]] = records
        self.test_rows_loaded = 0
        self.counts = ranker_train._loss_counts(audit, False)

    def batch_records(self, rows):
        return [record for row in rows for record in self.records[row["assembly_id"]]]


def provenance(data, variant, seed, indices, names, environment, implementation):
    context, spec = data.context, data.context["spec"]
    return OrderedDict([
        ("run_id", common.RUN_ID), ("spec_sha256", context["spec_sha256"]),
        ("ranker_spec_sha256", common.sha256_file(offline.RANKER_SPEC_PATH)),
        ("dataset_manifest_sha256", closure.FROZEN_SHA256["datasets/fast-labels-v1/manifest.json"]),
        ("feature_build_sha256", closure.FROZEN_SHA256["features/scalar-features-v3/manifest.json"]),
        ("graph_audit_sha256", context["audit_sha256"]),
        ("normalization_sha256", ranker_train._canonical_hash(data.normalization)),
        ("sliced_normalization_sha256", ranker_train._canonical_hash(masks.sliced_normalization(data.normalization, indices))),
        ("validation_folds_sha256", context["audit"]["validation_folds_sha256"]),
        ("transfer_manifest_sha256", context["manifest_sha256"]), ("transfer_verification_sha256", context["verification_sha256"]),
        ("implementation", OrderedDict((key, implementation[key]) for key in ("commit", "clean"))),
        ("environment", OrderedDict((key, environment[key]) for key in STATIC_ENVIRONMENT)),
        ("architecture", "graph_transformer"), ("variant", variant), ("seed", int(seed)),
        ("action_feature_names", list(names)), ("action_feature_indices", [int(value) for value in indices]),
        ("input_width", len(indices)), ("parameter_count", spec["experiments"]["D"]["variants"][variant]["parameter_count"]),
        ("protocol", spec["experiments"]["D"]["protocol"]), ("epochs", spec["experiments"]["D"]["protocol"]["epochs"]),
        ("test_rows_loaded", 0),
    ])


def predict(model, data, rows, indices, device):
    """ranker_train.predict with the masked batch and cached records; identical row structure."""
    predictions, elapsed = [], 0.0
    model.eval()
    with torch.no_grad():
        for batch_rows in ranker_train._assembly_batches(rows, 0, 0, shuffle=False):
            records = data.batch_records(batch_rows)
            batch = masks.masked_torch_batch(records, data.normalization, indices, device)
            started = time.perf_counter()
            outputs = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            probabilities = ranker_models.probabilities(outputs)
            for context, record in enumerate(records):
                start, stop = int(batch["candidate_ptr"][context]), int(batch["candidate_ptr"][context + 1])
                target = record["targets"]
                predictions.append({
                    "assembly_id": record["assembly_id"], "tier": record["tier"], "query_id": record["query_id"],
                    "dataset_split": record["dataset_split"],
                    "part_ids": [record["part_ids"][index] for index in record["candidate_index"]],
                    "R": target["R"], "R_valid": target["R_valid"], "T": target["T"], "T_valid": target["T_valid"],
                    "Y": target["Y"], "Y_valid": target["Y_valid"],
                    "fully_observed": bool(target["R_valid"].all() and target["T_valid"].all()),
                    "Y_score": probabilities["Y"][start:stop].detach().cpu().numpy(), "Y_logit": None,
                    "R_score": probabilities["R"][start:stop].detach().cpu().numpy(),
                    "R_logit": outputs["R"][start:stop].detach().cpu().numpy(),
                    "T_score": probabilities["T"][start:stop].detach().cpu().numpy(),
                    "T_logit": outputs["T"][start:stop].detach().cpu().numpy(),
                })
    return predictions, elapsed


def r_average_precision(predictions):
    targets = np.concatenate([row["R"][row["R_valid"]] for row in predictions])
    scores = np.concatenate([row["R_score"][row["R_valid"]] for row in predictions])
    return ranker_train._average_precision(targets, scores)


def state_digest(model):
    return ranker_train._state_digest(model)


def save_checkpoint(path, model, optimizer, record_provenance, epoch, updates, selection, best):
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION, "provenance": record_provenance, "epoch": epoch,
        "optimizer_updates": updates, "validation_selection": selection, "best": best,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "rng_state": ranker_train._rng_state(),
    }
    ranker_train._atomic_torch_save(path, payload)
    sidecar = OrderedDict([
        ("schema_version", CHECKPOINT_SCHEMA_VERSION), ("checkpoint", Path(path).name),
        ("checkpoint_sha256", common.sha256_file(path)), ("checkpoint_bytes", Path(path).stat().st_size),
        ("epoch", epoch), ("optimizer_updates", updates), ("validation_selection", selection),
        ("model_state_sha256", state_digest(model)),
    ])
    common.write_json_once(str(path) + ".json", sidecar)
    return sidecar


def load_checkpoint(path, model, optimizer, expected_provenance):
    sidecar = common.read_json(str(path) + ".json")
    common.require(common.sha256_file(path) == sidecar["checkpoint_sha256"], "checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    common.require(payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION, "checkpoint schema differs")
    common.require(common.canonical_json(payload["provenance"]) == common.canonical_json(expected_provenance),
                   "resume checkpoint identity differs from this run")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        ranker_train._restore_rng(payload["rng_state"])
    return payload, sidecar


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(common.canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _peak_host_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def train(data_root, variant, seed, attempt_root, *, epochs=None, resume=None, require_clean=True, data=None, run_name=None):
    """Train one D identity into attempt_root; epochs < 100 is only for smoke runs outside the registry."""
    attempt_root = Path(attempt_root)
    common.require(not (attempt_root / "result.json").exists(), "completed attempt cannot be overwritten")
    implementation = common.implementation_identity(require_clean=require_clean)
    ranker_train.configure_determinism(seed)
    environment = training_environment()
    _require_deterministic(environment)
    data = data or TrainingData(data_root)
    spec = data.context["spec"]
    indices, names = masks.action_mask(spec, variant)
    record_provenance = provenance(data, variant, seed, indices, names, environment, implementation)
    total_epochs = spec["experiments"]["D"]["protocol"]["epochs"] if epochs is None else int(epochs)
    common.require(1 <= total_epochs <= spec["experiments"]["D"]["protocol"]["epochs"], "epoch count differs")
    device = torch.device("cuda", 0)
    ranker_train.configure_determinism(seed)
    model = masks.build_model(spec, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    attempt_root.mkdir(parents=True, exist_ok=True)
    epoch_log = attempt_root / "epochs.jsonl"
    start_epoch, updates, best, history = 1, 0, None, []
    if resume:
        resume = Path(resume)
        payload, _ = load_checkpoint(resume, model, optimizer, record_provenance)
        start_epoch, updates, best = payload["epoch"] + 1, payload["optimizer_updates"], payload["best"]
        previous = [row for row in common.read_jsonl(resume.parent.parent / "epochs.jsonl") if row["epoch"] <= payload["epoch"]]
        common.require([row["epoch"] for row in previous] == list(range(1, payload["epoch"] + 1)), "resumed epoch history differs")
        source_best = resume.parent / "epoch-{:03d}.pt".format(best["epoch"])
        target_best = attempt_root / "checkpoints" / source_best.name
        target_best.parent.mkdir(parents=True, exist_ok=True)
        for source, target in ((source_best, target_best), (Path(str(source_best) + ".json"), Path(str(target_best) + ".json"))):
            common.require(source.is_file() and not target.exists(), "resumed best checkpoint cannot be staged")
            shutil.copy2(source, target)
        for row in previous:
            _append_jsonl(epoch_log, dict(row, resumed_from_attempt=resume.parent.parent.name))
        history = previous
    else:
        common.require(not epoch_log.exists(), "fresh attempt already has an epoch log")
    common.write_json_once(attempt_root / "telemetry-config.json", OrderedDict([
        ("run_name", run_name), ("variant", variant), ("seed", seed), ("provenance", record_provenance),
        ("environment_full", environment), ("epochs", total_epochs), ("start_epoch", start_epoch),
    ]))

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses, components = [], {name: [] for name in ("rank", "R", "T", "Y")}
        for assembly_rows in ranker_train._assembly_batches(data.train_rows, seed, epoch):
            batch = masks.masked_torch_batch(data.batch_records(assembly_rows), data.normalization, indices, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, details = ranker_models.model_loss(batch, outputs, data.counts, False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
            for name, value in details["components"].items():
                if value is not None:
                    components[name].append(value)
        training_seconds = time.perf_counter() - epoch_started
        predictions, inference_seconds = predict(model, data, data.validation_rows, indices, device)
        selection = ranker_train._selection_metrics(predictions)
        selection["epoch"] = epoch
        key = (selection["offline_context_success_at_1"], selection["first_feasible_mrr"], selection["Y_average_precision"], -epoch)
        if best is None or tuple(best["key"]) < key:
            best = {"epoch": epoch, "key": list(key), "selection": selection}
        checkpoint_path = attempt_root / "checkpoints" / "epoch-{:03d}.pt".format(epoch)
        sidecar = save_checkpoint(checkpoint_path, model, optimizer, record_provenance, epoch, updates, selection, best)
        row = OrderedDict([
            ("schema_version", EPOCH_SCHEMA_VERSION), ("epoch", epoch), ("optimizer_updates", updates),
            ("mean_batch_loss", float(np.mean(losses))),
            ("loss_components", OrderedDict((name, float(np.mean(values)) if values else None) for name, values in components.items())),
            ("validation_selection", selection), ("R_average_precision", r_average_precision(predictions)),
            ("best_epoch_so_far", best["epoch"]), ("checkpoint", checkpoint_path.relative_to(attempt_root).as_posix()),
            ("checkpoint_sha256", sidecar["checkpoint_sha256"]), ("model_state_sha256", sidecar["model_state_sha256"]),
            ("epoch_seconds", time.perf_counter() - epoch_started), ("training_seconds", training_seconds),
            ("validation_inference_seconds", inference_seconds), ("peak_host_kib", _peak_host_kib()),
            ("peak_cuda_allocated_bytes", int(torch.cuda.max_memory_allocated(device))),
            ("peak_cuda_reserved_bytes", int(torch.cuda.max_memory_reserved(device))),
            ("test_rows_loaded", 0),
        ])
        _append_jsonl(epoch_log, row)
        history.append(row)

    best_path = attempt_root / "checkpoints" / "epoch-{:03d}.pt".format(best["epoch"])
    report = OrderedDict([
        ("schema_version", RUN_SCHEMA_VERSION), ("status", "passed"), ("run_name", run_name), ("provenance", record_provenance),
        ("environment_full", environment), ("epoch_count", total_epochs), ("optimizer_update_count", updates),
        ("resumed_from", str(resume) if resume else None), ("best", best),
        ("best_checkpoint", best_path.relative_to(attempt_root).as_posix()), ("best_checkpoint_sha256", common.sha256_file(best_path)),
        ("epoch_log_sha256", common.sha256_file(epoch_log)),
        ("observations", OrderedDict([
            ("training_wall_seconds", time.perf_counter() - started),
            ("mean_epoch_seconds", float(np.mean([row["epoch_seconds"] for row in history if row["epoch"] >= start_epoch]))),
            ("peak_cuda_allocated_bytes", int(torch.cuda.max_memory_allocated(device))),
            ("peak_cuda_reserved_bytes", int(torch.cuda.max_memory_reserved(device))), ("peak_host_kib", _peak_host_kib()),
        ])),
        ("final_model_state_sha256", state_digest(model)), ("test_rows_loaded", 0),
    ])
    common.write_json_once(attempt_root / "result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--variant", required=True, choices=masks.VARIANTS)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    report = train(args.data_root, args.variant, args.seed, args.attempt_root, epochs=args.epochs, resume=args.resume, run_name=args.run_name)
    print(common.canonical_json({"status": report["status"], "best_checkpoint_sha256": report["best_checkpoint_sha256"],
                                 "epoch_count": report["epoch_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
