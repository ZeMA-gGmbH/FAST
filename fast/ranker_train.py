"""Run and audit deterministic ranker setup, GPU pilots, and training."""

import argparse
import copy
import hashlib
import math
import os
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from importlib import metadata

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch_geometric

from fast import graph_contract, graph_materialize, ranker_models
from plan_sequence import production_sim_labels as production


SETUP_SCHEMA_VERSION = "fast.environment-setup.v1"
CHECKPOINT_SCHEMA_VERSION = "fast.ranker-checkpoint.v1"
RUN_SCHEMA_VERSION = "fast.ranker-training-run.v1"
AUDIT_SCHEMA_VERSION = "fast.ranker-training-audit.v1"
PILOT_SCHEMA_VERSION = "fast.ranker-gpu-pilot.v1"


def _write_json(path, value):
    graph_materialize._write_json(path, value)


def _read_json(path):
    return graph_materialize._read_json(path)


def _canonical_hash(value):
    return hashlib.sha256(production._canonical_json(value).encode("utf-8")).hexdigest()


def configure_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _environment(spec):
    expected = spec["environment"]
    lock_path = Path(__file__).resolve().parents[1] / expected["lock"]
    graph_contract._require(production._sha256_file(lock_path) == expected["lock_sha256"], "ranker lock hash differs")
    actual = {
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_runtime_package": metadata.version("nvidia-cuda-runtime-cu12"),
        "cudnn_package": metadata.version("nvidia-cudnn-cu12"),
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "lock_sha256": expected["lock_sha256"],
        "container": expected["container"],
    }
    graph_contract._require(actual["python"].startswith(expected["python"] + "."), "Python version differs")
    graph_contract._require(actual["torch"].split("+")[0] == expected["torch"], "PyTorch version differs")
    graph_contract._require(actual["torch_geometric"] == expected["torch_geometric"], "PyG version differs")
    graph_contract._require(actual["cuda_runtime"] == expected["cuda_runtime"].rsplit(".", 1)[0], "CUDA runtime differs")
    graph_contract._require(actual["cuda_runtime_package"] == expected["cuda_runtime"], "CUDA runtime package differs")
    graph_contract._require(actual["cudnn_package"] == expected["cudnn"], "cuDNN package differs")
    return actual


def _git_head(root):
    git = Path(root) / ".git"
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[5:]
        path = git / ref
        if path.is_file():
            head = path.read_text(encoding="utf-8").strip()
        else:
            refs = {
                name: commit
                for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines()
                if line and not line.startswith(("#", "^"))
                for commit, name in (line.split(" ", 1),)
            }
            head = refs.get(ref, "")
    graph_contract._require(len(head) == 40 and all(char in "0123456789abcdef" for char in head), "invalid Git HEAD")
    return head


def _repository_identity(require_clean):
    """Commit and cleanliness of this checkout; a job without Git can pass both in the environment."""
    root = Path(__file__).resolve().parents[1]
    commit = os.environ.get("FAST_IMPLEMENTATION_COMMIT") or _git_head(root)
    if "FAST_IMPLEMENTATION_CLEAN" in os.environ:
        dirty = os.environ["FAST_IMPLEMENTATION_CLEAN"] != "1"
    else:
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                                    stdout=subprocess.PIPE, universal_newlines=True).stdout.strip())
    graph_contract._require(commit == _git_head(root), "submitted implementation commit differs")
    if require_clean:
        graph_contract._require(not dirty, "implementation repository is dirty")
    return {"commit": commit, "dirty": dirty}


def _training_matrix():
    return [
        {"process": process, "architecture": architecture, "seed": seed}
        for process, (architecture, seed) in enumerate(
            (architecture, seed)
            for architecture in ranker_models.ARCHITECTURES
            for seed in (0, 1, 2)
        )
    ]


def setup(spec_path, source_root, data_root, graph_root=None, run_root=None):
    configure_determinism(0)
    prepared = graph_materialize._prepare(spec_path, source_root, data_root, graph_root)
    spec = prepared["spec"]
    identity = _repository_identity(require_clean=True)
    environment = _environment(spec)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(prepared["repository_root"]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    graph_contract._require(tests.returncode == 0, "repository test suite failed:\n{}".format(tests.stdout[-4000:]))
    try:
        graph_contract.validate_development_record({"dataset_split": "test"})
    except ValueError as error:
        graph_contract._require("rejects test" in str(error), "development rejection happened after schema access")
    else:
        raise ValueError("development loader accepted test metadata")

    run_root = Path(run_root).resolve() if run_root else prepared["data_root"] / spec["artifacts"]["ranker_training"]
    run_root.mkdir(parents=True, exist_ok=True)
    jobs_path = run_root / "training-jobs.tsv"
    jobs_path.write_text(
        "process\tarchitecture\tseed\n"
        + "".join("{process}\t{architecture}\t{seed}\n".format(**row) for row in _training_matrix()),
        encoding="utf-8",
    )
    report = {
        "schema_version": SETUP_SCHEMA_VERSION,
        "status": "passed",
        "spec_sha256": prepared["spec_sha256"],
        "dataset_manifest_sha256": spec["inputs"]["dataset_manifest_sha256"],
        "feature_build_sha256": spec["inputs"]["feature_build_sha256"],
        "implementation": identity,
        "environment": environment,
        "test_command": "{} -m unittest discover -s tests".format(sys.executable),
        "test_output_tail": tests.stdout[-4000:],
        "training_job_table": jobs_path.name,
        "training_job_table_sha256": production._sha256_file(jobs_path),
        "training_job_count": 15,
        "test_rows_loaded": 0,
    }
    _write_json(run_root / "setup.json", report)
    return report


def _prepare(spec_path, source_root, data_root, graph_root=None, run_root=None, require_clean=True):
    prepared = graph_materialize._prepare(spec_path, source_root, data_root, graph_root)
    audit_path = prepared["output_root"] / "audit.json"
    graph_contract._require(audit_path.is_file(), "materialization audit is missing")
    audit = _read_json(audit_path)
    graph_contract._require(audit.get("status") == "passed" and not audit.get("pilot"), "materialization audit did not pass")
    graph_contract._require(audit.get("spec_sha256") == prepared["spec_sha256"], "materialization spec hash differs")
    graph_contract._require(audit.get("test_rows_loaded") == 0, "materialization audit loaded test rows")
    graph_contract._require(audit.get("normalization"), "training normalization is missing")
    graph_contract._require(all(row.get("dataset_split") in graph_contract.DEVELOPMENT_SPLITS for row in audit["inventory"]), "test metadata rejected")
    prepared.update({
        "audit": audit,
        "audit_path": audit_path,
        "audit_sha256": production._sha256_file(audit_path),
        "run_root": Path(run_root).resolve() if run_root else prepared["data_root"] / prepared["spec"]["artifacts"]["ranker_training"],
        "implementation": _repository_identity(require_clean),
        "environment": _environment(prepared["spec"]),
    })
    return prepared


def _load_assembly(prepared, inventory_row):
    graph_contract._require(inventory_row["dataset_split"] in graph_contract.DEVELOPMENT_SPLITS, "development loader rejects test metadata")
    path = (prepared["output_root"] / inventory_row["path"]).resolve()
    graph_contract._require(production._is_within(path, prepared["output_root"]), "graph shard escapes root")
    graph_contract._require(production._sha256_file(path) == inventory_row["sha256"], "graph shard hash differs")
    rows = graph_materialize._read_jsonl(path)
    graph_contract._require(rows[0].get("record_type") == "header" and rows[-1].get("complete") is True, "graph shard envelope differs")
    records = [graph_contract.validate_development_record(row) for row in rows[1:-1]]
    graph_contract._require(all(row["assembly_id"] == inventory_row["assembly_id"] for row in records), "graph shard assembly differs")
    return records


def _inventory(prepared, split):
    rows = [row for row in prepared["audit"]["inventory"] if row["dataset_split"] == split]
    expected = prepared["spec"]["materialization"]["development"][split + "_assembly_count"]
    graph_contract._require(len(rows) == expected, "{} inventory count differs".format(split))
    return rows


def _assembly_batches(rows, seed, epoch, batch_size=8, shuffle=True):
    order = np.arange(len(rows))
    if shuffle:
        order = np.random.RandomState(seed * 1000003 + epoch).permutation(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start:start + batch_size]]


def _selection_metrics(predictions):
    success = []
    reciprocal_ranks = []
    y_scores, y_targets = [], []
    for row in predictions:
        targets = row["Y"].reshape(-1)
        scores = row["Y_score"].reshape(-1)
        if row["fully_observed"] and targets.any():
            order = np.argsort(-scores, kind="stable")
            success.append(float(targets[order[0]]))
            reciprocal_ranks.append(1.0 / (int(np.flatnonzero(targets[order] == 1)[0]) + 1))
        valid = row["Y_valid"].reshape(-1)
        y_scores.extend(scores[valid].tolist())
        y_targets.extend(targets[valid].tolist())
    return {
        "offline_context_success_at_1": float(np.mean(success)) if success else None,
        "first_feasible_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "Y_average_precision": _average_precision(np.asarray(y_targets), np.asarray(y_scores)),
        "ranking_context_count": len(success),
        "Y_valid_count": len(y_targets),
    }


def _average_precision(targets, scores):
    if not len(targets) or not targets.any():
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_targets = targets[order]
    precision = np.cumsum(sorted_targets) / np.arange(1, len(sorted_targets) + 1)
    return float(precision[sorted_targets == 1].mean())


def predict(model, prepared, rows, device):
    predictions = []
    elapsed = 0.0
    model.eval()
    with torch.no_grad():
        for batch_rows in _assembly_batches(rows, 0, 0, shuffle=False):
            records = [record for row in batch_rows for record in _load_assembly(prepared, row)]
            batch = ranker_models.torch_batch(records, prepared["audit"]["normalization"], device)
            started = time.perf_counter()
            outputs = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            probs = ranker_models.probabilities(outputs)
            for context, record in enumerate(records):
                start, stop = int(batch["candidate_ptr"][context]), int(batch["candidate_ptr"][context + 1])
                target = record["targets"]
                row = {
                    "assembly_id": record["assembly_id"],
                    "tier": record["tier"],
                    "query_id": record["query_id"],
                    "R": target["R"], "R_valid": target["R_valid"],
                    "T": target["T"], "T_valid": target["T_valid"],
                    "Y": target["Y"], "Y_valid": target["Y_valid"],
                    "fully_observed": bool(target["R_valid"].all() and target["T_valid"].all()),
                    "Y_score": probs["Y"][start:stop].detach().cpu().numpy(),
                    "Y_logit": outputs.get("Y", torch.empty(0, device=device))[start:stop].detach().cpu().numpy() if "Y" in outputs else None,
                    "R_score": probs.get("R", torch.empty(0, device=device))[start:stop].detach().cpu().numpy() if "R" in probs else None,
                    "R_logit": outputs.get("R", torch.empty(0, device=device))[start:stop].detach().cpu().numpy() if "R" in outputs else None,
                    "T_score": probs.get("T", torch.empty(0, device=device))[start:stop].detach().cpu().numpy() if "T" in probs else None,
                    "T_logit": outputs.get("T", torch.empty(0, device=device))[start:stop].detach().cpu().numpy() if "T" in outputs else None,
                }
                predictions.append(row)
    return predictions, elapsed


def _provenance(prepared, architecture, seed):
    normalization = prepared["audit"]["normalization"]
    return {
        "spec_sha256": prepared["spec_sha256"],
        "dataset_manifest_sha256": prepared["spec"]["inputs"]["dataset_manifest_sha256"],
        "feature_build_sha256": prepared["spec"]["inputs"]["feature_build_sha256"],
        "materialization_audit_sha256": prepared["audit_sha256"],
        "normalization_sha256": _canonical_hash(normalization),
        "validation_folds_sha256": prepared["audit"]["validation_folds_sha256"],
        "implementation": prepared["implementation"],
        "environment": _environment(prepared["spec"]),
        "architecture": architecture,
        "parameter_count": ranker_models.EXPECTED_PARAMETERS[architecture],
        "seed": seed,
        "data_order": "numpy.RandomState(seed*1000003+epoch), whole assemblies, contexts canonical",
        "test_rows_loaded": 0,
    }


def _validate_provenance(prepared, provenance, architecture, seed, require_cuda=True):
    expected = _provenance(prepared, architecture, seed)
    recorded_environment = provenance.get("environment", {})
    current_environment = expected["environment"]
    static_environment = {
        "python", "torch", "torch_geometric", "cuda_runtime", "cudnn",
        "cuda_runtime_package", "cudnn_package", "lock_sha256", "container",
    }
    for name in static_environment:
        graph_contract._require(recorded_environment.get(name) == current_environment.get(name), "checkpoint environment differs: {}".format(name))
    for name in set(expected) - {"environment"}:
        graph_contract._require(provenance.get(name) == expected[name], "checkpoint provenance differs: {}".format(name))
    graph_contract._require(recorded_environment.get("deterministic_algorithms") is True, "checkpoint deterministic mode is disabled")
    graph_contract._require(recorded_environment.get("cudnn_benchmark") is False, "checkpoint cuDNN benchmark is enabled")
    graph_contract._require(recorded_environment.get("tf32_matmul") is False and recorded_environment.get("tf32_cudnn") is False, "checkpoint TF32 is enabled")
    graph_contract._require(recorded_environment.get("cublas_workspace_config") == ":4096:8", "checkpoint cuBLAS determinism differs")
    if require_cuda:
        graph_contract._require(recorded_environment.get("cuda_available") is True and recorded_environment.get("device") != "cpu", "checkpoint was not trained on CUDA")


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, model, optimizer, provenance, epoch, optimizer_updates, selection, best):
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": provenance,
        "epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "validation_selection": selection,
        "best": best,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(),
    }
    _atomic_torch_save(path, payload)
    sidecar = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint": Path(path).name,
        "checkpoint_sha256": production._sha256_file(path),
        "checkpoint_bytes": Path(path).stat().st_size,
        "provenance": provenance,
        "epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "validation_selection": selection,
    }
    _write_json(str(path) + ".json", sidecar)
    return sidecar


def load_checkpoint(path, model, optimizer, expected_provenance, restore_rng=True):
    path = Path(path)
    sidecar = _read_json(str(path) + ".json")
    graph_contract._require(production._sha256_file(path) == sidecar["checkpoint_sha256"], "checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    graph_contract._require(payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION, "checkpoint schema differs")
    graph_contract._require(payload.get("provenance") == expected_provenance, "checkpoint provenance differs")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        _restore_rng(payload["rng_state"])
    return payload, sidecar


def _loss_counts(audit, direct_y):
    counts = audit["component_assembly_counts"]
    if direct_y:
        return {"total": counts["total"], "rank": counts["rank_direct"], "Y": counts["Y_direct"]}
    return {name: counts[name] for name in ("total", "rank", "R", "T", "Y")}


def train(spec_path, source_root, data_root, graph_root=None, run_root=None, process=None, architecture=None, seed=None, attempt=1, resume=None):
    prepared = _prepare(spec_path, source_root, data_root, graph_root, run_root)
    matrix = _training_matrix()
    if process is not None:
        graph_contract._require(0 <= process < len(matrix), "training process is out of range")
        selected = matrix[process]
        if architecture is not None:
            graph_contract._require(architecture == selected["architecture"], "process architecture differs")
        if seed is not None:
            graph_contract._require(seed == selected["seed"], "process seed differs")
        architecture, seed = selected["architecture"], selected["seed"]
    graph_contract._require(architecture in ranker_models.ARCHITECTURES and seed in (0, 1, 2), "invalid training configuration")
    graph_contract._require(torch.cuda.is_available(), "full ranker training requires CUDA")
    device = torch.device("cuda", 0)
    configure_determinism(seed)
    model = ranker_models.build_model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    provenance = _provenance(prepared, architecture, seed)
    attempt_root = prepared["run_root"] / "training" / architecture / "seed-{}".format(seed) / "attempt-{}".format(attempt)
    graph_contract._require(not (attempt_root / "result.json").exists(), "completed attempt cannot be overwritten")
    attempt_root.mkdir(parents=True, exist_ok=True)
    start_epoch, updates, best, history = 1, 0, None, []
    if resume:
        payload, _ = load_checkpoint(resume, model, optimizer, provenance)
        start_epoch = payload["epoch"] + 1
        updates = payload["optimizer_updates"]
        best = payload["best"]
        source_best = Path(resume).parent / "epoch-{:03d}.pt".format(best["epoch"])
        target_best = attempt_root / "checkpoints" / source_best.name
        target_best.parent.mkdir(parents=True, exist_ok=True)
        if source_best.resolve() != target_best.resolve():
            graph_contract._require(source_best.is_file() and Path(str(source_best) + ".json").is_file(), "resumed best checkpoint is missing")
            shutil.copy2(source_best, target_best)
            shutil.copy2(str(source_best) + ".json", str(target_best) + ".json")

    train_rows, validation_rows = _inventory(prepared, "train"), _inventory(prepared, "validation")
    counts = _loss_counts(prepared["audit"], architecture == ranker_models.DIRECT_Y_ARCHITECTURE)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, prepared["spec"]["training"]["epochs"] + 1):
        model.train()
        epoch_losses = []
        for assembly_rows in _assembly_batches(train_rows, seed, epoch):
            records = [record for row in assembly_rows for record in _load_assembly(prepared, row)]
            batch = ranker_models.torch_batch(records, prepared["audit"]["normalization"], device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, _ = ranker_models.model_loss(batch, outputs, counts, architecture == ranker_models.DIRECT_Y_ARCHITECTURE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
            epoch_losses.append(float(loss.detach()))
        predictions, inference_seconds = predict(model, prepared, validation_rows, device)
        selection = _selection_metrics(predictions)
        selection["epoch"] = epoch
        key = (
            selection["offline_context_success_at_1"],
            selection["first_feasible_mrr"],
            selection["Y_average_precision"],
            -epoch,
        )
        if best is None or tuple(best["key"]) < key:
            best = {"epoch": epoch, "key": list(key), "selection": selection}
        checkpoint_path = attempt_root / "checkpoints" / "epoch-{:03d}.pt".format(epoch)
        checkpoint = save_checkpoint(checkpoint_path, model, optimizer, provenance, epoch, updates, selection, best)
        history.append({
            "epoch": epoch,
            "mean_batch_loss": float(np.mean(epoch_losses)),
            "optimizer_updates": updates,
            "validation_selection": selection,
            "checkpoint": checkpoint_path.relative_to(attempt_root).as_posix(),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "validation_inference_seconds": inference_seconds,
        })

    best_path = attempt_root / "checkpoints" / "epoch-{:03d}.pt".format(best["epoch"])
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "passed",
        "provenance": provenance,
        "attempt": attempt,
        "resumed_from": str(resume) if resume else None,
        "epoch_count": prepared["spec"]["training"]["epochs"],
        "optimizer_update_count": updates,
        "best": best,
        "best_checkpoint": best_path.relative_to(attempt_root).as_posix(),
        "best_checkpoint_sha256": production._sha256_file(best_path),
        "history": history,
        "observations": {
            "training_seconds": elapsed,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
            "peak_host_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "best_validation_inference_seconds": next(
                (row["validation_inference_seconds"] for row in history if row["epoch"] == best["epoch"]), None
            ),
        },
        "test_rows_loaded": 0,
    }
    _write_json(attempt_root / "result.json", report)
    return report


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _pilot_trace(architecture, seed, batch, counts, steps, device, initial=None):
    configure_determinism(seed)
    model = ranker_models.build_model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    if initial is not None:
        model.load_state_dict(copy.deepcopy(initial["model"]))
        optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
        _restore_rng(copy.deepcopy(initial["rng"]))
    else:
        initial = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": copy.deepcopy(_rng_state())}
    trace = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = ranker_models.model_loss(
            batch, model(batch), counts, architecture == ranker_models.DIRECT_Y_ARCHITECTURE
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        trace.append(float(loss.detach()))
    final = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": copy.deepcopy(_rng_state())}
    return trace, _state_digest(model), initial, final


def gpu_pilot(spec_path, source_root, data_root, graph_root=None, run_root=None):
    prepared = _prepare(spec_path, source_root, data_root, graph_root, run_root)
    graph_contract._require(torch.cuda.is_available(), "GPU pilot requires CUDA")
    device = torch.device("cuda", 0)
    train_rows = _inventory(prepared, "train")
    pilot_rows = train_rows[:8]
    records = [record for row in pilot_rows for record in _load_assembly(prepared, row)]
    batch = ranker_models.torch_batch(records, prepared["audit"]["normalization"], device)
    results = {}
    for architecture in ranker_models.ARCHITECTURES:
        counts = _loss_counts(prepared["audit"], architecture == ranker_models.DIRECT_Y_ARCHITECTURE)
        first_trace, first_digest, initial, _ = _pilot_trace(architecture, 0, batch, counts, 2, device)
        second_trace, second_digest, _, _ = _pilot_trace(architecture, 0, batch, counts, 2, device, initial)
        graph_contract._require(first_trace == second_trace and first_digest == second_digest, "deterministic replay differs")
        one_trace, _, initial, interrupted = _pilot_trace(architecture, 0, batch, counts, 1, device)
        resumed_trace, resumed_digest, _, _ = _pilot_trace(architecture, 0, batch, counts, 1, device, interrupted)
        uninterrupted_trace, uninterrupted_digest, _, _ = _pilot_trace(architecture, 0, batch, counts, 2, device, initial)
        graph_contract._require(one_trace + resumed_trace == uninterrupted_trace and resumed_digest == uninterrupted_digest, "resume replay differs")
        results[architecture] = {"loss_trace": first_trace, "parameter_state_sha256": first_digest}

    configure_determinism(0)
    model = ranker_models.build_model("graph_transformer").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    counts = _loss_counts(prepared["audit"], False)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    candidate_count = 0
    for assembly_rows in _assembly_batches(train_rows, 0, 1):
        records = [record for row in assembly_rows for record in _load_assembly(prepared, row)]
        candidate_count += sum(len(row["candidate_index"]) for row in records)
        training_batch = ranker_models.torch_batch(records, prepared["audit"]["normalization"], device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = ranker_models.model_loss(training_batch, model(training_batch), counts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "passed",
        "provenance": _provenance(prepared, "graph_transformer", 0),
        "fixed_training_assemblies": [row["assembly_id"] for row in pilot_rows],
        "determinism": results,
        "resource_probe": {
            "training_assemblies": len(train_rows),
            "candidate_count": candidate_count,
            "seconds": elapsed,
            "candidates_per_second": candidate_count / elapsed,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
            "peak_host_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scratch_bytes": 0,
        },
        "test_rows_loaded": 0,
    }
    output = prepared["run_root"] / "pilots" / "gpu-pilot.json"
    _write_json(output, report)
    return report


def training_audit(spec_path, source_root, data_root, graph_root=None, run_root=None):
    prepared = _prepare(spec_path, source_root, data_root, graph_root, run_root)
    expected = {(row["architecture"], row["seed"]): row for row in _training_matrix()}
    selected, attempts = {}, []
    expected_updates = prepared["spec"]["training"]["epochs"] * int(
        math.ceil(prepared["spec"]["materialization"]["development"]["train_assembly_count"] / 8.0)
    )
    training_root = prepared["run_root"] / "training"
    for result_path in sorted(training_root.glob("*/seed-*/attempt-*/result.json")):
        result = _read_json(result_path)
        provenance = result.get("provenance", {})
        key = (provenance.get("architecture"), provenance.get("seed"))
        graph_contract._require(key in expected, "unexpected training configuration")
        graph_contract._require(result.get("status") == "passed" and result.get("test_rows_loaded") == 0, "training run did not pass")
        graph_contract._require(result.get("epoch_count") == 100, "training epoch count differs")
        graph_contract._require(result.get("optimizer_update_count") == expected_updates, "optimizer update count differs")
        _validate_provenance(prepared, provenance, *key)
        checkpoint = result_path.parent / result["best_checkpoint"]
        graph_contract._require(production._sha256_file(checkpoint) == result["best_checkpoint_sha256"], "best checkpoint hash differs")
        attempts.append({"path": result_path.parent.relative_to(prepared["run_root"]).as_posix(), "attempt": result["attempt"], "key": key})
        graph_contract._require(key not in selected or selected[key]["attempt"] < result["attempt"], "duplicate attempt number")
        selected[key] = {"attempt": result["attempt"], "path": result_path, "result": result}
    graph_contract._require(set(selected) == set(expected), "complete 15-job training matrix is absent")

    inventory = []
    for key, row in sorted(selected.items()):
        result = row["result"]
        inventory.append({
            "architecture": key[0], "seed": key[1], "attempt": row["attempt"],
            "result_path": row["path"].relative_to(prepared["run_root"]).as_posix(),
            "result_sha256": production._sha256_file(row["path"]),
            "best_checkpoint": (row["path"].parent / result["best_checkpoint"]).relative_to(prepared["run_root"]).as_posix(),
            "best_checkpoint_sha256": result["best_checkpoint_sha256"],
            "parameter_count": result["provenance"]["parameter_count"],
            "optimizer_update_count": result["optimizer_update_count"],
            "observations": result["observations"],
        })
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "spec_sha256": prepared["spec_sha256"],
        "materialization_audit_sha256": prepared["audit_sha256"],
        "normalization_sha256": _canonical_hash(prepared["audit"]["normalization"]),
        "validation_folds_sha256": prepared["audit"]["validation_folds_sha256"],
        "successful_job_count": len(inventory),
        "optimizer_update_count_per_job": expected_updates,
        "inventory": inventory,
        "test_rows_loaded": 0,
    }
    _write_json(prepared["run_root"] / "training-audit.json", report)
    return report


def _parser():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(root / "configs" / "ranker_models.json"))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--graph-root")
    parser.add_argument("--run-root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup")
    commands.add_parser("gpu-pilot")
    train_parser = commands.add_parser("train")
    identity = train_parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--process", type=int)
    identity.add_argument("--architecture", choices=ranker_models.ARCHITECTURES)
    train_parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    train_parser.add_argument("--attempt", type=int, default=1)
    train_parser.add_argument("--resume")
    commands.add_parser("audit")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    common = (args.spec, args.source_root, args.data_root, args.graph_root, args.run_root)
    if args.command == "setup":
        result = setup(*common)
    elif args.command == "gpu-pilot":
        result = gpu_pilot(*common)
    elif args.command == "train":
        graph_contract._require(args.process is not None or args.seed is not None, "--seed is required with --architecture")
        result = train(*common, args.process, args.architecture, args.seed, args.attempt, args.resume)
    else:
        result = training_audit(*common)
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
