"""Train and audit the two PointNet development models."""

import argparse
import copy
import math
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fast import graph_contract, ranker_models, ranker_train, pointnet_models, pointnet_points
from plan_sequence import production_sim_labels as production


ARCHITECTURES = ("pointnet_graph_transformer", "cebulla_gcn")
EXPECTED_PARAMETERS = {
    "pointnet_graph_transformer": pointnet_models.EXPECTED_PARAMETERS,
    "cebulla_gcn": 72577,
}


def _write_json(path, value):
    ranker_train._write_json(path, value)


def _read_json(path):
    return ranker_train._read_json(path)


def _matrix():
    return [
        {"process": process, "architecture": architecture, "seed": seed}
        for process, (architecture, seed) in enumerate(
            (architecture, seed) for architecture in ARCHITECTURES for seed in (0, 1, 2)
        )
    ]


def _pilot_path(prepared, implementation=None):
    implementation = prepared["implementation"] if implementation is None else implementation
    return prepared["run_root"] / "pilots" / "gpu-pilot-{}.json".format(implementation["commit"])


def _prepare(spec_path, source_root, data_root, graph_root=None, point_root=None, run_root=None, require_clean=True):
    ranker_train.configure_determinism(0)
    point_prepared = pointnet_points._prepare(spec_path, source_root, data_root, point_root)
    spec = point_prepared["spec"]
    ranker_spec = point_prepared["repository_root"] / spec["inputs"]["ranker_spec"]["path"]
    graph = ranker_train._prepare(ranker_spec, source_root, data_root, graph_root, None, require_clean=require_clean)
    graph_contract._require(graph["audit_sha256"] == spec["inputs"]["ranker_graph_audit_sha256"], "ranker graph audit differs")
    point_audit_path = point_prepared["output_root"] / "audit.json"
    graph_contract._require(point_audit_path.is_file(), "point-cache audit is missing")
    point_audit = _read_json(point_audit_path)
    graph_contract._require(
        point_audit.get("status") == "passed"
        and not point_audit.get("pilot")
        and point_audit.get("test_rows_loaded") == 0,
        "point-cache audit did not pass",
    )
    point_inventory = {
        (row["source_split"], row["assembly_id"]): row for row in point_audit["inventory"]
    }
    graph_contract._require(len(point_inventory) == point_audit["assembly_count"], "point inventory is duplicated")
    run_root = Path(run_root).resolve() if run_root else point_prepared["data_root"] / spec["artifacts"]["pointnet_runs"]
    return {
        **graph,
        "ranker_spec": graph["spec"], "ranker_spec_sha256": graph["spec_sha256"],
        "spec": spec, "spec_path": Path(spec_path).resolve(), "spec_sha256": point_prepared["spec_sha256"],
        "point_root": point_prepared["output_root"], "point_audit": point_audit,
        "point_audit_path": point_audit_path, "point_audit_sha256": production._sha256_file(point_audit_path),
        "point_inventory": point_inventory, "run_root": run_root,
    }


def _model(architecture):
    if architecture == "pointnet_graph_transformer":
        model = pointnet_models.build_pointnet_model()
    else:
        graph_contract._require(architecture == "cebulla_gcn", "unknown PointNet architecture")
        model = pointnet_models.CebullaGCN()
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    graph_contract._require(count == EXPECTED_PARAMETERS[architecture], "{} parameter count differs".format(architecture))
    return model


def _inventory(prepared, split):
    rows = [row for row in prepared["audit"]["inventory"] if row["dataset_split"] == split]
    graph_contract._require(rows and all(row["source_split"] == "training_assembly" for row in rows), "development inventory differs")
    return rows


def _load_points(prepared, rows):
    result = {}
    for row in rows:
        key = (row["source_split"], row["assembly_id"])
        if key in result:
            continue
        item = prepared["point_inventory"].get(key)
        graph_contract._require(item is not None and item["dataset_split"] == row["dataset_split"], "point inventory row is absent")
        tensor_path = prepared["point_root"] / item["tensor_path"]
        sidecar_path = prepared["point_root"] / item["sidecar_path"]
        graph_contract._require(
            production._sha256_file(tensor_path) == item["tensor_sha256"]
            and production._sha256_file(sidecar_path) == item["sidecar_sha256"],
            "point-cache shard hash differs",
        )
        sidecar = _read_json(sidecar_path)
        tensor = np.load(tensor_path, allow_pickle=False)
        graph_contract._require(tensor.shape == (len(sidecar["part_ids"]), 1000, 3), "point-cache shard shape differs")
        result[key] = (sidecar["part_ids"], tensor)
    return result


def _batch(prepared, records, rows, architecture, device):
    if architecture == "pointnet_graph_transformer":
        return pointnet_models.torch_batch(records, prepared["audit"]["normalization"], _load_points(prepared, rows), device)
    return ranker_models.torch_batch(records, prepared["audit"]["normalization"], device)


def _pointnet_predictions(model, prepared, rows, device):
    predictions, elapsed = [], 0.0
    model.eval()
    with torch.no_grad():
        for batch_rows in ranker_train._assembly_batches(rows, 0, 0, shuffle=False):
            records = [record for row in batch_rows for record in ranker_train._load_assembly(prepared, row)]
            batch = _batch(prepared, records, batch_rows, "pointnet_graph_transformer", device)
            started = time.perf_counter()
            outputs = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            probabilities = ranker_models.probabilities(outputs)
            for context, record in enumerate(records):
                start, stop = int(batch["candidate_ptr"][context]), int(batch["candidate_ptr"][context + 1])
                targets = record["targets"]
                predictions.append({
                    "assembly_id": record["assembly_id"], "tier": record["tier"], "query_id": record["query_id"],
                    "R": targets["R"], "R_valid": targets["R_valid"], "T": targets["T"], "T_valid": targets["T_valid"],
                    "Y": targets["Y"], "Y_valid": targets["Y_valid"],
                    "fully_observed": bool(targets["R_valid"].all() and targets["T_valid"].all()),
                    "Y_score": probabilities["Y"][start:stop].cpu().numpy(),
                    "R_score": probabilities["R"][start:stop].cpu().numpy(), "R_logit": outputs["R"][start:stop].cpu().numpy(),
                    "T_score": probabilities["T"][start:stop].cpu().numpy(), "T_logit": outputs["T"][start:stop].cpu().numpy(),
                    "Y_logit": None,
                })
    return predictions, elapsed


def _cebulla_predictions(model, prepared, rows, device):
    values, elapsed = [], 0.0
    model.eval()
    with torch.no_grad():
        for batch_rows in ranker_train._assembly_batches(rows, 0, 0, shuffle=False):
            records = [record for row in batch_rows for record in ranker_train._load_assembly(prepared, row)]
            batch = _batch(prepared, records, batch_rows, "cebulla_gcn", device)
            started = time.perf_counter()
            logits = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            targets, valid = pointnet_models.cebulla_targets(batch)
            for context, record in enumerate(records):
                start, stop = int(batch["node_ptr"][context]), int(batch["node_ptr"][context + 1])
                candidate = batch["candidate_index"][int(batch["candidate_ptr"][context]):int(batch["candidate_ptr"][context + 1])]
                values.append({
                    "assembly_id": record["assembly_id"], "tier": record["tier"], "query_id": record["query_id"],
                    "score": logits[candidate].cpu().numpy(), "target": targets[candidate].cpu().numpy(),
                    "valid": valid[candidate].cpu().numpy(), "node_range": [start, stop],
                })
    return values, elapsed


def _cebulla_selection(predictions):
    success, targets, scores = [], [], []
    for row in predictions:
        if row["valid"].all() and row["target"].any():
            success.append(float(row["target"][np.argmax(row["score"])]))
        targets.extend(row["target"][row["valid"]].tolist())
        scores.extend(row["score"][row["valid"]].tolist())
    return {
        "validation_part_success_at_1": float(np.mean(success)) if success else None,
        "validation_average_precision": ranker_train._average_precision(np.asarray(targets), np.asarray(scores)),
        "ranking_context_count": len(success), "valid_part_count": len(targets),
    }


def _provenance(prepared, architecture, seed, implementation=None):
    return {
        "spec_sha256": prepared["spec_sha256"], "ranker_spec_sha256": prepared["ranker_spec_sha256"],
        "ranker_graph_audit_sha256": prepared["audit_sha256"], "point_audit_sha256": prepared["point_audit_sha256"],
        "normalization_sha256": ranker_train._canonical_hash(prepared["audit"]["normalization"]),
        "validation_folds_sha256": prepared["audit"]["validation_folds_sha256"],
        "implementation": prepared["implementation"] if implementation is None else implementation,
        "environment": prepared["environment"],
        "architecture": architecture, "parameter_count": EXPECTED_PARAMETERS[architecture], "seed": seed,
        "data_order": "numpy.RandomState(seed*1000003+epoch), whole assemblies, contexts canonical", "test_rows_loaded": 0,
    }


def _training_implementation(prepared):
    setup = _read_json(prepared["run_root"] / "setup.json")
    implementation = setup.get("implementation", {})
    table = prepared["run_root"] / "training-jobs.tsv"
    commit = implementation.get("commit", "")
    graph_contract._require(
        setup.get("status") == "passed"
        and setup.get("spec_sha256") == prepared["spec_sha256"]
        and setup.get("ranker_graph_audit_sha256") == prepared["audit_sha256"]
        and setup.get("point_audit_sha256") == prepared["point_audit_sha256"]
        and setup.get("training_job_count") == len(_matrix())
        and setup.get("training_job_table") == table.name
        and table.is_file()
        and production._sha256_file(table) == setup.get("training_job_table_sha256")
        and setup.get("test_rows_loaded") == 0
        and implementation.get("dirty") is False
        and len(commit) == 40
        and all(char in "0123456789abcdef" for char in commit),
        "PointNet training setup differs",
    )
    return implementation


def _validate_gpu_provenance(prepared, provenance, architecture, seed, implementation=None):
    expected = _provenance(prepared, architecture, seed, implementation)
    graph_contract._require(isinstance(provenance, dict), "GPU artifact provenance is absent")
    recorded_environment = provenance.get("environment", {})
    expected_environment = expected["environment"]
    graph_contract._require(
        {name: value for name, value in provenance.items() if name != "environment"}
        == {name: value for name, value in expected.items() if name != "environment"},
        "GPU artifact provenance differs",
    )
    graph_contract._require(
        isinstance(recorded_environment, dict)
        and set(recorded_environment) == set(expected_environment)
        and all(
            recorded_environment[name] == expected_environment[name]
            for name in set(expected_environment) - {"cuda_available", "device"}
        )
        and recorded_environment.get("cuda_available") is True
        and recorded_environment.get("device") == prepared["spec"]["resources"]["gpu"]["model"],
        "GPU artifact environment differs",
    )


def setup(spec_path, source_root, data_root, graph_root=None, point_root=None, run_root=None):
    prepared = _prepare(spec_path, source_root, data_root, graph_root, point_root, run_root)
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=str(prepared["repository_root"]), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    graph_contract._require(tests.returncode == 0, "repository test suite failed:\n{}".format(tests.stdout[-4000:]))
    prepared["run_root"].mkdir(parents=True, exist_ok=True)
    jobs = prepared["run_root"] / "training-jobs.tsv"
    jobs.write_text("process\tarchitecture\tseed\n" + "".join("{process}\t{architecture}\t{seed}\n".format(**row) for row in _matrix()), encoding="utf-8")
    report = {
        "schema_version": "fast.pointnet-setup.v1", "status": "passed", "spec_sha256": prepared["spec_sha256"],
        "implementation": prepared["implementation"], "environment": prepared["environment"],
        "ranker_graph_audit_sha256": prepared["audit_sha256"], "point_audit_sha256": prepared["point_audit_sha256"],
        "training_job_table": jobs.name, "training_job_table_sha256": production._sha256_file(jobs),
        "training_job_count": len(_matrix()), "test_rows_loaded": 0,
    }
    _write_json(prepared["run_root"] / "setup.json", report)
    return report


def train(spec_path, source_root, data_root, graph_root=None, point_root=None, run_root=None, process=None, architecture=None, seed=None, attempt=1, resume=None):
    prepared = _prepare(spec_path, source_root, data_root, graph_root, point_root, run_root)
    if process is not None:
        graph_contract._require(0 <= process < len(_matrix()), "training process is out of range")
        selected = _matrix()[process]
        architecture, seed = selected["architecture"], selected["seed"]
    graph_contract._require(architecture in ARCHITECTURES and seed in (0, 1, 2), "invalid training configuration")
    pilot = _read_json(_pilot_path(prepared))
    _validate_gpu_provenance(prepared, pilot.get("provenance"), "pointnet_graph_transformer", 0)
    attempt_root = prepared["run_root"] / "training" / architecture / "seed-{}".format(seed) / "attempt-{}".format(attempt)
    graph_contract._require(not (attempt_root / "result.json").exists(), "completed attempt cannot be overwritten")
    if architecture == "pointnet_graph_transformer" and pilot.get("eligible") is not True:
        attempt_root.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": "fast.pointnet-training-run.v1", "status": "ineligible",
            "reason": "pointnet_resource_pilot_failed", "provenance": _provenance(prepared, architecture, seed),
            "attempt": attempt, "epoch_count": 0, "optimizer_update_count": 0, "test_rows_loaded": 0,
        }
        _write_json(attempt_root / "result.json", report)
        return report
    graph_contract._require(torch.cuda.is_available(), "full PointNet training requires CUDA")
    device = torch.device("cuda", 0)
    ranker_train.configure_determinism(seed)
    model = _model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    provenance = _provenance(prepared, architecture, seed)
    attempt_root.mkdir(parents=True, exist_ok=True)
    start_epoch, updates, best, history = 1, 0, None, []
    if resume:
        payload, _ = ranker_train.load_checkpoint(resume, model, optimizer, provenance)
        start_epoch, updates, best = payload["epoch"] + 1, payload["optimizer_updates"], payload["best"]
        old_best = Path(resume).parent / "epoch-{:03d}.pt".format(best["epoch"])
        new_best = attempt_root / "checkpoints" / old_best.name
        new_best.parent.mkdir(parents=True, exist_ok=True)
        if old_best.resolve() != new_best.resolve():
            shutil.copy2(old_best, new_best)
            shutil.copy2(str(old_best) + ".json", str(new_best) + ".json")
    train_rows = _inventory(prepared, "train")
    validation_rows = _inventory(prepared, "validation")
    counts = ranker_train._loss_counts(prepared["audit"], False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, prepared["spec"]["training"]["epochs"] + 1):
        model.train()
        losses = []
        for rows in ranker_train._assembly_batches(train_rows, seed, epoch):
            records = [record for row in rows for record in ranker_train._load_assembly(prepared, row)]
            batch = _batch(prepared, records, rows, architecture, device)
            optimizer.zero_grad(set_to_none=True)
            loss = ranker_models.model_loss(batch, model(batch), counts)[0] if architecture == "pointnet_graph_transformer" else pointnet_models.cebulla_loss(batch, model(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        predictions, inference = (_pointnet_predictions if architecture == "pointnet_graph_transformer" else _cebulla_predictions)(model, prepared, validation_rows, device)
        selection = ranker_train._selection_metrics(predictions) if architecture == "pointnet_graph_transformer" else _cebulla_selection(predictions)
        selection["epoch"] = epoch
        key = ((selection["offline_context_success_at_1"], selection["first_feasible_mrr"], selection["Y_average_precision"])
               if architecture == "pointnet_graph_transformer" else (selection["validation_part_success_at_1"], selection["validation_average_precision"])) + (-epoch,)
        if best is None or tuple(best["key"]) < key:
            best = {"epoch": epoch, "key": list(key), "selection": selection}
        checkpoint_path = attempt_root / "checkpoints" / "epoch-{:03d}.pt".format(epoch)
        checkpoint = ranker_train.save_checkpoint(checkpoint_path, model, optimizer, provenance, epoch, updates, selection, best)
        history.append({"epoch": epoch, "mean_batch_loss": float(np.mean(losses)), "optimizer_updates": updates,
                        "validation_selection": selection, "checkpoint": checkpoint_path.relative_to(attempt_root).as_posix(),
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"], "validation_inference_seconds": inference})
    best_path = attempt_root / "checkpoints" / "epoch-{:03d}.pt".format(best["epoch"])
    report = {
        "schema_version": "fast.pointnet-training-run.v1", "status": "passed", "provenance": provenance,
        "attempt": attempt, "resumed_from": str(resume) if resume else None,
        "epoch_count": prepared["spec"]["training"]["epochs"], "optimizer_update_count": updates,
        "best": best, "best_checkpoint": best_path.relative_to(attempt_root).as_posix(),
        "best_checkpoint_sha256": production._sha256_file(best_path), "history": history,
        "observations": {"training_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
                         "peak_host_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "checkpoint_bytes": sum(path.stat().st_size for path in (attempt_root / "checkpoints").glob("*.pt*"))},
        "test_rows_loaded": 0,
    }
    _write_json(attempt_root / "result.json", report)
    return report


def _pilot_trace(batch, counts, steps, device, initial=None):
    ranker_train.configure_determinism(0)
    model = _model("pointnet_graph_transformer").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    if initial is None:
        initial = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": copy.deepcopy(ranker_train._rng_state())}
    else:
        model.load_state_dict(copy.deepcopy(initial["model"]))
        optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
        ranker_train._restore_rng(copy.deepcopy(initial["rng"]))
    trace = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = ranker_models.model_loss(batch, model(batch), counts)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        trace.append(float(loss.detach()))
    final = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": copy.deepcopy(ranker_train._rng_state())}
    return trace, ranker_train._state_digest(model), initial, final


def gpu_pilot(spec_path, source_root, data_root, graph_root=None, point_root=None, run_root=None):
    prepared = _prepare(spec_path, source_root, data_root, graph_root, point_root, run_root)
    graph_contract._require(torch.cuda.is_available(), "GPU pilot requires CUDA")
    device = torch.device("cuda", 0)
    train_rows = _inventory(prepared, "train")
    rows = train_rows[:8]
    records = [record for row in rows for record in ranker_train._load_assembly(prepared, row)]
    batch = _batch(prepared, records, rows, "pointnet_graph_transformer", device)
    counts = ranker_train._loss_counts(prepared["audit"], False)
    first, first_digest, initial, _ = _pilot_trace(batch, counts, 2, device)
    replay, replay_digest, _, _ = _pilot_trace(batch, counts, 2, device, initial)
    graph_contract._require(first == replay and first_digest == replay_digest, "deterministic replay differs")
    one, _, initial, interrupted = _pilot_trace(batch, counts, 1, device)
    resumed, resumed_digest, _, _ = _pilot_trace(batch, counts, 1, device, interrupted)
    uninterrupted, uninterrupted_digest, _, _ = _pilot_trace(batch, counts, 2, device, initial)
    graph_contract._require(one + resumed == uninterrupted and resumed_digest == uninterrupted_digest, "resume replay differs")
    ranker_train.configure_determinism(0)
    model, optimizer = _model("pointnet_graph_transformer").to(device), None
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    torch.cuda.reset_peak_memory_stats(device)
    started, candidates = time.perf_counter(), 0
    for batch_rows in ranker_train._assembly_batches(train_rows, 0, 1):
        records = [record for row in batch_rows for record in ranker_train._load_assembly(prepared, row)]
        candidates += sum(len(record["candidate_index"]) for record in records)
        training_batch = _batch(prepared, records, batch_rows, "pointnet_graph_transformer", device)
        optimizer.zero_grad(set_to_none=True)
        loss = ranker_models.model_loss(training_batch, model(training_batch), counts)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        model(batch)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    _, validation_inference_seconds = _pointnet_predictions(model, prepared, _inventory(prepared, "validation"), device)
    projected = (seconds + validation_inference_seconds) * prepared["spec"]["training"]["epochs"]
    gpu_bytes, host_kib = torch.cuda.max_memory_allocated(device), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    limits = prepared["spec"]["resources"]["gpu"]
    eligible = projected <= limits["wall_hours"] * 3600 and gpu_bytes <= 40 * 1024 ** 3 and host_kib <= limits["memory_gib"] * 1024 ** 2
    report = {
        "schema_version": "fast.pointnet-gpu-pilot.v1", "status": "passed", "eligible": eligible,
        "provenance": _provenance(prepared, "pointnet_graph_transformer", 0),
        "determinism": {"loss_trace": first, "parameter_state_sha256": first_digest, "resume_verified": True},
        "resource_probe": {"training_assemblies": len(train_rows), "candidate_count": candidates, "one_epoch_seconds": seconds,
                           "projected_100_epoch_seconds": projected, "candidates_per_second": candidates / seconds,
                           "inference_batch_candidates": int(len(batch["candidate_index"])),
                           "inference_batch_seconds": inference_seconds, "validation_inference_seconds": validation_inference_seconds,
                           "peak_gpu_bytes": gpu_bytes,
                           "peak_host_kib": host_kib, "scratch_bytes": 0},
        "limits": limits, "test_rows_loaded": 0,
    }
    output = _pilot_path(prepared)
    graph_contract._require(not output.exists(), "GPU pilot result cannot be overwritten")
    _write_json(output, report)
    return report


def audit(spec_path, source_root, data_root, graph_root=None, point_root=None, run_root=None):
    prepared = _prepare(spec_path, source_root, data_root, graph_root, point_root, run_root)
    artifact_implementation = _training_implementation(prepared)
    expected = {(row["architecture"], row["seed"]) for row in _matrix()}
    selected, ineligible = {}, []
    pilot_path = _pilot_path(prepared, artifact_implementation)
    pilot = _read_json(pilot_path)
    _validate_gpu_provenance(prepared, pilot.get("provenance"), "pointnet_graph_transformer", 0, artifact_implementation)
    graph_contract._require(len(list(pilot_path.parent.glob("gpu-pilot-*.json"))) <= 2, "more than one implementation correction was used")
    expected_updates = prepared["spec"]["training"]["epochs"] * int(
        math.ceil(len(_inventory(prepared, "train")) / prepared["spec"]["training"]["assemblies_per_batch"])
    )
    for result_path in sorted((prepared["run_root"] / "training").glob("*/seed-*/attempt-*/result.json")):
        result = _read_json(result_path)
        provenance = result.get("provenance", {})
        key = (provenance.get("architecture"), provenance.get("seed"))
        graph_contract._require(key in expected and result.get("status") in ("passed", "ineligible") and result.get("test_rows_loaded") == 0, "unexpected training result")
        _validate_gpu_provenance(prepared, provenance, *key, artifact_implementation)
        if result["status"] == "ineligible":
            graph_contract._require(key[0] == "pointnet_graph_transformer" and pilot.get("eligible") is False and result.get("epoch_count") == 0, "invalid ineligible training result")
            ineligible.append({"architecture": key[0], "seed": key[1], "result_path": result_path.relative_to(prepared["run_root"]).as_posix(),
                               "result_sha256": production._sha256_file(result_path), "reason": result["reason"]})
        else:
            graph_contract._require(
                result.get("epoch_count") == prepared["spec"]["training"]["epochs"]
                and result.get("optimizer_update_count") == expected_updates,
                "training budget differs",
            )
            checkpoint = result_path.parent / result["best_checkpoint"]
            graph_contract._require(production._sha256_file(checkpoint) == result["best_checkpoint_sha256"], "best checkpoint hash differs")
        graph_contract._require(key not in selected or selected[key]["result"]["attempt"] < result["attempt"], "duplicate attempt number")
        selected[key] = {"path": result_path, "result": result}
    graph_contract._require(set(selected) == expected, "complete six-job training matrix is absent")
    inventory = []
    for (architecture, seed), item in sorted(selected.items()):
        result, path = item["result"], item["path"]
        if result["status"] == "ineligible":
            continue
        inventory.append({"architecture": architecture, "seed": seed, "attempt": result["attempt"],
                          "result_path": path.relative_to(prepared["run_root"]).as_posix(), "result_sha256": production._sha256_file(path),
                          "best_checkpoint": (path.parent / result["best_checkpoint"]).relative_to(prepared["run_root"]).as_posix(),
                          "best_checkpoint_sha256": result["best_checkpoint_sha256"], "parameter_count": EXPECTED_PARAMETERS[architecture],
                          "optimizer_update_count": result["optimizer_update_count"], "observations": result["observations"]})
    report = {
        "schema_version": "fast.pointnet-training-audit.v1", "status": "passed", "spec_sha256": prepared["spec_sha256"],
        "ranker_graph_audit_sha256": prepared["audit_sha256"], "point_audit_sha256": prepared["point_audit_sha256"],
        "successful_job_count": len(inventory), "ineligible_job_count": len(ineligible),
        "optimizer_update_count_per_job": expected_updates, "inventory": inventory, "ineligible": ineligible,
        "pointnet_resource_eligible": pilot.get("eligible") is True,
        "artifact_implementation": artifact_implementation, "audit_implementation": prepared["implementation"], "test_rows_loaded": 0,
    }
    _write_json(prepared["run_root"] / "training-audit.json", report)
    return report


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(root / "configs" / "pointnet.json"))
    parser.add_argument("--source-root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--graph-root"); parser.add_argument("--point-root"); parser.add_argument("--run-root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup"); commands.add_parser("gpu-pilot")
    training = commands.add_parser("train")
    identity = training.add_mutually_exclusive_group(required=True)
    identity.add_argument("--process", type=int); identity.add_argument("--architecture", choices=ARCHITECTURES)
    training.add_argument("--seed", type=int, choices=(0, 1, 2)); training.add_argument("--attempt", type=int, default=1); training.add_argument("--resume")
    commands.add_parser("audit")
    args = parser.parse_args(argv)
    common = (args.spec, args.source_root, args.data_root, args.graph_root, args.point_root, args.run_root)
    if args.command == "setup": result = setup(*common)
    elif args.command == "gpu-pilot": result = gpu_pilot(*common)
    elif args.command == "train":
        graph_contract._require(args.process is not None or args.seed is not None, "--seed is required with --architecture")
        result = train(*common, args.process, args.architecture, args.seed, args.attempt, args.resume)
    else: result = audit(*common)
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
