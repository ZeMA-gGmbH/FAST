"""Local immutable registry and runner for the nine component-ablation D retraining jobs.

Identities are fixed by the frozen spec. The runner enforces the protocol order (pretraining gate,
D0-seed-0 smoke, D0-seed-0, D1/D2-seed-0, a recorded continuation before seeds 1-2 as matched
rounds), refuses to overwrite or duplicate completed identities, preserves every attempt, records
per-attempt stdout/stderr and resources, and binds the optional W&B mirror without making it authoritative.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import platform
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

from fast import ablation_common as common


REGISTRY_SCHEMA_VERSION = "fast.ablation-registry.v1"
LEDGER_SCHEMA_VERSION = "fast.ablation-attempt.v1"
SPEC_PATH = common.REPOSITORY_ROOT / "configs" / "component_ablation.json"
# Python of the environment/cu121 runtime, probed for the environment record.
COMPAT_PYTHON = os.environ.get("FAST_COMPAT_PYTHON", sys.executable)
SEED_ROUNDS = ("seed-0", "seed-1", "seed-2")


def _spec():
    spec = common.read_json(SPEC_PATH)
    common.require(spec["status"] == "frozen" and spec["run_id"] == common.RUN_ID, "spec is not frozen")
    return spec, common.sha256_file(SPEC_PATH)


def identity_sha256(row, spec_sha256):
    return common.canonical_sha256({"run_id": common.RUN_ID, "variant": row["variant"], "seed": row["seed"],
                                    "run_name": row["run_name"], "spec_sha256": spec_sha256})


def build_registry(spec, spec_sha256):
    matrix = spec["experiments"]["D"]["matrix"]
    names = [row["run_name"] for row in matrix]
    keys = [(row["variant"], row["seed"]) for row in matrix]
    common.require(len(matrix) == 9 and len(set(names)) == 9 and len(set(keys)) == 9, "training matrix is not nine unique identities")
    common.require(sorted(keys) == sorted((variant, seed) for variant in ("D0", "D1", "D2") for seed in (0, 1, 2)), "training matrix differs")
    rows = [OrderedDict(list(row.items()) + [("identity_sha256", identity_sha256(row, spec_sha256))]) for row in matrix]
    return OrderedDict([
        ("schema_version", REGISTRY_SCHEMA_VERSION), ("run_id", common.RUN_ID), ("spec_sha256", spec_sha256),
        ("matrix", rows), ("order", spec["experiments"]["D"]["execution_order"]),
        ("stop_rule", spec["experiments"]["D"]["stop_rule"]), ("concurrency_rule", spec["experiments"]["D"]["concurrency_rule"]),
        ("telemetry", spec["telemetry"]), ("test_rows_loaded", 0),
    ])


def load_registry(root):
    spec, spec_sha256 = _spec()
    registry = common.read_json(root / "registry.json")
    common.require(common.canonical_json(registry) == common.canonical_json(build_registry(spec, spec_sha256)), "registry differs from the frozen spec")
    return registry


def _run(command):
    try:
        return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120).stdout.strip()
    except Exception as error:
        return "unavailable: {}".format(error)


def workstation():
    meminfo = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    wslconfig = sorted(Path("/mnt/c/Users").glob("*/.wslconfig")) if Path("/mnt/c/Users").exists() else []
    return OrderedDict([
        ("uname", platform.uname()._asdict()), ("nproc", os.cpu_count()),
        ("mem_total_kib", int(next(line for line in meminfo if line.startswith("MemTotal")).split()[1])),
        ("swap_total_kib", int(next(line for line in meminfo if line.startswith("SwapTotal")).split()[1])),
        ("disk_free_bytes", shutil.disk_usage(str(common.REPOSITORY_ROOT)).free),
        ("nvidia_smi", _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])),
        ("wsl_lib_libcuda_present", Path("/usr/lib/wsl/lib/libcuda.so").exists()),
        ("linux_nvidia_driver_installed", Path("/proc/driver/nvidia/version").exists()),
        ("wslconfig", [{"path": str(path), "content": path.read_text(encoding="utf-8", errors="replace")} for path in wslconfig]),
    ])


def compatibility_research():
    return OrderedDict([
        ("accessed", "2026-10-09"), ("scope", "bounded official-documentation runtime check, not a literature review"),
        ("sources", [
            "https://pytorch.org/blog/pytorch-2-7/",
            "https://github.com/pytorch/pytorch/releases/tag/v2.7.0",
            "https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/.ci/manywheel/build_cuda.sh",
            "https://raw.githubusercontent.com/pytorch/pytorch/main/RELEASE.md",
            "https://pytorch.org/get-started/previous-versions/",
            "https://download.pytorch.org/whl/cu128/torch/",
            "https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html",
            "https://docs.pytorch.org/docs/2.7/notes/randomness.html",
            "https://docs.pytorch.org/docs/2.7/notes/cuda.html",
            "https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html",
            "https://github.com/pyg-team/pytorch_geometric/releases/tag/2.7.0",
            "https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html",
            "https://docs.nvidia.com/cuda/wsl-user-guide/index.html",
            "https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl",
        ]),
        ("findings", [
            "PyTorch 2.7.0 is the first stable release with CUDA 12.8 wheels; its cu128 build list includes sm_120 (12.0+PTX); latest 2.7 patch is 2.7.1",
            "torch 2.7 requires Python >= 3.9, so the frozen Python 3.8 runtime cannot run on Blackwell",
            "PyG 2.7.0 officially supports PyTorch 2.6-2.8 and requires Python >= 3.10; optional extensions are not needed",
            "CUDA 12.8 adds SM_120; Windows driver 591.86 exceeds the CUDA 12.8 minimum",
            "WSL uses the Windows driver via /usr/lib/wsl/lib; no Linux NVIDIA driver may be installed in WSL",
            "deterministic CUDA matmul requires CUBLAS_WORKSPACE_CONFIG=:4096:8; cudnn.allow_tf32 defaults to True and is disabled explicitly",
        ]),
        ("selected", "Python 3.10.20, torch 2.7.1+cu128 (official cu128 index), torch-geometric 2.7.0; all other shared pins equal the frozen ranker lock"),
        ("rejected", "nightly builds (a stable build passed); PyG 2.6.1 with torch 2.7 (not officially listed)"),
    ])


def prepare(data_root):
    """Publish the immutable registry, nine-row matrix, and environment record."""
    from fast import ablation_train
    root = common.run_root(data_root)
    spec, spec_sha256 = _spec()
    registry = build_registry(spec, spec_sha256)
    outputs = OrderedDict()
    for name, value in (("registry.json", registry), ("training-matrix.json", {"run_id": common.RUN_ID, "spec_sha256": spec_sha256, "matrix": registry["matrix"]})):
        path = root / name
        if path.exists():
            common.require(common.canonical_json(common.read_json(path)) == common.canonical_json(value), name + " differs")
            outputs[name] = common.sha256_file(path)
        else:
            outputs[name] = common.write_json_once(path, value)
    if not (root / "environment.json").exists():
        compat = _run([str(COMPAT_PYTHON), "-c", "import sys,torch,torch_geometric,numpy;print(sys.version.split()[0],torch.__version__,torch_geometric.__version__,numpy.__version__)"])
        environment = OrderedDict([
            ("run_id", common.RUN_ID), ("recorded_at", common.utc_now()), ("implementation", common.implementation_identity(require_clean=True)),
            ("workstation", workstation()),
            ("training_runtime", ablation_train.training_environment()),
            ("compatibility_runtime", OrderedDict([("python_torch_pyg_numpy", compat), ("lock", "environment/cu121/uv.lock"),
                                                   ("lock_sha256", common.sha256_file(common.REPOSITORY_ROOT / "environment/cu121/uv.lock")),
                                                   ("device", "cpu (CUDA hidden)")])),
            ("uv", _run([shutil.which("uv") or "uv", "--version"])),
            ("compatibility_research", compatibility_research()), ("test_rows_loaded", 0),
        ])
        outputs["environment.json"] = common.write_json_once(root / "environment.json", environment)
    else:
        outputs["environment.json"] = common.sha256_file(root / "environment.json")
    return outputs


# ---------------------------------------------------------------- attempts


def attempt_dirs(root, row):
    base = root / "training" / row["variant"] / "seed-{}".format(row["seed"])
    return sorted(base.glob("attempt-*"), key=lambda path: int(path.name.split("-")[1])) if base.exists() else []


def completed(root, row):
    passed = []
    for directory in attempt_dirs(root, row):
        result = directory / "result.json"
        if result.exists() and common.read_json(result).get("status") == "passed":
            passed.append(directory)
    common.require(len(passed) <= 1, "duplicate completed attempts for " + row["run_name"])
    return passed[0] if passed else None


def running(root):
    active = []
    for lock in root.glob("training/*/seed-*/attempt-*/running.pid"):
        try:
            pid = int(lock.read_text(encoding="ascii").split()[0])
            os.kill(pid, 0)
            active.append(lock.parent)
        except (OSError, ValueError):
            continue
    return active


def _ledger(root, value):
    with (root / "attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(common.canonical_json(dict(value, schema_version=LEDGER_SCHEMA_VERSION)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_gate(root):
    gate = common.read_json(root / "pretraining-gate.json")
    common.require(gate.get("status") == "passed" and gate.get("d_training_authorized") is True, "A/B/C/E pretraining gate has not passed")
    for name, digest in gate["bound_outputs"].items():
        common.require(common.sha256_file(root / name) == digest, "pretraining-gate output changed: " + name)
    return gate


def require_order(root, registry, row, allow_concurrent):
    matrix = {item["run_name"]: item for item in registry["matrix"]}
    smoke = root / "smoke" / "d0-seed-0-smoke.json"
    common.require(smoke.exists() and common.read_json(smoke).get("status") == "passed", "D0-seed-0 deterministic/resource smoke has not passed")
    if row["run_name"] != "D0-seed-0":
        common.require(completed(root, matrix["D0-seed-0"]) is not None, "D0-seed-0 must complete before other D jobs")
    if row["round"] != "seed-0":
        continuation = root / "seed-continuation.json"
        common.require(continuation.exists() and common.read_json(continuation).get("continue") is True,
                       "seeds 1-2 require a recorded continuation after the seed-0 status report")
        common.require(all(completed(root, item) for item in registry["matrix"] if item["round"] == "seed-0"), "seed-0 round is incomplete")
    if row["round"] == "seed-2":
        common.require(all(completed(root, item) for item in registry["matrix"] if item["round"] == "seed-1"), "seed-1 matched round is incomplete")
    active = running(root)
    if active:
        decision = root / "concurrency-decision.json"
        common.require(allow_concurrent and decision.exists(), "another D job is running and no concurrency decision allows overlap")
        allowed = common.read_json(decision)["allowed_concurrent_jobs"]
        common.require(len(active) + 1 <= allowed, "concurrency decision allows only {} jobs".format(allowed))


def launch(data_root, run_name, telemetry_mode="online", resume_attempt=None, allow_concurrent=False, python=None):
    root = common.run_root(data_root)
    registry = load_registry(root)
    rows = [row for row in registry["matrix"] if row["run_name"] == run_name]
    common.require(len(rows) == 1, "unknown run name")
    row = rows[0]
    implementation = common.implementation_identity(require_clean=True)
    require_gate(root)
    common.require(completed(root, row) is None, "completed run cannot be rerun or overwritten: " + run_name)
    require_order(root, registry, row, allow_concurrent)
    existing = attempt_dirs(root, row)
    attempt = 1 + max([int(path.name.split("-")[1]) for path in existing] or [0])
    attempt_root = root / "training" / row["variant"] / "seed-{}".format(row["seed"]) / "attempt-{}".format(attempt)
    attempt_root.mkdir(parents=True, exist_ok=False)
    resume = None
    if resume_attempt is not None:
        source = attempt_root.parent / "attempt-{}".format(resume_attempt)
        epochs = [entry["epoch"] for entry in common.read_jsonl(source / "epochs.jsonl")] if (source / "epochs.jsonl").exists() else []
        common.require(epochs, "resume source has no completed epoch")
        resume = source / "checkpoints" / "epoch-{:03d}.pt".format(max(epochs))
    python = python or sys.executable
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    command = [python, "-m", "fast.ablation_train", "--data-root", str(data_root), "--variant", row["variant"],
               "--seed", str(row["seed"]), "--attempt-root", str(attempt_root), "--run-name", run_name]
    if resume:
        command += ["--resume", str(resume)]
    started = time.time()
    start_record = OrderedDict([("event", "start"), ("run_name", run_name), ("identity_sha256", row["identity_sha256"]),
                                ("attempt", attempt), ("attempt_root", attempt_root.relative_to(root).as_posix()),
                                ("implementation_commit", implementation["commit"]), ("telemetry_mode", telemetry_mode),
                                ("resume_from", str(resume.relative_to(root)) if resume else None), ("started_at", common.utc_now())])
    with (attempt_root / "stdout.log").open("w") as stdout, (attempt_root / "stderr.log").open("w") as stderr:
        trainer = subprocess.Popen(command, cwd=str(common.REPOSITORY_ROOT), env=environment, stdout=stdout, stderr=stderr)
    (attempt_root / "running.pid").write_text("{}\n".format(trainer.pid), encoding="ascii")
    _ledger(root, dict(start_record, pid=trainer.pid))
    mirror = None
    if telemetry_mode != "disabled":
        with (attempt_root / "telemetry.log").open("w") as log:
            mirror = subprocess.Popen([python, "-m", "fast.ablation_telemetry", "--attempt-root", str(attempt_root),
                                       "--run-name", run_name, "--mode", telemetry_mode, "--trainer-pid", str(trainer.pid)],
                                      cwd=str(common.REPOSITORY_ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                      stdout=log, stderr=subprocess.STDOUT)
    else:
        from fast import ablation_telemetry
        ablation_telemetry.mirror(attempt_root, run_name, "disabled")
    exit_code = trainer.wait()
    wall_seconds = time.time() - started
    if mirror is not None:
        try:
            mirror.wait(timeout=900)
        except subprocess.TimeoutExpired:
            mirror.terminate()
    (attempt_root / "running.pid").unlink()
    result = attempt_root / "result.json"
    passed = exit_code == 0 and result.exists() and common.read_json(result).get("status") == "passed"
    if not passed:
        stderr_tail = (attempt_root / "stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        common.write_json_once(attempt_root / "failed.json", {"run_name": run_name, "attempt": attempt, "exit_code": exit_code,
                                                               "wall_seconds": wall_seconds, "stderr_tail": stderr_tail})
    _ledger(root, OrderedDict([("event", "end"), ("run_name", run_name), ("attempt", attempt), ("exit_code", exit_code),
                               ("status", "passed" if passed else "failed"), ("wall_seconds", wall_seconds),
                               ("result_sha256", common.sha256_file(result) if result.exists() else None),
                               ("wandb", common.read_json(attempt_root / "wandb.json") if (attempt_root / "wandb.json").exists() else None),
                               ("ended_at", common.utc_now())]))
    write_wandb_mapping(root)
    return {"run_name": run_name, "attempt": attempt, "status": "passed" if passed else "failed", "exit_code": exit_code, "wall_seconds": wall_seconds}


def write_wandb_mapping(root):
    mapping = []
    for path in sorted(root.glob("training/*/seed-*/attempt-*/wandb.json")):
        record = common.read_json(path)
        mapping.append(OrderedDict([("attempt_root", path.parent.relative_to(root).as_posix())] +
                                   [(key, record.get(key)) for key in ("name", "project", "group", "mode", "run_id", "url", "status", "logged_epochs", "offline_sync_required")]))
    return common.write_json_replace(root / "wandb-mapping.json", {"run_id": common.RUN_ID, "authoritative": False, "credentials_recorded": False, "runs": mapping})


def authorize_continuation(data_root, statement):
    """Record the explicit decision to run seeds 1-2 after reviewing the outcome-free seed-0 status."""
    root = common.run_root(data_root)
    common.require(statement.strip(), "a continuation statement is required")
    return common.write_json_once(root / "seed-continuation.json", {"continue": True, "statement": statement, "recorded_at": common.utc_now()})


def status(data_root):
    root = common.run_root(data_root)
    registry = load_registry(root)
    rows = []
    for row in registry["matrix"]:
        attempts = attempt_dirs(root, row)
        latest = attempts[-1] if attempts else None
        epochs = len(common.read_jsonl(latest / "epochs.jsonl")) if latest and (latest / "epochs.jsonl").exists() else 0
        rows.append({"run_name": row["run_name"], "attempts": len(attempts), "completed": completed(root, row) is not None,
                     "latest_epochs": epochs, "running": bool(latest and latest in running(root))})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    run_parser = commands.add_parser("launch")
    run_parser.add_argument("--run-name", required=True)
    run_parser.add_argument("--telemetry", choices=("online", "offline", "disabled"), default="disabled")
    run_parser.add_argument("--resume-attempt", type=int)
    run_parser.add_argument("--allow-concurrent", action="store_true")
    continuation = commands.add_parser("authorize-continuation")
    continuation.add_argument("--statement", required=True)
    commands.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.data_root)
    elif args.command == "launch":
        result = launch(args.data_root, args.run_name, args.telemetry, args.resume_attempt, args.allow_concurrent)
    elif args.command == "authorize-continuation":
        result = authorize_continuation(args.data_root, args.statement)
    else:
        result = status(args.data_root)
    print(common.canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
