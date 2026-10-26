"""Shared identity, guard, and immutable-output helpers for the component-ablation study."""

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from plan_sequence import production_sim_labels as production


RUN_ID = "component-ablation-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PARTS = ("test_assembly", "test")
DIRECTIONS = ("+Z", "-Z", "+X", "-X", "+Y", "-Y")
TIERS = ("small", "medium", "large")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    return production._sha256_file(path)


def canonical_json(value):
    return production._canonical_json(value)


def canonical_sha256(value):
    return production._sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require_development_path(path):
    """Reject any path component that names the sealed test split."""
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    require(not any(part in FORBIDDEN_PATH_PARTS for part in parts), "test-bearing path rejected: {}".format(path))
    return path


def _atomic_write(path, text, overwrite):
    path = Path(path)
    require(overwrite or not path.exists(), "immutable output cannot be overwritten: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(path)


def write_json_once(path, value):
    """Atomically publish a canonical JSON artifact and refuse to replace an existing one."""
    return _atomic_write(path, canonical_json(value) + "\n", overwrite=False)


def write_json_replace(path, value):
    """Atomically replace a mutable bookkeeping file such as an attempt ledger."""
    return _atomic_write(path, canonical_json(value) + "\n", overwrite=True)


def implementation_identity(require_clean=True, root=REPOSITORY_ROOT):
    def git(*args):
        return subprocess.run(["git", *args], cwd=str(root), check=True, text=True, stdout=subprocess.PIPE).stdout

    commit = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain", "--untracked-files=normal")
    attributes = Path(root) / ".git" / "info" / "attributes"
    overrides = [
        line.strip() for line in attributes.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ] if attributes.is_file() else []
    identity = {
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "clean": status == "",
        "local_attribute_overrides": overrides,
    }
    require(len(commit) == 40, "invalid Git HEAD")
    if require_clean:
        require(identity["clean"], "implementation repository is dirty:\n{}".format(status))
    return identity


def run_root(data_root):
    return Path(data_root).resolve() / "runs" / RUN_ID


def closure_root(data_root):
    return run_root(data_root) / "closure"
