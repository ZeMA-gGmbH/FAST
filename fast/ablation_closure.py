"""Build, transfer, and verify the fail-closed validation-only component-ablation data closure.

Every file is listed by namespace, source-relative path, byte count, and SHA-256 before
transfer; verification rejects unlisted, missing, altered, or test-bearing files. The source
is FAST_SOURCE_DATA_ROOT (FAST data root) and FAST_SOURCE_GEOMETRY_ROOT (ASAP assemblies),
read locally or over SSH from FAST_SOURCE_HOST when that is set.
"""

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from fast import ablation_common as common
from plan_sequence import production_sim_labels as production


MANIFEST_SCHEMA_VERSION = "fast.ablation-transfer-manifest.v1"
VERIFICATION_SCHEMA_VERSION = "fast.ablation-transfer-verification.v1"
GRAPH_ROOT = "features/graph-records-v2"
GRAPH_AUDIT = GRAPH_ROOT + "/audit.json"
LABEL_ROOT = "labels/fast-labels-v1/training_assembly"
GEOMETRY_ROOT = "training_assembly"
REPLAY_FIXTURE = "00014"
SMALL_VALIDATION_COUNT = 115
EXPECTED_LABEL_TOTALS = {"nonterminal_states": 788, "terminal_states": 344, "candidate_files": 6627}
GEOMETRY_FILE = re.compile(r"^(?:[^/]+\.obj|contact_graph\.json|config\.json)$")
ASSEMBLY_ID = re.compile(r"^[0-9]{5}$")
RANKER_RUN = "runs/ranker-training-v2"
POINTNET_RUN = "runs/pointnet-v1"
FAST_CHECKPOINTS = {0: "epoch-034.pt", 1: "epoch-099.pt", 2: "epoch-044.pt"}
CEBULLA_CHECKPOINTS = {0: "epoch-012.pt", 1: "epoch-091.pt", 2: "epoch-065.pt"}
REPLAY_TRACES = (
    "runs/planner-runtime-v1/fast-seed-0-00014.jsonl",
    "runs/planner-cpu-smoke-v1/request-1-threads-4/fast.jsonl",
    "runs/planner-cpu-smoke-v1/request-1-threads-4/asap_random.jsonl",
    "runs/planner-cpu-smoke-v1/request-1-threads-4/asap_heur_out.jsonl",
    "runs/planner-cpu-smoke-v1/request-1-threads-4/cebulla_gcn.jsonl",
    "runs/planner-cpu-smoke-v1/request-1-threads-4/heur_out_direction_prior.jsonl",
)
# Hashes of the frozen thesis inputs.
FROZEN_SHA256 = {
    "datasets/fast-labels-v1/manifest.json": "b57465c0dd2568081f6c511f860435e23242c9b64c76df32eafb7bf6df2ce131",
    "features/scalar-features-v3/manifest.json": "d0c86ad0144c77e49ae221b041e11a56a080719fec2be43b497fc6a7a007827a",
    GRAPH_AUDIT: "5ada40534489984c344ec0bd9ada9939e194f33328a8b6ef11cb908544302682",
    RANKER_RUN + "/training/graph_transformer/seed-0/attempt-1/checkpoints/epoch-034.pt": "5f44170fb749f4f71963599bf6a28b613f91d9b79c7df56dfbec394e27a0c1dc",
    RANKER_RUN + "/training/graph_transformer/seed-1/attempt-1/checkpoints/epoch-099.pt": "8ccfee223d9bd854034de14693ca43bde7c67bba0569b61ed3095f460d310e0e",
    RANKER_RUN + "/training/graph_transformer/seed-2/attempt-1/checkpoints/epoch-044.pt": "38cd41a9a98854fc3977aba64104e0189a0d4880bb49a38374f4bb13a396d422",
    RANKER_RUN + "/training-audit.json": "ff02fdfdc15cb938a1e23d2c4a111eb3a55096f341315e80ec390871f3fd6ff1",
    RANKER_RUN + "/validation/summary.json": "7f693ca32240af121e2c83e0c787d78193e78aba9bd988a5d0e74186cc09efb8",
    RANKER_RUN + "/validation-audit.json": "ed7cd1155a6283f817a060737f040797fa66fd7437bf24eb99ca1e85b95a9f3f",
    RANKER_RUN + "/bootstrap/resample-plan.json": "a3f37a724ac16af0dc10080d501f4d73318f2dbab322f77c14afbeaa70a670b8",
    RANKER_RUN + "/bootstrap/report.json": "0c3042a1ef6acf1a34e3ea28230b5fdfa5575550bab16ef3dee52396158bda7f",
    POINTNET_RUN + "/training/cebulla_gcn/seed-0/attempt-1/checkpoints/epoch-012.pt": "5b62d706ab379936f34591bcd701aa6e8da2f46253e2b096ae56b59f18b7cfc4",
    POINTNET_RUN + "/training/cebulla_gcn/seed-1/attempt-1/checkpoints/epoch-091.pt": "99c1e9375835853d9e2045add3945785a44088f775a105a85a030912ab489505",
    POINTNET_RUN + "/training/cebulla_gcn/seed-2/attempt-1/checkpoints/epoch-065.pt": "ba4e1d40daacafb76b4388e27ac04502c5331d04e462354a43fc139a555a94b5",
    POINTNET_RUN + "/training-audit.json": "7a478002d03901e5cb799c1c789bd496e551b320e7e0724d465255ce78c74459",
    POINTNET_RUN + "/validation/summary.json": "d17c57112ec78e3fd30f54cb436482bea2b8c033b69d2bcef66ef0783bc64436",
    POINTNET_RUN + "/validation-audit.json": "fe9e4b520d4e741e0a4c2c30bd95a67d46f1d3966e2740373892a74aec8b781b",
    POINTNET_RUN + "/planner/direction-prior.json": "853b7e2206e48c4a52d04674c09027956b6af8a0311e4163dd4c80b86b9afb39",
    REPLAY_TRACES[0]: "6c7e4a99b90a8bc72dca5c6e52c84a996110e7fdd2674097b067abd867de003b",
    REPLAY_TRACES[1]: "164a1f25b1d529ddd9e577d6ecd46c789c78aa935e14058dd3c77baad58a11e7",
    REPLAY_TRACES[2]: "dc977f1642018c06b04770f684bc0c2ae448c11da71ece97a9bc101073156fb3",
    REPLAY_TRACES[3]: "b61b91ff7db25fcc6ed7759d65d8d07aa468639f1a21601712d775f59f8a3e4b",
    REPLAY_TRACES[4]: "a52197b5cac2d695b45168263c004bd74b8cf1b3982d8a2858b27ea6a902e023",
    REPLAY_TRACES[5]: "1490b1e85c13b2472b51ff4654e6824e29ebab134b488fa84270ad6ab8ac58b9",
}


def _checkpoint_files(run, architecture, checkpoints):
    files = []
    for seed, name in sorted(checkpoints.items()):
        attempt = "{}/training/{}/seed-{}/attempt-1".format(run, architecture, seed)
        files += [attempt + "/checkpoints/" + name, attempt + "/checkpoints/" + name + ".json", attempt + "/result.json"]
    return files


def closure_requests(small_validation_ids):
    """Return (namespace, item, kind, path) requests; kind is file, tree, or geometry."""
    common.require(
        len(small_validation_ids) == SMALL_VALIDATION_COUNT
        and all(ASSEMBLY_ID.match(value) for value in small_validation_ids)
        and REPLAY_FIXTURE not in small_validation_ids,
        "small-validation assembly IDs differ",
    )
    requests = [
        ("fast-data", "1_dataset_index_manifest", "file", "datasets/fast-labels-v1/manifest.json"),
        ("fast-data", "2_feature_manifest", "file", "features/scalar-features-v3/manifest.json"),
        ("fast-data", "3_ranker_graph_tree", "tree", GRAPH_ROOT),
    ]
    requests += [("fast-data", "4_fast_checkpoints", "file", path) for path in
                 _checkpoint_files(RANKER_RUN, "graph_transformer", FAST_CHECKPOINTS) + [RANKER_RUN + "/training-audit.json"]]
    requests += [("fast-data", "5_cebulla_checkpoints", "file", path) for path in
                 _checkpoint_files(POINTNET_RUN, "cebulla_gcn", CEBULLA_CHECKPOINTS) + [
                     POINTNET_RUN + "/training-audit.json", POINTNET_RUN + "/validation/summary.json",
                     POINTNET_RUN + "/validation-audit.json",
                 ] + [POINTNET_RUN + "/validation/cebulla_gcn/seed-{}.json".format(seed) for seed in (0, 1, 2)]]
    requests.append(("fast-data", "6_direction_prior", "file", POINTNET_RUN + "/planner/direction-prior.json"))
    requests += [("fast-data", "7_ranker_validation", "file", path) for path in [
        RANKER_RUN + "/validation/summary.json", RANKER_RUN + "/validation-audit.json",
        RANKER_RUN + "/bootstrap/resample-plan.json", RANKER_RUN + "/bootstrap/report.json",
    ] + [RANKER_RUN + "/validation/graph_transformer/seed-{}.json".format(seed) for seed in (0, 1, 2)]]
    requests += [("fast-data", "8_small_validation_labels", "tree", LABEL_ROOT + "/" + value) for value in small_validation_ids]
    requests.append(("fast-data", "9_replay_fixture_labels", "tree", LABEL_ROOT + "/" + REPLAY_FIXTURE))
    requests += [("fast-data", "9_replay_traces", "file", path) for trace in REPLAY_TRACES for path in (trace, trace + ".sha256")]
    requests += [("asap-data", "10_small_validation_geometry", "geometry", GEOMETRY_ROOT + "/" + value) for value in small_validation_ids]
    requests.append(("asap-data", "9_replay_fixture_geometry", "geometry", GEOMETRY_ROOT + "/" + REPLAY_FIXTURE))
    for _namespace, _item, _kind, path in requests:
        common.require_development_path(path)
    return requests


def small_validation_ids(graph_audit):
    common.require(graph_audit.get("status") == "passed" and graph_audit.get("test_rows_loaded") == 0, "graph audit did not pass")
    inventory = graph_audit["inventory"]
    common.require(all(row["dataset_split"] in ("train", "validation") and row["source_split"] == "training_assembly"
                       for row in inventory), "graph audit contains non-development rows")
    counts = Counter((row["dataset_split"], row["tier"]) for row in inventory)
    common.require(sum(value for (split, _), value in counts.items() if split == "train") == 1703
                   and sum(value for (split, _), value in counts.items() if split == "validation") == 191,
                   "development split counts differ")
    selected = sorted(row["assembly_id"] for row in inventory if row["dataset_split"] == "validation" and row["tier"] == "small")
    common.require(len(selected) == SMALL_VALIDATION_COUNT, "small-validation count differs")
    common.require(any(row["assembly_id"] == REPLAY_FIXTURE and row["dataset_split"] == "train" for row in inventory),
                   "replay fixture is not a training assembly")
    return selected


def select_files(requests, listing):
    """Apply inclusion rules to a source listing of {(namespace, path): bytes}."""
    by_namespace = {}
    for (namespace, path), size in listing.items():
        by_namespace.setdefault(namespace, {})[path] = size
    selected, excluded = [], Counter()
    seen = set()
    for namespace, item, kind, request in requests:
        available = by_namespace.get(namespace, {})
        if kind == "file":
            matches = [request] if request in available else []
            common.require(matches, "requested closure file is absent: {}:{}".format(namespace, request))
        else:
            prefix = request.rstrip("/") + "/"
            matches = sorted(path for path in available if path.startswith(prefix))
            common.require(matches, "requested closure tree is empty: {}:{}".format(namespace, request))
        for path in matches:
            common.require_development_path(path)
            if kind == "geometry":
                name = path[len(request.rstrip("/")) + 1:]
                if "/" in name or not GEOMETRY_FILE.match(name):
                    excluded[PurePosixPath(name).suffix or name] += 1
                    continue
            if kind == "tree" and request.startswith(LABEL_ROOT):
                common.require(path.endswith((".json", ".jsonl")), "unexpected label payload: {}".format(path))
            key = (namespace, path)
            common.require(key not in seen, "closure file is requested twice: {}:{}".format(namespace, path))
            seen.add(key)
            selected.append({"namespace": namespace, "item": item, "path": path, "bytes": int(available[path])})
    for namespace, item, kind, request in requests:
        if kind == "geometry":
            names = {row["path"].rsplit("/", 1)[1] for row in selected if row["path"].startswith(request + "/")}
            common.require(any(name.endswith(".obj") for name in names) and "contact_graph.json" in names,
                           "geometry is incomplete: {}".format(request))
    return sorted(selected, key=lambda row: (row["namespace"], row["path"])), dict(sorted(excluded.items()))


def source():
    """Return the optional SSH host and the source root of each namespace."""
    roots = {"fast-data": os.environ.get("FAST_SOURCE_DATA_ROOT"), "asap-data": os.environ.get("FAST_SOURCE_GEOMETRY_ROOT")}
    common.require(all(roots.values()), "set FAST_SOURCE_DATA_ROOT and FAST_SOURCE_GEOMETRY_ROOT")
    return os.environ.get("FAST_SOURCE_HOST") or None, roots


def rsync_source(host, root):
    return "{}:{}/".format(host, root) if host else root.rstrip("/") + "/"


def _run(host, command, stdin_text=None, text=True):
    argv = ["ssh", "-o", "BatchMode=yes", host, command] if host else ["bash", "-c", command]
    result = subprocess.run(argv, input=stdin_text, text=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    common.require(result.returncode == 0, "source command failed ({}): {}".format(result.returncode, result.stderr[-2000:]))
    return result.stdout


def _source_bytes(namespace, path):
    host, roots = source()
    return _run(host, "cat -- '{}/{}'".format(roots[namespace], path), text=False)


def _source_listing(requests):
    host, roots = source()
    listing = {}
    for namespace, root in roots.items():
        paths = sorted({request for space, _item, _kind, request in requests if space == namespace})
        output = _run(host, "cd '{}' && xargs -d '\\n' -I{{}} find {{}} -type f -printf '%s\\t%p\\n'".format(root), "\n".join(paths) + "\n")
        for line in output.splitlines():
            size, path = line.split("\t", 1)
            listing[(namespace, path)] = int(size)
    return listing


def _source_hashes(selected):
    host, roots = source()
    hashes = {}
    for namespace, root in roots.items():
        paths = [row["path"] for row in selected if row["namespace"] == namespace]
        if not paths:
            continue
        output = _run(host, "cd '{}' && xargs -d '\\n' sha256sum --".format(root), "\n".join(paths) + "\n")
        for line in output.splitlines():
            digest, path = line.split("  ", 1)
            hashes[(namespace, path)] = digest
    return hashes


def plan(data_root):
    """Bind the complete closure file list and source hashes before any transfer."""
    identity = common.implementation_identity(require_clean=True)
    root = common.run_root(data_root)
    host, roots = source()
    audit_bytes = _source_bytes("fast-data", GRAPH_AUDIT)
    common.require(hashlib.sha256(audit_bytes).hexdigest() == FROZEN_SHA256[GRAPH_AUDIT], "source graph audit hash differs")
    import json
    selected_ids = small_validation_ids(json.loads(audit_bytes.decode("utf-8")))
    requests = closure_requests(selected_ids)
    files, excluded = select_files(requests, _source_listing(requests))
    hashes = _source_hashes(files)
    for row in files:
        row["sha256"] = hashes[(row["namespace"], row["path"])]
    frozen_checks = []
    for path, expected in sorted(FROZEN_SHA256.items()):
        actual = next((row["sha256"] for row in files if row["namespace"] == "fast-data" and row["path"] == path), None)
        common.require(actual == expected, "frozen input hash differs: {}".format(path))
        frozen_checks.append({"path": path, "sha256": expected, "matched": True})
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": common.RUN_ID,
        "created_at": common.utc_now(),
        "implementation": identity,
        "source_host": host,
        "source_roots": roots,
        "small_validation_assembly_ids": selected_ids,
        "replay_fixture_assembly_ids": [REPLAY_FIXTURE],
        "requests": [{"namespace": n, "item": i, "kind": k, "path": p} for n, i, k, p in requests],
        "geometry_rule": "include only *.obj, contact_graph.json, and optional config.json; exclude SDF and every other file",
        "geometry_excluded_file_counts": excluded,
        "item_file_counts": dict(sorted(Counter(row["item"] for row in files).items())),
        "item_bytes": dict(sorted((item, sum(row["bytes"] for row in files if row["item"] == item))
                                  for item in {row["item"] for row in files})),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "frozen_hash_checks": frozen_checks,
        "files": files,
        "test_paths_requested": 0,
        "test_rows_loaded": 0,
    }
    path = root / "transfer-manifest.json"
    digest = common.write_json_once(path, manifest)
    return {"manifest": str(path), "sha256": digest, "file_count": len(files), "total_bytes": manifest["total_bytes"]}


def _load_manifest(data_root):
    path = common.run_root(data_root) / "transfer-manifest.json"
    manifest = common.read_json(path)
    common.require(manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION and manifest.get("run_id") == common.RUN_ID,
                   "transfer manifest differs")
    return path, manifest


def transfer(data_root):
    _, manifest = _load_manifest(data_root)
    destination_root = common.closure_root(data_root)
    for namespace, source_root in manifest["source_roots"].items():
        paths = [row["path"] for row in manifest["files"] if row["namespace"] == namespace]
        destination = destination_root / namespace
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".files") as listing:
            listing.write("\n".join(paths) + "\n")
            listing.flush()
            result = subprocess.run([
                "rsync", "-a", "--partial", "--files-from=" + listing.name,
                rsync_source(manifest["source_host"], source_root), str(destination) + "/",
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        common.require(result.returncode == 0, "rsync failed for {}: {}".format(namespace, result.stderr[-2000:]))
    return {"closure_root": str(destination_root), "namespaces": sorted(manifest["source_roots"])}


def _local_files(root):
    found = {}
    for namespace_dir in sorted(path for path in root.iterdir()):
        common.require(namespace_dir.is_dir(), "unexpected closure entry: {}".format(namespace_dir))
        for path in namespace_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError("closure symlink rejected: {}".format(path))
            if path.is_file():
                found[(namespace_dir.name, path.relative_to(namespace_dir).as_posix())] = path
    return found


def _label_totals(label_root, assembly_ids):
    totals = Counter()
    for assembly_id in assembly_ids:
        directory = label_root / assembly_id
        summary = common.read_json(directory / "summary.json")
        common.require(summary.get("complete") is True and summary.get("assembly_id") == assembly_id, "label summary incomplete: " + assembly_id)
        listed = {"summary.json"}
        for key in ("candidate_files", "state_pose_files"):
            for row in summary[key]:
                common.require(common.sha256_file(directory / row["path"]) == row["sha256"], "label file hash differs: " + row["path"])
                listed.add(row["path"])
        present = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
        common.require(present == listed, "label directory has unlisted or missing files: " + assembly_id)
        totals["nonterminal_states"] += len(summary["states"])
        totals["terminal_states"] += len(summary["terminal_states"])
        totals["candidate_files"] += len(summary["candidate_files"])
    return dict(totals)


def verify(data_root):
    """Verify every transferred byte and the semantic bindings among closure artifacts."""
    identity = common.implementation_identity(require_clean=True)
    manifest_path, manifest = _load_manifest(data_root)
    root = common.closure_root(data_root)
    expected = {(row["namespace"], row["path"]): row for row in manifest["files"]}
    found = _local_files(root)
    unlisted = sorted("{}:{}".format(*key) for key in set(found) - set(expected))
    missing = sorted("{}:{}".format(*key) for key in set(expected) - set(found))
    common.require(not unlisted, "closure contains unlisted files: {}".format(unlisted[:10]))
    common.require(not missing, "closure is missing files: {}".format(missing[:10]))
    for key, row in expected.items():
        common.require_development_path(row["path"])
        path = found[key]
        common.require(path.stat().st_size == row["bytes"], "closure byte count differs: {}".format(row["path"]))
        common.require(common.sha256_file(path) == row["sha256"], "closure hash differs: {}".format(row["path"]))
    for path, digest in FROZEN_SHA256.items():
        common.require(common.sha256_file(root / "fast-data" / path) == digest, "frozen input hash differs: " + path)

    data = root / "fast-data"
    audit = common.read_json(data / GRAPH_AUDIT)
    common.require(small_validation_ids(audit) == manifest["small_validation_assembly_ids"], "small-validation IDs differ")
    inventory = {row["path"]: row for row in audit["inventory"]}
    shard_files = {path.relative_to(data / GRAPH_ROOT).as_posix() for path in (data / GRAPH_ROOT / "shards").rglob("*") if path.is_file()}
    common.require(shard_files == set(inventory), "graph shard tree differs from audit inventory")
    for relative, row in inventory.items():
        common.require(common.sha256_file(data / GRAPH_ROOT / relative) == row["sha256"], "graph shard hash differs: " + relative)
    common.require(common.sha256_file(data / GRAPH_ROOT / audit["validation_folds_path"]) == audit["validation_folds_sha256"],
                   "validation fold hash differs")
    pilot_names = sorted(path.name for path in (data / GRAPH_ROOT / "pilots").iterdir())
    common.require(pilot_names == ["00011.jsonl", "00014.jsonl", "00101.jsonl"], "graph pilot shards differ")

    fixture_ids = manifest["small_validation_assembly_ids"] + [REPLAY_FIXTURE]
    label_totals = _label_totals(data / LABEL_ROOT, manifest["small_validation_assembly_ids"])
    common.require(label_totals == EXPECTED_LABEL_TOTALS, "small-validation label totals differ: {}".format(label_totals))
    _label_totals(data / LABEL_ROOT, [REPLAY_FIXTURE])

    by_assembly = {row["assembly_id"]: row for row in audit["inventory"]}
    geometry = []
    for assembly_id in fixture_ids:
        shard = common.read_jsonl(data / GRAPH_ROOT / by_assembly[assembly_id]["path"])
        recorded = {record["input_sha256"] for record in shard[1:-1]}
        common.require(len(recorded) == 1, "graph shard input hash is not unique: " + assembly_id)
        actual = production._hash_assembly(root / "asap-data" / GEOMETRY_ROOT / assembly_id)
        common.require(actual == next(iter(recorded)), "raw geometry differs from graph input hash: " + assembly_id)
        geometry.append({"assembly_id": assembly_id, "input_sha256": actual})

    for trace in REPLAY_TRACES:
        sidecar = (data / (trace + ".sha256")).read_text(encoding="ascii").split()
        common.require(sidecar[0] == FROZEN_SHA256[trace], "trace sidecar differs: " + trace)
        rows = common.read_jsonl(data / trace)
        common.require(rows[0].get("assembly_id") == REPLAY_FIXTURE and rows[-1].get("complete") is True
                       and rows[-1].get("test_rows_loaded") == 0, "replay trace differs: " + trace)

    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    report = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "passed",
        "run_id": common.RUN_ID,
        "verified_at": common.utc_now(),
        "implementation": identity,
        "transfer_manifest_sha256": common.sha256_file(manifest_path),
        "file_count": len(expected),
        "total_bytes": sum(row["bytes"] for row in expected.values()),
        "unlisted_file_count": 0,
        "missing_file_count": 0,
        "frozen_hash_checks_passed": len(FROZEN_SHA256),
        "graph_shards_verified": len(inventory),
        "small_validation_label_totals": label_totals,
        "geometry_input_hash_matches": len(geometry),
        "replay_traces_verified": len(REPLAY_TRACES),
        "closure_read_only": True,
        "test_paths_present": 0,
        "test_rows_loaded": 0,
    }
    path = common.run_root(data_root) / "transfer-verification.json"
    digest = common.write_json_once(path, report)
    return {"verification": str(path), "sha256": digest, **{key: report[key] for key in ("status", "file_count", "total_bytes")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("command", choices=("plan", "transfer", "verify"))
    args = parser.parse_args(argv)
    result = {"plan": plan, "transfer": transfer, "verify": verify}[args.command](args.data_root)
    print(common.canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
