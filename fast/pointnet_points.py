"""Materialize the frozen development-only PointNet point cache."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from assets.load import load_assembly
from fast import graph_contract, graph_materialize
from plan_sequence import production_sim_labels as production


SPEC_SCHEMA_VERSION = "fast.pointnet-spec.v1"
SIDECAR_SCHEMA_VERSION = "fast.point-shard.v1"
AUDIT_SCHEMA_VERSION = "fast.point-audit.v1"


def _read_json(path):
    return graph_materialize._read_json(path)


def _write_json(path, value):
    graph_materialize._write_json(path, value)


def load_spec(path):
    path = Path(path).resolve()
    spec = _read_json(path)
    graph_contract._require(spec.get("schema_version") == SPEC_SCHEMA_VERSION, "invalid PointNet spec schema")
    graph_contract._require(spec.get("status") == "frozen", "PointNet spec is not frozen")
    points = spec.get("points", {})
    graph_contract._require(
        points.get("contract") == "fast.points.v1"
        and points.get("root_seed") == 20260924
        and points.get("count_per_part") == 1000
        and points.get("dtype") == "<f4"
        and points.get("layout") == "C",
        "point contract differs",
    )
    graph_contract._require(spec.get("test_access", {}).get("development_test_rows") == 0, "test access boundary differs")
    return spec


def _canonical_hash(value):
    return hashlib.sha256(production._canonical_json(value).encode("utf-8")).hexdigest()


def part_seed(contract, root_seed, input_sha256, part_id):
    payload = [contract, root_seed, input_sha256, part_id]
    return int(_canonical_hash(payload)[:8], 16)


def _sample_surface(mesh, count, rng):
    """trimesh-3.10.8 sample_surface with an explicit local RNG."""
    weights = np.cumsum(mesh.area_faces)
    graph_contract._require(len(weights) and np.isfinite(weights[-1]) and weights[-1] > 0, "part has no sampleable surface")
    face_index = np.searchsorted(weights, rng.random_sample(count) * weights[-1])
    origins = mesh.vertices[mesh.faces[:, 0]]
    vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    vectors -= np.tile(origins, (1, 2)).reshape((-1, 2, 3))
    origins, vectors = origins[face_index], vectors[face_index]
    lengths = rng.random_sample((count, 2, 1))
    reflected = lengths.sum(axis=1).reshape(-1) > 1.0
    lengths[reflected] -= 1.0
    return (vectors * np.abs(lengths)).sum(axis=1) + origins


def sample_assembly(assembly_dir, input_sha256, points_spec):
    assembly_dir = Path(assembly_dir)
    assembly = load_assembly(str(assembly_dir), transform="final")
    part_ids = sorted(assembly)
    graph_contract._require(part_ids, "assembly has no parts")
    vertices = np.vstack([np.asarray(assembly[part]["mesh"].vertices) for part in part_ids])
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2.0
    extent = float((upper - lower).max())
    graph_contract._require(np.isfinite(center).all() and np.isfinite(extent) and extent > 0, "assembly AABB is invalid")
    seeds, samples = {}, []
    for part_id in part_ids:
        seed = part_seed(points_spec["contract"], points_spec["root_seed"], input_sha256, part_id)
        seeds[part_id] = seed
        points = _sample_surface(
            assembly[part_id]["mesh"], points_spec["count_per_part"], np.random.RandomState(seed)
        )
        samples.append(2.0 * (points - center) / extent)
    tensor = np.ascontiguousarray(np.asarray(samples, dtype=np.dtype("<f4")))
    graph_contract._require(
        tensor.shape == (len(part_ids), points_spec["count_per_part"], 3)
        and tensor.dtype == np.dtype("<f4")
        and tensor.flags.c_contiguous
        and np.isfinite(tensor).all(),
        "point tensor contract differs",
    )
    return tensor, part_ids, seeds, {
        "vertex_aabb_lower": lower.tolist(),
        "vertex_aabb_upper": upper.tolist(),
        "center": center.tolist(),
        "max_extent": extent,
        "formula": points_spec["normalization"],
    }


def _atomic_npy(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, tensor, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _prepare(spec_path, source_root, data_root, output_root=None):
    spec_path = Path(spec_path).resolve()
    repository_root = spec_path.parent.parent
    source_root, data_root = Path(source_root).resolve(), Path(data_root).resolve()
    graph_contract._require(source_root.is_dir() and data_root.is_dir(), "source or data root is missing")
    spec = load_spec(spec_path)
    graph_contract._require(
        np.__version__ == spec["environment"]["numpy"]
        and trimesh.__version__ == spec["environment"]["trimesh"],
        "point-cache NumPy/trimesh environment differs",
    )
    for name in ("dataset_protocol", "ranker_spec"):
        item = spec["inputs"][name]
        path = (repository_root / item["path"]).resolve()
        graph_contract._require(production._is_within(path, repository_root), "{} escapes repository".format(name))
        graph_contract._require(production._sha256_file(path) == item["sha256"], "{} hash differs".format(name))
    graph_contract._require(
        production._sha256_file(repository_root / "plan_sequence" / "generator" / "network" / "model.py")
        == spec["inputs"]["asap_pointnet_source_sha256"],
        "released ASAP PointNet source hash differs",
    )
    graph_root = (data_root / spec["artifacts"]["ranker_graphs"]).resolve()
    graph_audit_path = graph_root / "audit.json"
    graph_contract._require(production._sha256_file(graph_audit_path) == spec["inputs"]["ranker_graph_audit_sha256"], "ranker graph audit hash differs")
    graph_audit = _read_json(graph_audit_path)
    graph_contract._require(
        graph_audit.get("status") == "passed"
        and graph_audit.get("test_rows_loaded") == 0
        and all(row.get("dataset_split") in graph_contract.DEVELOPMENT_SPLITS for row in graph_audit["inventory"]),
        "ranker development graph audit differs",
    )
    output_root = Path(output_root).resolve() if output_root else (data_root / spec["artifacts"]["pointnet_points"]).resolve()
    graph_contract._require(not production._is_within(output_root, source_root), "point output is inside source root")
    return {
        "spec": spec, "spec_path": spec_path, "spec_sha256": production._sha256_file(spec_path),
        "repository_root": repository_root, "source_root": source_root, "data_root": data_root,
        "graph_root": graph_root, "graph_audit": graph_audit, "graph_audit_path": graph_audit_path,
        "output_root": output_root,
    }


def _jobs(prepared):
    jobs = []
    for index, row in enumerate(prepared["graph_audit"]["inventory"]):
        graph_contract._require(row["dataset_split"] in graph_contract.DEVELOPMENT_SPLITS and row["source_split"] == "training_assembly", "point cache rejects non-development metadata")
        graph_path = prepared["graph_root"] / row["path"]
        graph_contract._require(production._sha256_file(graph_path) == row["sha256"], "ranker graph shard hash differs")
        with graph_path.open("r", encoding="utf-8") as stream:
            header = json.loads(next(stream))
        graph_contract._require(header.get("record_type") == "header" and header.get("dataset_split") == row["dataset_split"], "graph header differs")
        jobs.append({
            "process": index, "dataset_split": row["dataset_split"], "source_split": row["source_split"],
            "assembly_id": row["assembly_id"], "tier": row["tier"], "input_sha256": header["input_sha256"],
        })
    return jobs


def prepare(spec_path, source_root, data_root, output_root=None):
    prepared = _prepare(spec_path, source_root, data_root, output_root)
    jobs = _jobs(prepared)
    prepared["output_root"].mkdir(parents=True, exist_ok=True)
    table = prepared["output_root"] / "development-jobs.tsv"
    lines = ["process\tdataset_split\tsource_split\tassembly_id\ttier\tinput_sha256"]
    lines += ["{process}\t{dataset_split}\t{source_split}\t{assembly_id}\t{tier}\t{input_sha256}".format(**row) for row in jobs]
    descriptor, temporary = tempfile.mkstemp(prefix="." + table.name, dir=str(table.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        os.replace(temporary, table)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report = {
        "schema_version": "fast.point-setup.v1", "status": "passed",
        "spec_sha256": prepared["spec_sha256"], "ranker_graph_audit_sha256": production._sha256_file(prepared["graph_audit_path"]),
        "job_table": table.name, "job_table_sha256": production._sha256_file(table),
        "assembly_count": len(jobs), "train_assembly_count": sum(row["dataset_split"] == "train" for row in jobs),
        "validation_assembly_count": sum(row["dataset_split"] == "validation" for row in jobs), "test_rows_loaded": 0,
    }
    _write_json(prepared["output_root"] / "setup.json", report)
    return report


def _paths(prepared, job, pilot):
    root = prepared["output_root"] / ("pilots" if pilot else "shards")
    if not pilot:
        root = root / job["dataset_split"] / job["source_split"]
    return root / (job["assembly_id"] + ".npy"), root / (job["assembly_id"] + ".json")


def _job(prepared, assembly_id=None, job_index=None):
    jobs = _jobs(prepared)
    if job_index is not None:
        graph_contract._require(0 <= job_index < len(jobs), "job index is out of range")
        return jobs[job_index]
    matches = [row for row in jobs if row["assembly_id"] == assembly_id]
    graph_contract._require(len(matches) == 1, "development assembly is absent or duplicated")
    return matches[0]


def _validate_shard(tensor_path, sidecar_path, prepared, job):
    sidecar = _read_json(sidecar_path)
    graph_contract._require(sidecar.get("schema_version") == SIDECAR_SCHEMA_VERSION and sidecar.get("complete") is True, "point sidecar envelope differs")
    expected_identity = {name: job[name] for name in ("dataset_split", "source_split", "assembly_id", "tier", "input_sha256")}
    graph_contract._require(sidecar.get("identity") == expected_identity, "point sidecar identity differs")
    graph_contract._require(sidecar.get("spec_sha256") == prepared["spec_sha256"], "point sidecar spec differs")
    contract = prepared["spec"]["points"]
    graph_contract._require(
        sidecar.get("cache_key") == _canonical_hash({"contract": contract, "input_sha256": job["input_sha256"]})
        and sidecar.get("contract_sha256") == _canonical_hash(contract)
        and sidecar.get("sampler_source_sha256") == production._sha256_file(Path(__file__))
        and sidecar.get("numpy_version") == np.__version__
        and sidecar.get("trimesh_version") == trimesh.__version__,
        "point-cache contract provenance differs",
    )
    graph_contract._require(production._sha256_file(tensor_path) == sidecar.get("tensor_sha256"), "point tensor hash differs")
    tensor = np.load(tensor_path, allow_pickle=False)
    graph_contract._require(
        tensor.shape == (len(sidecar["part_ids"]), prepared["spec"]["points"]["count_per_part"], 3)
        and tensor.dtype == np.dtype("<f4") and tensor.flags.c_contiguous and np.isfinite(tensor).all(),
        "point tensor contract differs",
    )
    part_ids = sorted(path.stem for path in (prepared["source_root"] / job["source_split"] / job["assembly_id"]).glob("*.obj"))
    graph_contract._require(sidecar["part_ids"] == part_ids, "point sidecar part IDs differ")
    graph_contract._require(
        sidecar.get("part_seeds") == {
            part_id: part_seed(contract["contract"], contract["root_seed"], job["input_sha256"], part_id)
            for part_id in part_ids
        }
        and sidecar.get("shape") == list(tensor.shape)
        and sidecar.get("dtype") == tensor.dtype.str
        and sidecar.get("normalization", {}).get("formula") == contract["normalization"],
        "point sidecar tensor metadata differs",
    )
    source_hashes = {part_id: production._sha256_file(prepared["source_root"] / job["source_split"] / job["assembly_id"] / (part_id + ".obj")) for part_id in part_ids}
    graph_contract._require(sidecar["source_obj_sha256"] == source_hashes, "source OBJ hashes differ")
    return sidecar


def materialize(spec_path, source_root, data_root, output_root=None, assembly_id=None, job_index=None, pilot=False):
    prepared = _prepare(spec_path, source_root, data_root, output_root)
    job = _job(prepared, assembly_id, job_index)
    if pilot:
        graph_contract._require(job["assembly_id"] in prepared["spec"]["points"]["pilot_training_assemblies"], "assembly is not in the frozen pilot")
    tensor_path, sidecar_path = _paths(prepared, job, pilot)
    if tensor_path.exists() or sidecar_path.exists():
        graph_contract._require(tensor_path.is_file() and sidecar_path.is_file(), "partial point shard exists")
        sidecar = _validate_shard(tensor_path, sidecar_path, prepared, job)
        return {"status": "already_complete", "tensor_sha256": sidecar["tensor_sha256"], "test_rows_loaded": 0}
    assembly_dir = prepared["source_root"] / job["source_split"] / job["assembly_id"]
    graph_contract._require(assembly_dir.is_dir(), "source assembly is missing")
    graph_contract._require(production._hash_assembly(assembly_dir) == job["input_sha256"], "source assembly hash differs")
    tensor, part_ids, seeds, normalization = sample_assembly(assembly_dir, job["input_sha256"], prepared["spec"]["points"])
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npy(tensor_path, tensor)
    contract = prepared["spec"]["points"]
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION, "complete": True,
        "spec_sha256": prepared["spec_sha256"],
        "identity": {name: job[name] for name in ("dataset_split", "source_split", "assembly_id", "tier", "input_sha256")},
        "cache_key": _canonical_hash({"contract": contract, "input_sha256": job["input_sha256"]}),
        "contract_sha256": _canonical_hash(contract), "part_ids": part_ids,
        "part_seeds": seeds, "normalization": normalization,
        "source_obj_sha256": {part_id: production._sha256_file(assembly_dir / (part_id + ".obj")) for part_id in part_ids},
        "sampler_source_sha256": production._sha256_file(Path(__file__)),
        "numpy_version": np.__version__, "trimesh_version": trimesh.__version__,
        "shape": list(tensor.shape), "dtype": tensor.dtype.str, "tensor": tensor_path.name,
        "tensor_sha256": production._sha256_file(tensor_path), "test_rows_loaded": 0,
    }
    _write_json(sidecar_path, sidecar)
    _validate_shard(tensor_path, sidecar_path, prepared, job)
    return {"status": "complete", "tensor_sha256": sidecar["tensor_sha256"], "test_rows_loaded": 0}


def audit(spec_path, source_root, data_root, output_root=None, pilot=False):
    prepared = _prepare(spec_path, source_root, data_root, output_root)
    jobs = _jobs(prepared)
    if pilot:
        wanted = set(prepared["spec"]["points"]["pilot_training_assemblies"])
        jobs = [row for row in jobs if row["assembly_id"] in wanted]
    inventory, point_count, byte_count = [], 0, 0
    for job in jobs:
        tensor_path, sidecar_path = _paths(prepared, job, pilot)
        graph_contract._require(tensor_path.is_file() and sidecar_path.is_file(), "point shard is missing")
        sidecar = _validate_shard(tensor_path, sidecar_path, prepared, job)
        if pilot:
            regenerated, part_ids, seeds, normalization = sample_assembly(
                prepared["source_root"] / job["source_split"] / job["assembly_id"],
                job["input_sha256"], prepared["spec"]["points"],
            )
            graph_contract._require(
                np.array_equal(regenerated, np.load(tensor_path, allow_pickle=False))
                and part_ids == sidecar["part_ids"]
                and seeds == sidecar["part_seeds"]
                and normalization == sidecar["normalization"],
                "independent pilot regeneration differs",
            )
        point_count += int(np.prod(sidecar["shape"][:-1]))
        byte_count += tensor_path.stat().st_size
        inventory.append({**{name: job[name] for name in ("dataset_split", "source_split", "assembly_id", "tier", "input_sha256")},
                          "tensor_path": tensor_path.relative_to(prepared["output_root"]).as_posix(), "tensor_sha256": sidecar["tensor_sha256"],
                          "sidecar_path": sidecar_path.relative_to(prepared["output_root"]).as_posix(), "sidecar_sha256": production._sha256_file(sidecar_path),
                          "part_count": len(sidecar["part_ids"])})
    if not pilot:
        graph_contract._require(len(inventory) == prepared["graph_audit"]["assembly_count"], "point audit assembly count differs")
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION, "status": "passed", "pilot": bool(pilot),
        "spec_sha256": prepared["spec_sha256"], "ranker_graph_audit_sha256": production._sha256_file(prepared["graph_audit_path"]),
        "assembly_count": len(inventory), "part_count": sum(row["part_count"] for row in inventory),
        "sample_count": point_count, "tensor_bytes": byte_count, "inventory": inventory, "test_rows_loaded": 0,
        "independent_regeneration_verified": bool(pilot),
    }
    _write_json(prepared["output_root"] / ("pilot-audit.json" if pilot else "audit.json"), report)
    return report


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(root / "configs" / "pointnet.json"))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    materialize_parser = commands.add_parser("materialize")
    identity = materialize_parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--assembly-id")
    identity.add_argument("--job-index", type=int)
    materialize_parser.add_argument("--pilot", action="store_true")
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    common = (args.spec, args.source_root, args.data_root, args.output_root)
    if args.command == "prepare":
        result = prepare(*common)
    elif args.command == "materialize":
        result = materialize(*common, args.assembly_id, args.job_index, args.pilot)
    else:
        result = audit(*common, args.pilot)
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
