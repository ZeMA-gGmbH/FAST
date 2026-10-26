"""Extract and audit leakage-safe scalar FAST features from the frozen index."""

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from assets.load import load_assembly
from plan_sequence import build_dataset_index as dataset_index
from plan_sequence import production_sim_labels as production


CONFIG_SCHEMA_VERSION = "fast.feature-manifest.v1"
SHARD_SCHEMA_VERSION = "fast.feature-shard.v1"
RECORD_SCHEMA_VERSION = "fast.candidate-features.v1"
BUILD_SCHEMA_VERSION = "fast.feature-build.v1"
GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float64)
SUPPORT_NORMAL = -GRAVITY
EPSILON = 1e-12

CANDIDATE_FEATURES = (
    "geometry__log_volume_fraction",
    "geometry__surface_area_over_scale2",
    "geometry__compactness",
    "geometry__extent_min_over_max",
    "geometry__extent_mid_over_max",
    "geometry__inertia_min_fraction",
    "geometry__inertia_mid_fraction",
    "geometry__inertia_max_fraction",
    "state__remaining_part_fraction",
    "state__remaining_volume_fraction",
    "state__candidate_volume_fraction",
    "state__part_count_over_70",
    "pose_support__candidate_com_x",
    "pose_support__candidate_com_y",
    "pose_support__candidate_com_z",
    "pose_support__candidate_outside_distance",
    "pose_support__candidate_height",
    "pose_support__candidate_bottom_gap",
    "pose_support__candidate_is_support",
    "pose_support__state_com_height",
    "pose_support__support_part_fraction",
    "pose_support__approx_support_margin",
    "topology__candidate_degree_fraction",
    "topology__candidate_component_fraction",
    "topology__candidate_is_articulation",
    "topology__support_hops_fraction",
    "topology__state_component_fraction",
    "topology__state_edge_density",
    "child_stability__remaining_volume_fraction",
    "child_stability__com_shift_x",
    "child_stability__com_shift_y",
    "child_stability__com_shift_z",
    "child_stability__com_height",
    "child_stability__support_part_fraction",
    "child_stability__approx_support_margin",
    "child_stability__support_margin_change",
    "child_stability__component_fraction",
    "child_stability__components_added_fraction",
    "child_stability__supported_volume_fraction",
    "child_stability__unsupported_volume_fraction",
    "child_stability__removed_support",
)

ACTION_FEATURES = (
    "direction__x",
    "direction__y",
    "direction__z",
    "direction__gravity_alignment",
    "direction__support_normal_alignment",
    "directional_clearance__outward_projection",
    "directional_clearance__candidate_extent",
    "directional_clearance__boundary_gap",
    "directional_clearance__exit_distance",
    "directional_clearance__projected_area_fraction",
    "directional_clearance__blocker_count_fraction",
    "directional_clearance__blocker_volume_fraction",
    "directional_clearance__blocker_overlap_fraction",
    "directional_clearance__minimum_blocker_gap",
    "directional_clearance__contact_ahead_fraction",
    "directional_clearance__has_no_aabb_blocker",
)

FORBIDDEN_FEATURE_TERMS = (
    "label",
    "status",
    "timeout",
    "runtime",
    "collision",
    "force",
    "path",
    "held",
    "convergence",
    "displacement",
    "reason",
)
FEATURE_GROUPS = (
    "geometry",
    "state",
    "pose_support",
    "topology",
    "child_stability",
    "direction",
    "directional_clearance",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _resolve_under(root, value, name):
    relative = production._relative_path(value, name)
    path = (Path(root).resolve() / relative).resolve()
    _require(production._is_within(path, root), "{} escapes its root".format(name))
    return path


def _feature_groups(names):
    return sorted({name.split("__", 1)[0] for name in names})


def _read_config(manifest_path, source_root, data_root):
    manifest_path = Path(manifest_path).resolve()
    source_root = Path(source_root).resolve()
    data_root = Path(data_root).resolve()
    _require(manifest_path.is_file(), "feature manifest is missing")
    _require(source_root.is_dir(), "source root is missing")
    _require(data_root.is_dir(), "data root is missing")
    config = _read_json(manifest_path)
    production._require_fields(
        config,
        {
            "schema_version",
            "phase",
            "protocol",
            "dataset_root",
            "output_root",
            "screen_jobs",
            "support_tolerance_fraction",
            "feature_groups",
        },
        "scalar feature manifest",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA_VERSION, "invalid feature schema")
    tolerance = config["support_tolerance_fraction"]
    _require(
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and 0 < tolerance < 0.1,
        "support tolerance must be in (0, 0.1)",
    )
    _require(config["feature_groups"] == list(FEATURE_GROUPS), "feature groups differ")

    repository_root = manifest_path.parent.parent
    protocol_path = _resolve_under(repository_root, config["protocol"], "protocol")
    protocol = _read_json(protocol_path)
    _require(protocol.get("status") == "frozen", "evaluation protocol is not frozen")
    _require(
        protocol["dataset"]["external_root"] == config["dataset_root"],
        "feature manifest dataset differs from the frozen protocol",
    )
    dataset_root = _resolve_under(data_root, config["dataset_root"], "dataset_root")
    output_root = _resolve_under(data_root, config["output_root"], "output_root")
    _require(dataset_root.is_dir(), "frozen dataset root is missing")
    _require(not production._is_within(dataset_root, output_root), "feature output overlaps input")
    for name, artifact in protocol["dataset"]["artifacts"].items():
        path = _resolve_under(dataset_root, artifact["path"], "{} artifact".format(name))
        _require(path.is_file(), "{} artifact is missing".format(name))
        _require(
            production._sha256_file(path) == artifact["sha256"],
            "{} artifact hash differs".format(name),
        )
    return {
        "config": config,
        "manifest_path": manifest_path,
        "manifest_sha256": production._sha256_file(manifest_path),
        "protocol_path": protocol_path,
        "protocol": protocol,
        "protocol_sha256": production._sha256_file(protocol_path),
        "source_root": source_root,
        "data_root": data_root,
        "dataset_root": dataset_root,
        "output_root": output_root,
        "extractor_sha256": production._sha256_file(Path(__file__)),
    }


def _load_contact_graph(path, part_ids):
    payload = _read_json(path)
    graph = nx.Graph()
    graph.add_nodes_from(part_ids)
    for node in payload.get("nodes", []):
        graph.add_node(str(node["id"]))
    for edge in payload.get("links", []):
        graph.add_edge(str(edge["source"]), str(edge["target"]))
    _require(set(graph.nodes) == set(part_ids), "contact graph part IDs differ")
    return graph


def _as_mesh(mesh, part_id):
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    _require(isinstance(mesh, trimesh.Trimesh), "part {} is not a mesh".format(part_id))
    return mesh


def _load_parts(assembly_dir):
    assembly = load_assembly(assembly_dir, transform="final")
    parts = {}
    for part_id, item in assembly.items():
        mesh = _as_mesh(item["mesh"], part_id)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        _require(vertices.ndim == 2 and vertices.shape[1] == 3, "invalid vertices")
        volume = abs(float(mesh.volume))
        area = float(mesh.area)
        center = np.asarray(mesh.center_mass, dtype=np.float64)
        inertia = np.linalg.eigvalsh(np.asarray(mesh.moment_inertia, dtype=np.float64))
        extent = np.ptp(vertices, axis=0)
        _require(
            volume > EPSILON
            and area > EPSILON
            and np.isfinite(vertices).all()
            and np.isfinite(center).all()
            and np.isfinite(inertia).all(),
            "part {} has invalid physical geometry".format(part_id),
        )
        parts[part_id] = {
            "vertices": vertices,
            "volume": volume,
            "area": area,
            "center": center,
            "inertia": np.maximum(inertia, 0.0),
            "extent": np.maximum(extent, EPSILON),
        }
    return parts


def _pose_key(pose):
    matrix = np.eye(4) if pose is None else np.asarray(pose, dtype=np.float64)
    _require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), "invalid pose")
    return tuple(matrix.reshape(-1)), matrix


def _posed_parts(static_parts, pose, cache):
    key, matrix = _pose_key(pose)
    if key not in cache:
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        posed = {}
        for part_id, part in static_parts.items():
            vertices = part["vertices"].dot(rotation.T) + translation
            posed[part_id] = {
                "minimum": vertices.min(axis=0),
                "maximum": vertices.max(axis=0),
                "center": part["center"].dot(rotation.T) + translation,
            }
        cache[key] = posed
    return cache[key]


def _weighted_center(part_ids, posed, static_parts):
    volumes = np.array([static_parts[part_id]["volume"] for part_id in part_ids])
    centers = np.array([posed[part_id]["center"] for part_id in part_ids])
    return np.average(centers, axis=0, weights=volumes)


def _support_margin(point_xy, support_ids, posed, scale):
    if not support_ids:
        return -1.0
    corners = []
    for part_id in support_ids:
        minimum = posed[part_id]["minimum"]
        maximum = posed[part_id]["maximum"]
        corners.extend(
            (
                (minimum[0], minimum[1]),
                (minimum[0], maximum[1]),
                (maximum[0], minimum[1]),
                (maximum[0], maximum[1]),
            )
        )
    points = np.unique(np.asarray(corners, dtype=np.float64), axis=0)
    if len(points) < 3:
        return -1.0
    try:
        hull = ConvexHull(points)
    except QhullError:
        return -1.0
    equations = hull.equations
    norms = np.linalg.norm(equations[:, :2], axis=1)
    signed = -(equations[:, :2].dot(point_xy) + equations[:, 2]) / norms
    return float(signed.min() / scale)


def _state_summary(part_ids, posed, static_parts, graph, tolerance, reference_scale=None):
    _require(part_ids, "state contains no parts")
    minimum = np.min([posed[part_id]["minimum"] for part_id in part_ids], axis=0)
    maximum = np.max([posed[part_id]["maximum"] for part_id in part_ids], axis=0)
    scale = float(np.linalg.norm(maximum - minimum))
    _require(scale > EPSILON, "state has zero scale")
    support_scale = reference_scale or scale
    support_ids = [
        part_id
        for part_id in part_ids
        if posed[part_id]["minimum"][2] <= tolerance * support_scale
        and posed[part_id]["maximum"][2] >= -tolerance * support_scale
    ]
    volume = sum(static_parts[part_id]["volume"] for part_id in part_ids)
    center = _weighted_center(part_ids, posed, static_parts)
    subgraph = graph.subgraph(part_ids).copy()
    components = list(nx.connected_components(subgraph))
    supported = set()
    for component in components:
        if set(support_ids).intersection(component):
            supported.update(component)
    supported_volume = sum(static_parts[part_id]["volume"] for part_id in supported)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "bbox_center": 0.5 * (minimum + maximum),
        "scale": scale,
        "volume": volume,
        "center": center,
        "support_ids": support_ids,
        "support_margin": _support_margin(
            center[:2], support_ids, posed, support_scale
        ),
        "graph": subgraph,
        "components": components,
        "supported_volume_fraction": supported_volume / volume,
    }


def _component_fraction(graph, part_id):
    for component in nx.connected_components(graph):
        if part_id in component:
            return len(component) / max(len(graph), 1)
    return 0.0


def _support_hops(graph, part_id, support_ids):
    distances = []
    for support_id in support_ids:
        if nx.has_path(graph, part_id, support_id):
            distances.append(nx.shortest_path_length(graph, part_id, support_id))
    return min(distances) / max(len(graph) - 1, 1) if distances else 1.0


def _candidate_features(
    header, static_parts, posed, graph, full_volume, tolerance
):
    part_ids = [str(part_id) for part_id in header["parts"]]
    child_ids = [str(part_id) for part_id in header["parts_after"]]
    candidate_id = str(header["part_move"])
    _require(candidate_id in part_ids and candidate_id not in child_ids, "invalid candidate")
    parent = _state_summary(part_ids, posed, static_parts, graph, tolerance)
    child = _state_summary(
        child_ids, posed, static_parts, graph, tolerance, reference_scale=parent["scale"]
    )
    candidate = static_parts[candidate_id]
    posed_candidate = posed[candidate_id]
    scale = parent["scale"]
    volume_fraction = candidate["volume"] / parent["volume"]
    extents = np.sort(candidate["extent"])
    inertia_total = max(float(candidate["inertia"].sum()), EPSILON)
    inertia_fraction = candidate["inertia"] / inertia_total
    relative_com = (posed_candidate["center"] - parent["bbox_center"]) / scale
    com_shift = (child["center"] - parent["center"]) / scale
    parent_graph = parent["graph"]
    child_graph = child["graph"]
    articulation = set(nx.articulation_points(parent_graph)) if len(parent_graph) > 1 else set()
    parent_components = len(parent["components"])
    child_components = len(child["components"])
    n_parts = len(part_ids)
    child_n = len(child_ids)
    compactness = candidate["area"] / max(candidate["volume"] ** (2.0 / 3.0), EPSILON)
    features = {
        "geometry__log_volume_fraction": math.log(max(volume_fraction, EPSILON)),
        "geometry__surface_area_over_scale2": candidate["area"] / (scale * scale),
        "geometry__compactness": compactness,
        "geometry__extent_min_over_max": extents[0] / extents[2],
        "geometry__extent_mid_over_max": extents[1] / extents[2],
        "geometry__inertia_min_fraction": inertia_fraction[0],
        "geometry__inertia_mid_fraction": inertia_fraction[1],
        "geometry__inertia_max_fraction": inertia_fraction[2],
        "state__remaining_part_fraction": n_parts / len(static_parts),
        "state__remaining_volume_fraction": parent["volume"] / full_volume,
        "state__candidate_volume_fraction": volume_fraction,
        "state__part_count_over_70": n_parts / 70.0,
        "pose_support__candidate_com_x": relative_com[0],
        "pose_support__candidate_com_y": relative_com[1],
        "pose_support__candidate_com_z": relative_com[2],
        "pose_support__candidate_outside_distance": float(np.linalg.norm(relative_com)),
        "pose_support__candidate_height": posed_candidate["center"][2] / scale,
        "pose_support__candidate_bottom_gap": posed_candidate["minimum"][2] / scale,
        "pose_support__candidate_is_support": float(candidate_id in parent["support_ids"]),
        "pose_support__state_com_height": parent["center"][2] / scale,
        "pose_support__support_part_fraction": len(parent["support_ids"]) / n_parts,
        "pose_support__approx_support_margin": parent["support_margin"],
        "topology__candidate_degree_fraction": parent_graph.degree(candidate_id)
        / max(n_parts - 1, 1),
        "topology__candidate_component_fraction": _component_fraction(
            parent_graph, candidate_id
        ),
        "topology__candidate_is_articulation": float(candidate_id in articulation),
        "topology__support_hops_fraction": _support_hops(
            parent_graph, candidate_id, parent["support_ids"]
        ),
        "topology__state_component_fraction": parent_components / n_parts,
        "topology__state_edge_density": nx.density(parent_graph),
        "child_stability__remaining_volume_fraction": child["volume"] / full_volume,
        "child_stability__com_shift_x": com_shift[0],
        "child_stability__com_shift_y": com_shift[1],
        "child_stability__com_shift_z": com_shift[2],
        "child_stability__com_height": child["center"][2] / scale,
        "child_stability__support_part_fraction": len(child["support_ids"])
        / max(child_n, 1),
        "child_stability__approx_support_margin": child["support_margin"],
        "child_stability__support_margin_change": child["support_margin"]
        - parent["support_margin"],
        "child_stability__component_fraction": child_components / max(child_n, 1),
        "child_stability__components_added_fraction": (child_components - parent_components)
        / max(child_n, 1),
        "child_stability__supported_volume_fraction": child[
            "supported_volume_fraction"
        ],
        "child_stability__unsupported_volume_fraction": 1.0
        - child["supported_volume_fraction"],
        "child_stability__removed_support": float(
            candidate_id in parent["support_ids"] and not child["support_ids"]
        ),
    }
    _validate_features(features, CANDIDATE_FEATURES)
    return features, parent


def _interval_overlap(first_min, first_max, second_min, second_max):
    return max(0.0, min(first_max, second_max) - max(first_min, second_min))


def _action_features(action, candidate_id, part_ids, posed, static_parts, graph, parent):
    direction = np.asarray(action["direction"], dtype=np.float64)
    _require(
        direction.shape == (3,) and np.count_nonzero(direction) == 1,
        "action direction is not principal",
    )
    axis = int(np.flatnonzero(direction)[0])
    sign = float(direction[axis])
    orthogonal = [index for index in range(3) if index != axis]
    candidate = posed[candidate_id]
    scale = parent["scale"]
    state_minimum = parent["minimum"]
    state_maximum = parent["maximum"]
    projected_state_area = np.prod(state_maximum[orthogonal] - state_minimum[orthogonal])
    projected_candidate_area = np.prod(
        candidate["maximum"][orthogonal] - candidate["minimum"][orthogonal]
    )
    blockers = []
    contact_ahead = 0
    for other_id in part_ids:
        if other_id == candidate_id:
            continue
        other = posed[other_id]
        forward = float(np.dot(other["center"] - candidate["center"], direction))
        overlap = np.prod(
            [
                _interval_overlap(
                    candidate["minimum"][index],
                    candidate["maximum"][index],
                    other["minimum"][index],
                    other["maximum"][index],
                )
                for index in orthogonal
            ]
        )
        if forward > 0 and graph.has_edge(candidate_id, other_id):
            contact_ahead += 1
        if forward <= 0 or overlap <= 0:
            continue
        gap = (
            other["minimum"][axis] - candidate["maximum"][axis]
            if sign > 0
            else candidate["minimum"][axis] - other["maximum"][axis]
        )
        blockers.append((other_id, gap, overlap))
    if sign > 0:
        boundary_gap = state_maximum[axis] - candidate["maximum"][axis]
        exit_distance = state_maximum[axis] - candidate["minimum"][axis]
    else:
        boundary_gap = candidate["minimum"][axis] - state_minimum[axis]
        exit_distance = candidate["maximum"][axis] - state_minimum[axis]
    count_denominator = max(len(part_ids) - 1, 1)
    features = {
        "direction__x": direction[0],
        "direction__y": direction[1],
        "direction__z": direction[2],
        "direction__gravity_alignment": float(np.dot(direction, GRAVITY)),
        "direction__support_normal_alignment": float(np.dot(direction, SUPPORT_NORMAL)),
        "directional_clearance__outward_projection": float(
            np.dot(candidate["center"] - parent["center"], direction) / scale
        ),
        "directional_clearance__candidate_extent": (
            candidate["maximum"][axis] - candidate["minimum"][axis]
        )
        / scale,
        "directional_clearance__boundary_gap": boundary_gap / scale,
        "directional_clearance__exit_distance": exit_distance / scale,
        "directional_clearance__projected_area_fraction": projected_candidate_area
        / max(projected_state_area, EPSILON),
        "directional_clearance__blocker_count_fraction": len(blockers)
        / count_denominator,
        "directional_clearance__blocker_volume_fraction": sum(
            static_parts[part_id]["volume"] for part_id, _, _ in blockers
        )
        / parent["volume"],
        "directional_clearance__blocker_overlap_fraction": sum(
            overlap for _, _, overlap in blockers
        )
        / max(projected_state_area, EPSILON),
        "directional_clearance__minimum_blocker_gap": min(
            (gap / scale for _, gap, _ in blockers), default=1.0
        ),
        "directional_clearance__contact_ahead_fraction": contact_ahead
        / count_denominator,
        "directional_clearance__has_no_aabb_blocker": float(not blockers),
    }
    _validate_features(features, ACTION_FEATURES)
    return features


def _validate_features(features, expected_names):
    _require(
        isinstance(features, dict) and set(features) == set(expected_names),
        "feature names differ",
    )
    for name, value in features.items():
        _require(
            not any(term in name.lower() for term in FORBIDDEN_FEATURE_TERMS),
            "forbidden feature name: {}".format(name),
        )
        _require(
            isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value)),
            "feature {} is not finite".format(name),
        )


def _target(value, valid):
    _require(type(valid) is bool, "target validity must be boolean")
    _require(type(value) in (bool, type(None)), "target must be boolean or null")
    _require(valid == (type(value) is bool), "target and validity differ")
    return {"value": value, "valid": valid}


def _joint_target(removability, stability):
    if removability == {"value": False, "valid": True} or stability == {
        "value": False,
        "valid": True,
    }:
        return {"value": False, "valid": True}
    if removability == {"value": True, "valid": True} and stability == {
        "value": True,
        "valid": True,
    }:
        return {"value": True, "valid": True}
    return {"value": None, "valid": False}


def _candidate_record(
    row, header, stability, actions, static_parts, posed, graph, full_volume, tolerance
):
    candidate_id = "{}:{}:{}:{}:{}".format(
        row["source_split"],
        row["assembly_id"],
        header["state_id"],
        header["pose_id"],
        header["part_move"],
    )
    features, parent = _candidate_features(
        header, static_parts, posed, graph, full_volume, tolerance
    )
    part_ids = [str(part_id) for part_id in header["parts"]]
    part_move = str(header["part_move"])
    action_records = []
    for action in actions:
        action_records.append(
            {
                "action_id": "{}:{}".format(candidate_id, action["direction_id"]),
                "direction_id": action["direction_id"],
                "features": _action_features(
                    action, part_move, part_ids, posed, static_parts, graph, parent
                ),
                "targets": {
                    "R": _target(action.get("R"), type(action.get("R")) is bool),
                    "Y": _target(action.get("Y"), type(action.get("Y")) is bool),
                },
            }
        )
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_type": "candidate",
        "query_id": "{}:{}:{}:{}".format(
            row["source_split"], row["assembly_id"], header["state_id"], header["pose_id"]
        ),
        "candidate_id": candidate_id,
        "dataset_split": row["dataset_split"],
        "source_split": row["source_split"],
        "assembly_id": row["assembly_id"],
        "tier": row["tier"],
        "state_id": header["state_id"],
        "pose_id": header["pose_id"],
        "part_move": part_move,
        "features": features,
        "targets": {
            "T": _target(
                stability.get("T"),
                stability.get("queried") is True and type(stability.get("T")) is bool,
            )
        },
        "actions": action_records,
    }


def _find_assembly(prepared, source_split, assembly_id):
    rows = _read_jsonl(prepared["dataset_root"] / "assemblies.jsonl")
    matches = [
        row
        for row in rows
        if row["source_split"] == source_split and row["assembly_id"] == assembly_id
    ]
    _require(len(matches) == 1, "assembly is absent or duplicated in frozen index")
    return matches[0]


def _shard_path(prepared, row):
    return (
        prepared["output_root"]
        / "shards"
        / row["source_split"]
        / (row["assembly_id"] + ".jsonl")
    )


def _write_atomic(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(production._canonical_json(record) + "\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def extract(manifest_path, source_root, data_root, source_split, assembly_id):
    prepared = _read_config(manifest_path, source_root, data_root)
    row = _find_assembly(prepared, source_split, assembly_id)
    if not row["included"]:
        return {"status": "excluded", "source_split": source_split, "assembly_id": assembly_id}
    assembly_dir = prepared["source_root"] / source_split / assembly_id
    _require(assembly_dir.is_dir(), "source assembly is missing")
    _require(
        production._hash_assembly(assembly_dir) == row["input_sha256"],
        "source assembly hash differs from frozen index",
    )
    output = _shard_path(prepared, row)
    if output.is_file():
        records = _read_jsonl(output)
        _require(records and records[0].get("record_type") == "header", "invalid shard")
        header = records[0]
        _require(
            header.get("feature_manifest_sha256") == prepared["manifest_sha256"]
            and header.get("extractor_sha256") == prepared["extractor_sha256"]
            and header.get("input_sha256") == row["input_sha256"],
            "existing feature shard belongs to another build",
        )
        return {"status": "already_complete", "path": str(output)}

    static_parts = _load_parts(assembly_dir)
    _require(len(static_parts) == row["part_count"], "source part count differs")
    graph = _load_contact_graph(assembly_dir / "contact_graph.json", sorted(static_parts))
    full_volume = sum(part["volume"] for part in static_parts.values())
    pose_cache = {}
    candidate_records = []
    valid_counts = {name: Counter() for name in ("R", "T", "Y")}
    for header, stability, actions in dataset_index.index_candidates(prepared["dataset_root"], row):
        posed = _posed_parts(static_parts, header["pose"], pose_cache)
        record = _candidate_record(
            row,
            header,
            stability,
            actions,
            static_parts,
            posed,
            graph,
            full_volume,
            prepared["config"]["support_tolerance_fraction"],
        )
        candidate_records.append(record)
        target = record["targets"]["T"]
        valid_counts["T"]["valid" if target["valid"] else "null"] += 1
        for action in record["actions"]:
            for name in ("R", "Y"):
                target = action["targets"][name]
                valid_counts[name]["valid" if target["valid"] else "null"] += 1

    header_record = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "record_type": "header",
        "phase": prepared["config"]["phase"],
        "feature_manifest_sha256": prepared["manifest_sha256"],
        "protocol_sha256": prepared["protocol_sha256"],
        "extractor_sha256": prepared["extractor_sha256"],
        "dataset_split": row["dataset_split"],
        "source_split": row["source_split"],
        "assembly_id": row["assembly_id"],
        "tier": row["tier"],
        "input_sha256": row["input_sha256"],
        "candidate_features": list(CANDIDATE_FEATURES),
        "action_features": list(ACTION_FEATURES),
    }
    summary_record = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "record_type": "summary",
        "complete": True,
        "candidate_count": len(candidate_records),
        "action_count": 6 * len(candidate_records),
        "target_counts": {
            name: {key: valid_counts[name][key] for key in ("valid", "null")}
            for name in ("R", "T", "Y")
        },
    }
    _write_atomic(output, [header_record] + candidate_records + [summary_record])
    return {
        "status": "complete",
        "path": str(output),
        "candidate_count": len(candidate_records),
        "action_count": 6 * len(candidate_records),
    }


def _validate_target(target, name):
    _require(isinstance(target, dict) and set(target) == {"value", "valid"}, "invalid target")
    _target(target["value"], target["valid"])
    if name == "T":
        return


def _validate_shard(path, row, prepared):
    records = _read_jsonl(path)
    _require(len(records) >= 2, "feature shard is empty")
    header, summary = records[0], records[-1]
    candidates = records[1:-1]
    _require(
        header.get("schema_version") == SHARD_SCHEMA_VERSION
        and header.get("record_type") == "header"
        and summary.get("schema_version") == SHARD_SCHEMA_VERSION
        and summary.get("record_type") == "summary"
        and summary.get("complete") is True,
        "feature shard envelope is invalid",
    )
    _require(
        header.get("feature_manifest_sha256") == prepared["manifest_sha256"]
        and header.get("protocol_sha256") == prepared["protocol_sha256"]
        and header.get("extractor_sha256") == prepared["extractor_sha256"],
        "feature shard provenance differs",
    )
    _require(
        header.get("source_split") == row["source_split"]
        and header.get("assembly_id") == row["assembly_id"]
        and header.get("dataset_split") == row["dataset_split"]
        and header.get("tier") == row["tier"]
        and header.get("input_sha256") == row["input_sha256"],
        "feature shard assembly identity differs",
    )
    _require(header.get("candidate_features") == list(CANDIDATE_FEATURES), "candidate schema differs")
    _require(header.get("action_features") == list(ACTION_FEATURES), "action schema differs")
    _require(len(candidates) == row["candidate_count"], "feature candidate count differs")
    target_counts = {name: Counter() for name in ("R", "T", "Y")}
    seen_candidates = set()
    seen_actions = set()
    for candidate in candidates:
        _require(
            candidate.get("schema_version") == RECORD_SCHEMA_VERSION
            and candidate.get("record_type") == "candidate",
            "candidate feature record is invalid",
        )
        _require(candidate["candidate_id"] not in seen_candidates, "duplicate candidate")
        seen_candidates.add(candidate["candidate_id"])
        _validate_features(candidate["features"], CANDIDATE_FEATURES)
        _validate_target(candidate["targets"]["T"], "T")
        target_counts["T"]["valid" if candidate["targets"]["T"]["valid"] else "null"] += 1
        _require(len(candidate["actions"]) == 6, "candidate does not have six actions")
        _require(
            [action["direction_id"] for action in candidate["actions"]]
            == [direction_id for direction_id, _ in dataset_index.DIRECTIONS],
            "action direction order differs",
        )
        for action in candidate["actions"]:
            _require(action["action_id"] not in seen_actions, "duplicate action")
            seen_actions.add(action["action_id"])
            _validate_features(action["features"], ACTION_FEATURES)
            for name in ("R", "Y"):
                _validate_target(action["targets"][name], name)
                target_counts[name][
                    "valid" if action["targets"][name]["valid"] else "null"
                ] += 1
            r_target = action["targets"]["R"]
            y_target = action["targets"]["Y"]
            t_target = candidate["targets"]["T"]
            _require(
                y_target == _joint_target(r_target, t_target),
                "R/T/Y conjunction differs",
            )
    expected_counts = {
        name: {key: target_counts[name][key] for key in ("valid", "null")}
        for name in ("R", "T", "Y")
    }
    _require(summary.get("candidate_count") == len(candidates), "summary candidate count differs")
    _require(summary.get("action_count") == len(seen_actions), "summary action count differs")
    _require(summary.get("target_counts") == expected_counts, "summary target counts differ")
    return {
        "path": path.relative_to(prepared["output_root"]).as_posix(),
        "sha256": production._sha256_file(path),
        "dataset_split": row["dataset_split"],
        "tier": row["tier"],
        "candidate_count": len(candidates),
        "action_count": len(seen_actions),
        "target_counts": expected_counts,
    }


def audit(manifest_path, source_root, data_root):
    prepared = _read_config(manifest_path, source_root, data_root)
    rows = [
        row
        for row in _read_jsonl(prepared["dataset_root"] / "assemblies.jsonl")
        if row["included"]
    ]
    shards = []
    totals = {}
    for row in rows:
        path = _shard_path(prepared, row)
        _require(path.is_file(), "feature shard is missing: {}".format(path))
        shard = _validate_shard(path, row, prepared)
        shards.append(shard)
        key = "{}/{}".format(row["dataset_split"], row["tier"])
        scope = totals.setdefault(
            key,
            {
                "assembly_count": 0,
                "candidate_count": 0,
                "action_count": 0,
                "target_counts": {
                    name: {"valid": 0, "null": 0} for name in ("R", "T", "Y")
                },
            },
        )
        scope["assembly_count"] += 1
        scope["candidate_count"] += shard["candidate_count"]
        scope["action_count"] += shard["action_count"]
        for name in ("R", "T", "Y"):
            for status in ("valid", "null"):
                scope["target_counts"][name][status] += shard["target_counts"][name][status]
    report = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "phase": prepared["config"]["phase"],
        "status": "passed",
        "feature_manifest_sha256": prepared["manifest_sha256"],
        "protocol_sha256": prepared["protocol_sha256"],
        "extractor_sha256": prepared["extractor_sha256"],
        "candidate_features": list(CANDIDATE_FEATURES),
        "action_features": list(ACTION_FEATURES),
        "assembly_count": len(rows),
        "candidate_count": sum(shard["candidate_count"] for shard in shards),
        "action_count": sum(shard["action_count"] for shard in shards),
        "per_split_tier": dict(sorted(totals.items())),
        "shards": shards,
    }
    expected = prepared["protocol"]["dataset"]["observed_profile"]
    report["target_counts"] = {
        name: {
            status: sum(
                scope["target_counts"][name][status] for scope in totals.values()
            )
            for status in ("valid", "null")
        }
        for name in ("R", "T", "Y")
    }
    _require(report["assembly_count"] == prepared["protocol"]["dataset"]["eligible_assembly_count"], "assembly total differs")
    _require(report["candidate_count"] == expected["candidate_count"], "candidate total differs")
    _require(report["action_count"] == expected["action_count"], "action total differs")
    for name in ("R", "T", "Y"):
        expected_labels = expected["labels"][name]
        _require(
            report["target_counts"][name]
            == {
                "valid": expected_labels["true"] + expected_labels["false"],
                "null": expected_labels["null"],
            },
            "{} target counts differ".format(name),
        )
    _write_atomic(prepared["output_root"] / "manifest.json", [report])
    return report


def _parser():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(root / "configs" / "features.json")
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", "--output-root", dest="data_root", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--source-split", choices=("training_assembly", "test_assembly"))
    parser.add_argument("--assembly-id")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.validate_only:
        prepared = _read_config(args.manifest, args.source_root, args.data_root)
        result = {"status": "valid", "phase": prepared["config"]["phase"]}
    elif args.audit:
        result = audit(args.manifest, args.source_root, args.data_root)
    else:
        _require(args.source_split is not None, "--source-split is required")
        _require(
            isinstance(args.assembly_id, str) and args.assembly_id.isdigit(),
            "--assembly-id must be numeric",
        )
        result = extract(
            args.manifest,
            args.source_root,
            args.data_root,
            args.source_split,
            args.assembly_id,
        )
    print(production._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
