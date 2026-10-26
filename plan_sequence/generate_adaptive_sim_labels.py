"""Generate runtime-adaptive FAST labels with an ASAP-style primary spine."""

import argparse
import json
import sys
from pathlib import Path

from plan_sequence import generate_budgeted_sim_labels as worker
from plan_sequence import generate_reachable_sim_labels as reachable
from plan_sequence import generate_sim_labels as labels
from plan_sequence import mixed_search
from plan_sequence import run_sim_label_manifest as manifest_tools


MANIFEST_SCHEMA_VERSION = "fast.adaptive-label-manifest.v1"
_ROOT_FIELDS = {
    "schema_version",
    "label_schema_version",
    "phase",
    "protocol",
    "parameters",
    "assembly",
    "cache",
    "output_root",
    "audit_output",
    "expected",
}
_ASSEMBLY_FIELDS = {"path", "part_count", "part_ids", "input_sha256"}
_PARAMETER_FIELDS = {
    "max_poses",
    "pose_probability",
    "removal_timeout_s",
    "stability_timeout_s",
    "candidate_time_budget_s",
    "save_sdf",
    "seed",
}
_EXPECTED_FIELDS = {"state_part_attempt_budget", "candidate_evaluation_budget"}
_PROTOCOL = {
    "state_scope": "runtime_adaptive_oracle_reachable_remaining_part_states",
    "traversal": "heur_out_spine_then_alternating_breadth_depth",
    "candidate_ranking": "heur-out",
    "candidate_tie_break": "seeded_sha256",
    "breadth_state_priority": "shallowest_then_least_expanded",
    "depth_state_priority": "deepest_then_least_expanded",
    "breadth_candidate_order": "seeded_uniform",
    "depth_candidate_order": "heur-out",
    "pose_policy": "seeded_order_stop_on_first_feasible_pose",
    "action_space": "A6_world",
    "direction_frame": "world",
    "gravity": [0, 0, -1],
    "max_grippers": 2,
    "max_holding_parts": 1,
    "base_part": None,
    "support": "ground_plane",
    "support_normal": [0, 0, 1],
    "stability_oracle": "ASAP-greedy",
    "removability_coverage": "all_six_directions",
    "stability_policy": "query_once_if_any_direction_is_true",
    "state_expansion": "first_pose_with_any_direction_Y_true",
    "terminal_state": "one_part_remaining",
    "candidate_isolation": "one_fresh_process_per_candidate_pose",
    "stable_pose_policy": "compute_once_per_state_and_checkpoint",
    "budget_atomicity": "candidate_pose",
    "time_budget_scope": "cumulative_recorded_candidate_wall_time",
}


def _mesh_complexity(assembly_dir, part_count):
    mesh_bytes = 0
    vertex_count = 0
    face_count = 0
    for path in sorted(assembly_dir.glob("*.obj")):
        mesh_bytes += path.stat().st_size
        with path.open("rb") as stream:
            for line in stream:
                vertex_count += line.startswith(b"v ")
                face_count += line.startswith(b"f ")
    with (assembly_dir / "contact_graph.json").open("r", encoding="utf-8") as stream:
        contact_graph = json.load(stream)
    links = contact_graph.get("links")
    if not isinstance(links, list):
        raise ValueError("contact graph links must be a list")
    return {
        "part_count": part_count,
        "mesh_bytes": mesh_bytes,
        "mesh_vertex_count": vertex_count,
        "mesh_face_count": face_count,
        "contact_graph_edge_count": len(links),
    }


def _scheduler_sha256():
    root = Path(__file__).resolve().parents[1]
    return labels._hash_files(
        root, [Path(__file__), Path(worker.__file__), Path(mixed_search.__file__)]
    )[0]


def _validate_manifest(manifest_path, source_root, output_root):
    manifest_path = Path(manifest_path).resolve()
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("manifest not found: {}".format(manifest_path))
    if not source_root.is_dir():
        raise FileNotFoundError("source root not found: {}".format(source_root))
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("output root is not a directory: {}".format(output_root))
    if manifest_tools._is_within(source_root, output_root) or manifest_tools._is_within(
        output_root, source_root
    ):
        raise ValueError("source and output roots must not contain one another")

    manifest = manifest_tools._read_manifest(manifest_path)
    manifest_tools._require_fields(manifest, _ROOT_FIELDS, "manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive-label manifest schema")
    if manifest.get("label_schema_version") != reachable.SCHEMA_VERSION:
        raise ValueError("manifest label schema does not match candidate worker")
    manifest_tools._component(manifest.get("phase"), "phase")
    if manifest.get("protocol") != _PROTOCOL:
        raise ValueError("manifest protocol does not match adaptive worker")

    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    manifest_tools._require_fields(parameters, _PARAMETER_FIELDS, "parameters")
    if parameters.get("max_poses") != 3:
        raise ValueError("parameters.max_poses must be 3")
    pose_probability = manifest_tools._positive_number(
        parameters.get("pose_probability"), "parameters.pose_probability"
    )
    if pose_probability > 1:
        raise ValueError("parameters.pose_probability must be at most 1")
    removal_timeout = manifest_tools._positive_number(
        parameters.get("removal_timeout_s"), "parameters.removal_timeout_s"
    )
    stability_timeout = manifest_tools._positive_number(
        parameters.get("stability_timeout_s"), "parameters.stability_timeout_s"
    )
    candidate_time_budget = manifest_tools._positive_number(
        parameters.get("candidate_time_budget_s"),
        "parameters.candidate_time_budget_s",
    )
    if parameters.get("save_sdf") is not True:
        raise ValueError("parameters.save_sdf must be true")
    seed = parameters.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("parameters.seed must be a non-negative integer")

    assembly = manifest.get("assembly")
    if not isinstance(assembly, dict):
        raise ValueError("assembly must be an object")
    manifest_tools._require_fields(assembly, _ASSEMBLY_FIELDS, "assembly")
    assembly_relative = manifest_tools._relative_path(
        assembly.get("path"), "assembly.path"
    )
    assembly_dir = manifest_tools._resolve_relative(
        source_root, assembly_relative, "assembly.path"
    )
    part_ids = assembly.get("part_ids")
    part_count = assembly.get("part_count")
    if (
        type(part_count) is not int
        or not 3 <= part_count <= 70
        or not isinstance(part_ids, list)
        or len(part_ids) != part_count
        or part_ids != sorted(part_ids)
        or len(set(part_ids)) != len(part_ids)
        or any(not isinstance(part, str) or not part for part in part_ids)
    ):
        raise ValueError("assembly part IDs must define a 3-to-70-part assembly")
    input_sha256 = reachable._sha256(
        assembly.get("input_sha256"), "assembly.input_sha256"
    )
    if not assembly_dir.is_dir():
        raise FileNotFoundError("assembly not found: {}".format(assembly_dir))
    if not (assembly_dir / "contact_graph.json").is_file():
        raise FileNotFoundError("contact graph not found: {}".format(assembly_dir))
    actual_parts = sorted(path.stem for path in assembly_dir.glob("*.obj") if path.is_file())
    if actual_parts != part_ids:
        raise ValueError("assembly part IDs do not match its OBJ files")
    if labels._hash_assembly(assembly_dir)[0] != input_sha256:
        raise ValueError("assembly input SHA-256 does not match")

    state_part_budget = 2 * (part_count - 1)
    candidate_budget = 3 * state_part_budget
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("expected must be an object")
    manifest_tools._require_fields(expected, _EXPECTED_FIELDS, "expected")
    if expected.get("state_part_attempt_budget") != state_part_budget:
        raise ValueError("expected.state_part_attempt_budget must equal 2*(part_count-1)")
    if expected.get("candidate_evaluation_budget") != candidate_budget:
        raise ValueError("expected.candidate_evaluation_budget must equal 6*(part_count-1)")

    cache_relative = manifest_tools._relative_path(manifest.get("cache"), "cache")
    labels_relative = manifest_tools._relative_path(
        manifest.get("output_root"), "output_root"
    )
    audit_relative = manifest_tools._relative_path(
        manifest.get("audit_output"), "audit_output", ".json"
    )
    cache_dir = manifest_tools._resolve_relative(output_root, cache_relative, "cache")
    labels_root = manifest_tools._resolve_relative(
        output_root, labels_relative, "output_root"
    )
    audit_output = manifest_tools._resolve_relative(
        output_root, audit_relative, "audit_output"
    )
    if manifest_tools._is_within(cache_dir, labels_root) or manifest_tools._is_within(
        labels_root, cache_dir
    ):
        raise ValueError("output_root must not overlap cache")
    if manifest_tools._is_within(labels_root, audit_output):
        raise ValueError("audit_output must not be inside output_root")

    return {
        "manifest": manifest,
        "manifest_sha256": labels._sha256_bytes(
            labels._canonical_json(manifest).encode("utf-8")
        ),
        "assembly_dir": assembly_dir,
        "part_ids": part_ids,
        "input_sha256": input_sha256,
        "parameters": {
            "max_poses": 3,
            "pose_probability": pose_probability,
            "removal_timeout": removal_timeout,
            "stability_timeout": stability_timeout,
            "candidate_time_budget_s": candidate_time_budget,
            "save_sdf": True,
            "seed": seed,
            "state_part_budget": state_part_budget,
            "candidate_budget": candidate_budget,
        },
        "search_mode": "adaptive",
        "stop_on_first_feasible_pose": True,
        "seed_pose_order": True,
        "candidate_isolation": "one_process_per_candidate",
        "audit_policy": "adaptive",
        "scheduler_sha256": _scheduler_sha256(),
        "complexity": _mesh_complexity(assembly_dir, part_count),
        "cache_dir": cache_dir,
        "labels_root": labels_root,
        "summary_path": labels_root / "summary.json",
        "audit_output": audit_output,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--asset-folder")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    prepared = _validate_manifest(args.manifest, args.source_root, args.output_root)
    if args.validate_only:
        print(
            "validated {} for {}".format(
                prepared["manifest"]["phase"], prepared["assembly_dir"].name
            ),
            flush=True,
        )
        return 0
    result = worker.run(prepared, asset_folder=args.asset_folder)
    print(labels._canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
