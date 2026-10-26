"""Planner rankers for FAST and the baselines, and planning-study run preparation."""

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from fast import cad_features, graph_contract, ranker_models, ranker_train, pointnet_models, action_planner
from plan_sequence import production_sim_labels as production


METHODS = ("asap_random", "asap_heur_out", "asap_released_gatv2", "cebulla_gcn", "heur_out_direction_prior", "fast")
LEGACY_METHODS = METHODS[:3]
MLP_CHECK_METHOD = "mlp_check"  # post-hoc MLP planning check; deliberately not in METHODS
DIRECTION_PRIOR_SCHEMA = "fast.direction-prior.v1"
SMOKE_AUDIT_SCHEMA = "fast.planner-smoke-audit.v1"
RELEASED_CHECKPOINT_SHA256 = "430a51253c25170e8f845ee21834a1e33714225fdf5f87b7cb687e4fe260d99a"
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_REPLICATES = 10000


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    from fast.graph_materialize import _write_json as write
    write(path, value)


class OnlineFeatureBuilder:
    """Recompute the frozen 41+16 CAD-only inputs for one live search context."""

    def __init__(self, assembly_dir, feature_config, assembly_id=None):
        self.assembly_dir = Path(assembly_dir)
        self.assembly_id = assembly_id or self.assembly_dir.name
        self.static_parts = cad_features._load_parts(self.assembly_dir)
        self.all_parts = sorted(self.static_parts)
        self.graph = cad_features._load_contact_graph(self.assembly_dir / "contact_graph.json", self.all_parts)
        self.full_volume = sum(part["volume"] for part in self.static_parts.values())
        self.tolerance = float(feature_config["support_tolerance_fraction"])
        self.input_sha256 = production._hash_assembly(self.assembly_dir)
        self.pose_cache = {}

    def record(self, parts, pose_id, pose):
        part_ids = sorted(str(part) for part in parts)
        graph_contract._require(len(part_ids) >= 2 and set(part_ids).issubset(self.static_parts), "online parts differ")
        posed = cad_features._posed_parts(self.static_parts, pose, self.pose_cache)
        node_x, action_x = [], []
        for part in part_ids:
            header = {"parts": part_ids, "parts_after": [other for other in part_ids if other != part], "part_move": part}
            candidate, parent = cad_features._candidate_features(
                header, self.static_parts, posed, self.graph, self.full_volume, self.tolerance
            )
            node_x.append([candidate[name] for name in cad_features.CANDIDATE_FEATURES])
            action_x.append([
                [cad_features._action_features(
                    {"direction": vector}, part, part_ids, posed, self.static_parts, self.graph, parent
                )[name] for name in cad_features.ACTION_FEATURES]
                for vector in graph_contract.DIRECTION_VECTORS
            ])
        index = {part: offset for offset, part in enumerate(part_ids)}
        edges = sorted(
            (index[str(source)], index[str(target)])
            for source, target in self.graph.subgraph(part_ids).edges
            for source, target in ((source, target), (target, source))
        )
        count = len(part_ids)
        return {
            "part_ids": part_ids, "node_x": np.asarray(node_x, dtype=np.float32),
            "edge_index": np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64),
            "action_x": np.asarray(action_x, dtype=np.float32),
        }


def _torch_online_batch(record, normalization):
    count = len(record["part_ids"])
    node_x = (record["node_x"] - np.asarray(normalization["node"]["mean"], dtype=np.float32)) / np.asarray(
        normalization["node"]["std"], dtype=np.float32
    )
    action_x = (record["action_x"] - np.asarray(normalization["action"]["mean"], dtype=np.float32)) / np.asarray(
        normalization["action"]["std"], dtype=np.float32
    )
    return {
        "node_x": torch.as_tensor(node_x), "edge_index": torch.as_tensor(record["edge_index"]),
        "node_ptr": torch.as_tensor([0, count]), "candidate_index": torch.arange(count),
        "action_x": torch.as_tensor(action_x),
    }


def _load_checkpoint(path, expected_sha256, model, architecture, seed):
    path = Path(path)
    if expected_sha256 is None:  # released weights-only state dict
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model.eval()
    graph_contract._require(production._sha256_file(path) == expected_sha256, "planner checkpoint hash differs")
    sidecar = _read_json(str(path) + ".json")
    graph_contract._require(sidecar.get("checkpoint_sha256") == expected_sha256, "planner checkpoint sidecar differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    provenance = payload.get("provenance", {})
    graph_contract._require(
        payload.get("schema_version") == ranker_train.CHECKPOINT_SCHEMA_VERSION
        and provenance.get("architecture") == architecture
        and provenance.get("seed") == seed
        and provenance.get("test_rows_loaded") == 0,
        "planner checkpoint provenance differs",
    )
    model.load_state_dict(payload["model_state"])
    return model.eval()


class FactorizedRanker:
    def __init__(self, feature_builder, normalization, checkpoint, checkpoint_sha256, seed, architecture="graph_transformer"):
        self.features = feature_builder
        self.normalization = normalization
        self.model = _load_checkpoint(
            checkpoint, checkpoint_sha256, ranker_models.build_model(architecture), architecture, seed
        )

    def __call__(self, parts, pose_id, pose, context_id):
        del context_id
        record = self.features.record(parts, pose_id, pose)
        batch = _torch_online_batch(record, self.normalization)
        with torch.no_grad():
            probabilities = ranker_models.probabilities(self.model(batch))["Y"].numpy()
        scores = {
            (part, direction): probabilities[index, direction_index]
            for index, part in enumerate(record["part_ids"])
            for direction_index, direction in enumerate(action_planner.DIRECTIONS)
        }
        return action_planner.scored_actions(parts, scores)


class CebullaRanker:
    def __init__(self, feature_builder, normalization, checkpoint, checkpoint_sha256, seed):
        self.features = feature_builder
        self.normalization = normalization
        self.model = _load_checkpoint(
            checkpoint, checkpoint_sha256, pointnet_models.CebullaGCN(), "cebulla_gcn", seed
        )

    def __call__(self, parts, pose_id, pose, context_id):
        del context_id
        record = self.features.record(parts, pose_id, pose)
        batch = _torch_online_batch(record, self.normalization)
        with torch.no_grad():
            logits = self.model(batch).numpy()
        scores = {part: float(logits[index]) for index, part in enumerate(record["part_ids"])}
        order = sorted(parts, key=lambda part: (-scores[part], str(part)))
        action_scores = {(part, direction): scores[part] for part in parts for direction in action_planner.DIRECTIONS}
        return action_planner.ordered_actions(parts, order, scores=action_scores)


class ReleasedASAPRanker:
    def __init__(self, asset_folder, assembly_dir, checkpoint, seed):
        import trimesh
        from assets.load import load_assembly
        from plan_sequence.physics_planner import get_contact_graph

        graph_contract._require(production._sha256_file(checkpoint) == RELEASED_CHECKPOINT_SHA256, "released ASAP checkpoint hash differs")
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            assembly = load_assembly(assembly_dir)
            self.graph = get_contact_graph(str(asset_folder), str(assembly_dir), save_sdf=True)
            point_clouds = []
            for node in self.graph.nodes:
                mesh = assembly[str(node)]["mesh"]
                graph_contract._require(mesh.is_watertight, "released ASAP baseline requires watertight parts")
                points = torch.from_numpy(trimesh.sample.sample_surface(mesh, 1000)[0]).float()
                point_clouds.append(points)
                self.graph.nodes[node].update(x=points, volume=torch.tensor(mesh.volume).float(),
                                              center_mass=torch.tensor(mesh.center_mass).float())
            points = torch.vstack(point_clouds)
            minimum, maximum = points.min(dim=0).values, points.max(dim=0).values
            center, scale = 0.5 * (minimum + maximum), 2.0 / (maximum - minimum).max()
            for node in self.graph.nodes:
                self.graph.nodes[node]["x"] = (self.graph.nodes[node]["x"] - center) * scale
                self.graph.nodes[node]["volume"] *= scale
                self.graph.nodes[node]["center_mass"] = (self.graph.nodes[node]["center_mass"] - center) * scale
                self.graph.nodes[node]["distance"] = torch.linalg.norm(self.graph.nodes[node]["center_mass"])
        finally:
            np.random.set_state(state)
        model_path = Path(__file__).resolve().parents[1] / "plan_sequence" / "generator" / "network" / "model.py"
        model_spec = importlib.util.spec_from_file_location("_fast_released_asap_model", model_path)
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        args = argparse.Namespace(max_pc_size=1000, feat_len=512, dropout=0.0, batch_norm=1)
        self.model = model_module.ModelGraph(args)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
        self.model.eval()

    def __call__(self, parts, pose_id, pose, context_id):
        del pose_id, pose, context_id
        from torch_geometric.utils import from_networkx
        graph = self.graph.subgraph(list(parts))
        for node in graph:
            graph.nodes[node]["edge_count"] = torch.tensor(len(graph.edges(node))).float()
            graph.nodes[node]["file"] = node
        graph = from_networkx(graph)
        with torch.no_grad():
            logits_tensor = self.model.forward_logits(graph).reshape(-1)
            logits = logits_tensor.numpy()
            probabilities = torch.sigmoid(logits_tensor).numpy()
        scores = {str(part): float(logit) for part, logit in zip(graph.file, logits)}
        order = np.asarray(graph.file, dtype=str)[np.argsort(probabilities)[::-1]].tolist()
        action_scores = {(part, direction): scores[part] for part in parts for direction in action_planner.DIRECTIONS}
        return action_planner.ordered_actions(parts, order, scores=action_scores)


class LegacyPartGenerator:
    """Part-order adapter for the separately reported ASAP 50/400 counter."""

    def __init__(self, method, seed, asset_folder, assembly_dir, released_checkpoint):
        graph_contract._require(method in LEGACY_METHODS, "legacy ASAP method differs")
        self.method, self.seed = method, seed
        self.asset_folder, self.assembly_dir = asset_folder, assembly_dir
        self.base_part, self.save_sdf = None, True
        if method == "asap_heur_out":
            from assets.load import load_assembly
            self.meshes = {part: value["mesh"] for part, value in load_assembly(assembly_dir).items()}
        else:
            self.meshes = None
        self.released = ReleasedASAPRanker(asset_folder, assembly_dir, released_checkpoint, seed) \
            if method == "asap_released_gatv2" else None

    def generate_candidate_part(self, assembled_parts):
        parts = list(assembled_parts)
        if self.method == "asap_random":
            return np.random.permutation(np.asarray(parts, dtype=object)).tolist()
        if self.method == "asap_heur_out":
            import trimesh
            center = trimesh.Scene([self.meshes[part] for part in parts]).centroid
            shuffled = np.random.permutation(np.asarray(parts, dtype=object)).tolist()
            return sorted(shuffled, key=lambda part: -float(np.linalg.norm(self.meshes[part].center_mass - center)))
        actions = self.released(parts, "legacy", None, "legacy:{}".format(action_planner._state_id(parts)))
        return [actions[index]["part"] for index in range(0, len(actions), len(action_planner.DIRECTIONS))]


def make_ranker(method, seed, assembly_dir, features, normalization, *, fast_checkpoint=None,
                fast_sha256=None, cebulla_checkpoint=None, cebulla_sha256=None,
                released_checkpoint=None, direction_rates=None, asset_folder=None, mlp_checkpoint=None, mlp_sha256=None):
    graph_contract._require(method in METHODS + (MLP_CHECK_METHOD,) and seed in (0, 1, 2), "planner method or seed differs")
    if method == "asap_random":
        return lambda parts, pose_id, pose, context_id: action_planner.random_actions(parts, seed, context_id)
    from plan_sequence.generate_budgeted_sim_labels import _heur_out_ranker
    heur_out = _heur_out_ranker(Path(assembly_dir), seed)
    if method == "asap_heur_out":
        return lambda parts, pose_id, pose, context_id: action_planner.ordered_actions(parts, heur_out(parts))
    if method == "heur_out_direction_prior":
        graph_contract._require(set(direction_rates or {}) == set(action_planner.DIRECTIONS), "direction prior differs")
        return lambda parts, pose_id, pose, context_id: action_planner.direction_prior_actions(parts, heur_out(parts), direction_rates)
    if method == "asap_released_gatv2":
        return ReleasedASAPRanker(asset_folder, assembly_dir, released_checkpoint, seed)
    if method == "cebulla_gcn":
        return CebullaRanker(features, normalization, cebulla_checkpoint, cebulla_sha256, seed)
    if method == MLP_CHECK_METHOD:
        return FactorizedRanker(features, normalization, mlp_checkpoint, mlp_sha256, seed, architecture="mlp")
    return FactorizedRanker(features, normalization, fast_checkpoint, fast_sha256, seed)


def build_direction_prior(graph_root, output_path):
    """Freeze unsmoothed direction rates from ranker train shards only."""
    graph_root, output_path = Path(graph_root), Path(output_path)
    graph_contract._require(not output_path.exists(), "direction-prior artifact cannot be overwritten")
    audit_path = graph_root / "audit.json"
    audit = _read_json(audit_path)
    rows = [row for row in audit["inventory"] if row.get("dataset_split") == "train"]
    graph_contract._require(audit.get("status") == "passed" and audit.get("test_rows_loaded") == 0 and len(rows) == 1703,
                     "graph audit cannot authorize direction prior")
    positive = np.zeros(6, dtype=np.int64)
    valid = np.zeros(6, dtype=np.int64)
    for inventory in rows:
        graph_contract._require(inventory.get("source_split") == "training_assembly", "direction prior references non-training source")
        path = (graph_root / inventory["path"]).resolve()
        graph_contract._require(production._is_within(path, graph_root) and production._sha256_file(path) == inventory["sha256"],
                         "direction-prior shard differs")
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        graph_contract._require(records[0].get("dataset_split") == "train" and records[-1].get("complete") is True,
                         "direction-prior shard envelope differs")
        for raw in records[1:-1]:
            record = graph_contract.validate_development_record(raw)
            record_positive, record_valid = _direction_counts(record)
            positive += record_positive
            valid += record_valid
    result = {
        "schema_version": DIRECTION_PRIOR_SCHEMA, "status": "locked",
        "estimator": "unsmoothed_train_only_R_positive_rate", "directions": list(action_planner.DIRECTIONS),
        "positive": {direction: int(positive[index]) for index, direction in enumerate(action_planner.DIRECTIONS)},
        "valid": {direction: int(valid[index]) for index, direction in enumerate(action_planner.DIRECTIONS)},
        "rates": {direction: float(positive[index] / valid[index]) for index, direction in enumerate(action_planner.DIRECTIONS)},
        "training_assembly_count": len(rows), "graph_audit_sha256": production._sha256_file(audit_path),
        "builder_sha256": production._sha256_file(Path(__file__)), "test_rows_loaded": 0,
    }
    _write_json(output_path, result)
    return result


def _direction_counts(record):
    mask = record["targets"]["R_valid"]
    return (mask & (record["targets"]["R"] == 1)).sum(axis=0), mask.sum(axis=0)


def stage_released_checkpoint(source, target):
    source, target = Path(source), Path(target)
    graph_contract._require(production._sha256_file(source) == RELEASED_CHECKPOINT_SHA256, "released checkpoint source differs")
    if target.exists():
        graph_contract._require(production._sha256_file(target) == RELEASED_CHECKPOINT_SHA256, "staged released checkpoint differs")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    graph_contract._require(production._sha256_file(target) == RELEASED_CHECKPOINT_SHA256, "staged released checkpoint differs")
    return target


def _checkpoint_rows(audit, architecture):
    rows = sorted((row for row in audit["inventory"] if row["architecture"] == architecture), key=lambda row: row["seed"])
    graph_contract._require([row["seed"] for row in rows] == [0, 1, 2], "planner checkpoint seed matrix differs")
    return rows


def frozen_artifacts(spec_path, data_root):
    spec_path, data_root = Path(spec_path).resolve(), Path(data_root).resolve()
    spec = _read_json(spec_path)
    graph_contract._require(spec.get("schema_version") == "fast.pointnet-spec.v1" and spec.get("status") == "frozen",
                     "PointNet planner spec differs")
    repository_root = spec_path.parent.parent
    graph_root = data_root / spec["artifacts"]["ranker_graphs"]
    ranker_root = data_root / spec["artifacts"]["ranker_training"]
    pointnet_root = data_root / spec["artifacts"]["pointnet_runs"]
    graph_audit_path = graph_root / "audit.json"
    graph_audit = _read_json(graph_audit_path)
    ranker_audit_path = ranker_root / "training-audit.json"
    ranker_audit = _read_json(ranker_audit_path)
    pointnet_audit_path = pointnet_root / "training-audit.json"
    pointnet_audit = _read_json(pointnet_audit_path)
    decision_path = pointnet_root / "representation-decision.json"
    decision = _read_json(decision_path)
    validation_audit_path = pointnet_root / "validation-audit.json"
    validation_audit = _read_json(validation_audit_path)
    graph_contract._require(
        graph_audit.get("status") == ranker_audit.get("status") == pointnet_audit.get("status") == validation_audit.get("status") == "passed"
        and graph_audit.get("test_rows_loaded") == ranker_audit.get("test_rows_loaded") == pointnet_audit.get("test_rows_loaded") == validation_audit.get("test_rows_loaded") == 0
        and decision.get("selected_representation") == "graph_transformer_control"
        and decision.get("pointnet_advanced") is False and decision.get("test_rows_loaded") == 0,
        "audited planner inputs differ",
    )
    control = _checkpoint_rows(ranker_audit, "graph_transformer")
    cebulla = _checkpoint_rows(pointnet_audit, "cebulla_gcn")
    expected = spec["control"]["checkpoints"]
    graph_contract._require(all(row["best_checkpoint_sha256"] == expected[str(row["seed"])] for row in control),
                     "selected control checkpoint differs")
    feature_config_path = repository_root / "configs" / "features.json"
    return {
        "spec": spec, "spec_path": spec_path, "spec_sha256": production._sha256_file(spec_path),
        "repository_root": repository_root, "data_root": data_root, "graph_root": graph_root,
        "ranker_root": ranker_root, "pointnet_root": pointnet_root,
        "graph_audit": graph_audit, "graph_audit_path": graph_audit_path,
        "ranker_audit": ranker_audit, "ranker_audit_path": ranker_audit_path,
        "pointnet_audit": pointnet_audit, "pointnet_audit_path": pointnet_audit_path,
        "decision": decision, "decision_path": decision_path,
        "validation_audit": validation_audit, "validation_audit_path": validation_audit_path,
        "control": control, "cebulla": cebulla, "normalization": graph_audit["normalization"],
        "feature_config": _read_json(feature_config_path), "feature_config_path": feature_config_path,
    }


def _method_inputs(artifacts, method, seed, released_checkpoint, direction_prior):
    result = {"released_checkpoint": released_checkpoint}
    if method == "fast":
        row = artifacts["control"][seed]
        result.update(fast_checkpoint=artifacts["ranker_root"] / row["best_checkpoint"], fast_sha256=row["best_checkpoint_sha256"])
    elif method == "cebulla_gcn":
        row = artifacts["cebulla"][seed]
        result.update(cebulla_checkpoint=artifacts["pointnet_root"] / row["best_checkpoint"], cebulla_sha256=row["best_checkpoint_sha256"])
    elif method == MLP_CHECK_METHOD:
        row = _checkpoint_rows(artifacts["ranker_audit"], "mlp")[seed]
        result.update(mlp_checkpoint=artifacts["ranker_root"] / row["best_checkpoint"], mlp_sha256=row["best_checkpoint_sha256"])
    if method == "heur_out_direction_prior":
        result["direction_rates"] = direction_prior["rates"]
    return result


def run_planner(spec_path, data_root, asset_folder, assembly_dir, output_path, method, seed, *,
                released_checkpoint, direction_prior_path, registry=None, registry_sha256=None,
                development=False, trace_context=None):
    if development:
        graph_contract._require("training_assembly" in Path(assembly_dir).parts, "development smoke requires a training assembly")
        test_rows_loaded = 0
        resolved_scope = "development"
    else:
        graph_contract._require(registry is not None and registry_sha256 is not None, "locked registry is required before test access")
        locked = action_planner.validate_registry(registry, registry_sha256)
        simulator = importlib.util.find_spec("redmax_py")
        graph_contract._require(simulator is not None and simulator.origin and Path(simulator.origin).is_file(),
                         "RedMax runtime is unavailable")
        graph_contract._require(
            locked.get("direction_prior", {}).get("sha256") == production._sha256_file(direction_prior_path)
            and locked.get("released_baseline", {}).get("sha256") == production._sha256_file(released_checkpoint)
            and locked.get("redmax", {}).get("binary_sha256") == production._sha256_file(simulator.origin)
            and locked.get("trace_schema", {}).get("implementation_sha256") == production._sha256_file(Path(action_planner.__file__)),
            "runtime differs from locked registry",
        )
        graph_contract._require(
            locked.get("implementation") == ranker_train._repository_identity(require_clean=True),
            "planning-study implementation differs from locked registry",
        )
        test_rows_loaded, resolved_scope = 1, "locked_test"
    artifacts = frozen_artifacts(spec_path, data_root)
    prior = _read_json(direction_prior_path)
    graph_contract._require(prior.get("schema_version") == DIRECTION_PRIOR_SCHEMA and prior.get("status") == "locked"
                     and prior.get("test_rows_loaded") == 0, "direction prior differs")
    graph_contract._require(production._sha256_file(released_checkpoint) == RELEASED_CHECKPOINT_SHA256,
                     "released ASAP checkpoint differs")
    features = OnlineFeatureBuilder(assembly_dir, artifacts["feature_config"])
    ranker = make_ranker(
        method, seed, assembly_dir, features, artifacts["normalization"],
        asset_folder=asset_folder, **_method_inputs(artifacts, method, seed, released_checkpoint, prior)
    )
    from assets.load import load_part_ids
    parts = tuple(load_part_ids(assembly_dir))
    oracle = action_planner.ASAPDirectionOracle(asset_folder, assembly_dir, removal_timeout=60.0,
                                                 stability_timeout=900.0, save_sdf=True)
    actions, summary = action_planner.run_action_dfs(
        parts, action_planner.ASAPStablePoseProvider(assembly_dir), ranker, oracle,
        max_action_validations=8 * (len(parts) - 1), method=method, seed=seed,
        assembly_id=Path(assembly_dir).name, wall_timeout_seconds=7200.0, test_rows_loaded=test_rows_loaded,
    )
    simulator = importlib.util.find_spec("redmax_py")
    graph_contract._require(simulator is not None and simulator.origin and Path(simulator.origin).is_file(), "RedMax runtime is unavailable")
    header = {
        "record_type": "header", "schema_version": action_planner.TRACE_SCHEMA_VERSION,
        "complete": False, "assembly_id": Path(assembly_dir).name, "input_sha256": features.input_sha256,
        "method": method, "seed": seed, "dataset_scope": resolved_scope,
        "spec_sha256": artifacts["spec_sha256"], "representation_decision_sha256": production._sha256_file(artifacts["decision_path"]),
        "registry_sha256": registry_sha256, "mandatory_redmax_validation": True,
        "ranker_inputs": {name: str(value) for name, value in _method_inputs(
            artifacts, method, seed, released_checkpoint, prior
        ).items() if name.endswith("sha256")},
        "direction_prior_sha256": production._sha256_file(direction_prior_path),
        "released_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
        "redmax_binary_sha256": production._sha256_file(simulator.origin),
        "test_rows_loaded": test_rows_loaded,
    }
    graph_contract._require(
        trace_context is None or not set(header).intersection(trace_context),
        "trace context overwrites frozen header fields",
    )
    header.update(trace_context or {})
    digest = action_planner.write_trace(output_path, header, actions, summary)
    return {"trace": str(output_path), "trace_sha256": digest, "summary": summary}


def audit_development_smoke(trace_path, output_path):
    trace_path, output_path = Path(trace_path), Path(output_path)
    graph_contract._require(not output_path.exists(), "smoke audit cannot be overwritten")
    with trace_path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    header, actions, summary = rows[0], rows[1:-1], rows[-1]
    sidecar = Path(str(trace_path) + ".sha256").read_text(encoding="ascii").split()[0]
    graph_contract._require(
        sidecar == production._sha256_file(trace_path)
        and header.get("dataset_scope") == "development" and header.get("test_rows_loaded") == 0
        and summary.get("complete") is True and summary.get("test_rows_loaded") == 0
        and summary.get("mandatory_redmax_validation") is True and summary.get("unvalidated_leaf_shortcut") is False
        and summary.get("action_validation_count") == len(actions) and len(actions) > 0
        and all(not row["accepted_transition"] or (row["R"]["label"] is True and row["T"]["label"] is True) for row in actions)
        and not any(row["R"]["status"] == "error" or (row["T"] and row["T"]["status"] == "error") for row in actions),
        "development RedMax smoke did not pass",
    )
    result = {
        "schema_version": SMOKE_AUDIT_SCHEMA, "status": "passed", "trace": trace_path.name,
        "trace_sha256": sidecar, "method": header["method"], "seed": header["seed"],
        "action_validation_count": len(actions), "accepted_transition_count": summary["accepted_transition_count"],
        "redmax_rollout_count": summary["redmax_rollout_count"], "stability_restart_count": summary["stability_restart_count"],
        "redmax_binary_sha256": header["redmax_binary_sha256"],
        "mandatory_redmax_validation": True, "test_rows_loaded": 0,
    }
    _write_json(output_path, result)
    return result


def write_readiness(output_path):
    """Record the built RedMax binary and runtime label that a locked registry binds."""
    simulator = importlib.util.find_spec("redmax_py")
    graph_contract._require(simulator is not None and simulator.origin, "RedMax runtime is unavailable")
    readiness = {"status": "ready", "redmax_sha256": production._sha256_file(simulator.origin),
                 "git_commit": ranker_train._repository_identity(require_clean=True)["commit"],
                 "image": os.environ.get("FAST_IMAGE", "local")}
    graph_contract._require(not Path(output_path).exists(), "readiness record cannot be overwritten")
    _write_json(output_path, readiness)
    return readiness


def _registry_model_entries(root, rows, evidence_root, evidence_name):
    entries = []
    for row in rows:
        checkpoint = root / row["best_checkpoint"]
        evidence = evidence_root / "seed-{}.json".format(row["seed"])
        graph_contract._require(production._sha256_file(checkpoint) == row["best_checkpoint_sha256"], "registry checkpoint differs")
        report = _read_json(evidence)
        graph_contract._require(report.get("test_rows_loaded") == 0, "registry model evidence used test rows")
        entry = {
            "seed": row["seed"], "checkpoint": checkpoint.relative_to(root.parent.parent).as_posix(),
            "checkpoint_sha256": row["best_checkpoint_sha256"], "checkpoint_sidecar_sha256": production._sha256_file(str(checkpoint) + ".json"),
            "parameter_count": row["parameter_count"],
        }
        entry[evidence_name] = evidence.relative_to(root.parent.parent).as_posix()
        entry[evidence_name + "_sha256"] = production._sha256_file(evidence)
        entries.append(entry)
    return entries


def build_registry(spec_path, data_root, direction_prior_path, released_checkpoint, smoke_audit_path,
                   disclosure_path, redmax_readiness_path, output_path):
    artifacts = frozen_artifacts(spec_path, data_root)
    identity = ranker_train._repository_identity(require_clean=True)
    prior = _read_json(direction_prior_path)
    smoke = _read_json(smoke_audit_path)
    disclosure = _read_json(disclosure_path)
    graph_contract._require(prior.get("status") == "locked" and smoke.get("status") == disclosure.get("status") == "passed"
                     and prior.get("test_rows_loaded") == smoke.get("test_rows_loaded") == 0,
                     "registry prerequisites did not pass")
    redmax_readiness = _read_json(redmax_readiness_path)
    graph_contract._require(redmax_readiness.get("status") == "ready" and len(redmax_readiness.get("redmax_sha256", "")) == 64,
                     "RedMax readiness differs")
    protocol_path = artifacts["repository_root"] / artifacts["spec"]["inputs"]["dataset_protocol"]["path"]
    protocol = _read_json(protocol_path)
    point_audit = artifacts["data_root"] / artifacts["spec"]["artifacts"]["pointnet_points"] / "audit.json"
    graph_contract._require(_read_json(point_audit).get("test_rows_loaded") == 0, "point cache audit used test rows")
    graph_contract._require(production._sha256_file(released_checkpoint) == RELEASED_CHECKPOINT_SHA256, "released baseline differs")
    released_relative = Path(released_checkpoint).resolve().relative_to(artifacts["data_root"])
    prior_relative = Path(direction_prior_path).resolve().relative_to(artifacts["data_root"])
    registry = {
        "schema_version": "fast.study-registry.v1", "status": "locked", "implementation": identity,
        "container": redmax_readiness["image"],
        "redmax": {"module": "redmax_py", "binary_sha256": redmax_readiness["redmax_sha256"],
                   "readiness_sha256": production._sha256_file(redmax_readiness_path),
                   "producer_commit": redmax_readiness["git_commit"], "producer_image": redmax_readiness["image"]},
        "dataset_protocol": {"path": protocol_path.relative_to(artifacts["repository_root"]).as_posix(),
                             "sha256": production._sha256_file(protocol_path),
                             "build_spec_sha256": protocol["dataset"]["build_spec_sha256"],
                             "artifact_hashes": protocol["dataset"]["artifacts"],
                             "source_hashes": protocol["dataset"]["source_hashes"]},
        "exclusions": protocol["dataset"]["exclusions"],
        "disclosure": {"record": disclosure, "sha256": production._sha256_file(disclosure_path)},
        "point_cache": {"path": point_audit.relative_to(artifacts["data_root"]).as_posix(), "sha256": production._sha256_file(point_audit)},
        "representation_decision": {"path": artifacts["decision_path"].relative_to(artifacts["data_root"]).as_posix(),
                                    "sha256": production._sha256_file(artifacts["decision_path"]),
                                    "selected": "graph_transformer_control"},
        "models": {"architecture": "graph_transformer", "factorization": "sigmoid(R)*sigmoid(T)",
                   "seeds": _registry_model_entries(artifacts["ranker_root"], artifacts["control"],
                                                     artifacts["ranker_root"] / "calibration" / "graph_transformer", "calibration")},
        "released_baseline": {"checkpoint": released_relative.as_posix(), "sha256": RELEASED_CHECKPOINT_SHA256},
        "direction_prior": {"path": prior_relative.as_posix(), "sha256": production._sha256_file(direction_prior_path),
                            "rates": prior["rates"]},
        "cebulla": {"adaptation_not_reproduction": True,
                    "seeds": _registry_model_entries(artifacts["pointnet_root"], artifacts["cebulla"],
                                                      artifacts["pointnet_root"] / "validation" / "cebulla_gcn", "validation")},
        "seeds": [0, 1, 2],
        "planner_contract": {"methods": list(METHODS), "directions": list(action_planner.DIRECTIONS),
                             "budgets": [1, 2, 4, 8], "max_action_validations": "8*(part_count-1)",
                             "redmax_validation_required": True, "unvalidated_two_part_leaf": False,
                             "legacy_methods": list(LEGACY_METHODS), "legacy_budgets": [50, 400],
                             "legacy_counter": "bundled ASAP part-pose evaluation",
                             "legacy_unvalidated_two_part_leaf": True,
                             "inference": {"experimental_unit": "assembly", "seed_aggregation": "mean_within_assembly",
                                           "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                                           "bootstrap": "paired_tier_stratified",
                                           "hypothesis_test": "two_sided_centered_paired_bootstrap",
                                           "correction": "Holm_across_four_budgets_per_baseline", "familywise_alpha": 0.05},
                             "max_stable_poses": 3, "stable_pose_probability": 0.9, "wall_timeout_seconds": 7200,
                             "removal_timeout_seconds": 60, "stability_timeout_seconds": 900},
        "hardware": {"cpus": 4, "memory_gib": 16, "disk_gib": 20, "gpus": 0},
        "trace_schema": {"version": action_planner.TRACE_SCHEMA_VERSION,
                         "implementation_sha256": production._sha256_file(Path(action_planner.__file__)),
                         "development_smoke_audit_sha256": production._sha256_file(smoke_audit_path)},
        "aggregate_outputs": {"root": "runs/planning-study-v4/aggregate", "per_assembly_test_payload_export": False},
    }
    return action_planner.freeze_registry(output_path, registry)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prior = subparsers.add_parser("direction-prior")
    prior.add_argument("--graph-root", required=True)
    prior.add_argument("--output", required=True)
    stage = subparsers.add_parser("stage-released-checkpoint")
    stage.add_argument("--source", required=True)
    stage.add_argument("--target", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--asset-folder", required=True)
    run.add_argument("--assembly-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--method", choices=METHODS + (MLP_CHECK_METHOD,), required=True)
    run.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    run.add_argument("--released-checkpoint", required=True)
    run.add_argument("--direction-prior", required=True)
    run.add_argument("--registry")
    run.add_argument("--registry-sha256")
    run.add_argument("--development", action="store_true")
    smoke_audit = subparsers.add_parser("audit-smoke")
    smoke_audit.add_argument("--trace", required=True)
    smoke_audit.add_argument("--output", required=True)
    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--output", required=True)
    registry = subparsers.add_parser("freeze-registry")
    registry.add_argument("--spec", required=True)
    registry.add_argument("--data-root", required=True)
    registry.add_argument("--direction-prior", required=True)
    registry.add_argument("--released-checkpoint", required=True)
    registry.add_argument("--smoke-audit", required=True)
    registry.add_argument("--disclosure", required=True)
    registry.add_argument("--redmax-readiness", required=True)
    registry.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "direction-prior":
        result = build_direction_prior(args.graph_root, args.output)
    elif args.command == "stage-released-checkpoint":
        result = {"path": str(stage_released_checkpoint(args.source, args.target)), "sha256": RELEASED_CHECKPOINT_SHA256}
    elif args.command == "run":
        result = run_planner(
            args.spec, args.data_root, args.asset_folder, args.assembly_dir, args.output, args.method, args.seed,
            released_checkpoint=args.released_checkpoint, direction_prior_path=args.direction_prior,
            registry=args.registry, registry_sha256=args.registry_sha256, development=args.development,
        )
    elif args.command == "readiness":
        result = write_readiness(args.output)
    elif args.command == "audit-smoke":
        result = audit_development_smoke(args.trace, args.output)
    else:
        result = {"registry_sha256": build_registry(
            args.spec, args.data_root, args.direction_prior, args.released_checkpoint,
            args.smoke_audit, args.disclosure, args.redmax_readiness, args.output,
        )}
    print(production._canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
