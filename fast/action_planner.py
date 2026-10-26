"""Mandatory-validation action-level DFS used by every planner arm."""

import hashlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from fast import graph_contract
from plan_sequence import production_sim_labels as production


DIRECTIONS = ("+Z", "-Z", "+X", "-X", "+Y", "-Y")
TRACE_SCHEMA_VERSION = "fast.planner-trace.v1"


def ordered_actions(parts, part_order, direction_order=None, scores=None):
    parts, part_order = tuple(parts), list(part_order)
    graph_contract._require(sorted(part_order) == sorted(parts) and len(set(part_order)) == len(parts), "part order differs from state")
    direction_order = direction_order or {part: DIRECTIONS for part in parts}
    actions = []
    for part in part_order:
        directions = tuple(direction_order[part])
        graph_contract._require(sorted(directions) == sorted(DIRECTIONS) and len(set(directions)) == 6, "direction order differs")
        for direction in directions:
            actions.append({"part": part, "direction": direction, "score": None if scores is None else float(scores[(part, direction)])})
    return actions


def direction_prior_actions(parts, part_order, positive_rates):
    direction_order = {
        part: sorted(DIRECTIONS, key=lambda direction: (-float(positive_rates[direction]), DIRECTIONS.index(direction)))
        for part in parts
    }
    return ordered_actions(parts, part_order, direction_order)


def scored_actions(parts, scores):
    graph_contract._require(set(scores) == {(part, direction) for part in parts for direction in DIRECTIONS}, "action scores are incomplete")
    ordered = sorted(scores, key=lambda action: (-float(scores[action]), str(action[0]), DIRECTIONS.index(action[1])))
    return [{"part": part, "direction": direction, "score": float(scores[(part, direction)])} for part, direction in ordered]


def random_actions(parts, seed, context_id):
    token = production._canonical_json([int(seed), context_id])
    context_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    order = np.random.RandomState(context_seed).permutation(np.asarray(parts, dtype=object)).tolist()
    return ordered_actions(parts, order)


class ASAPStablePoseProvider:
    """Memoize ASAP's frozen max-three, cumulative-0.9 stable-pose order."""

    def __init__(self, assembly_dir):
        self.assembly_dir = assembly_dir
        self.cache = {}

    def __call__(self, parts):
        parts = tuple(parts)
        if parts not in self.cache:
            from plan_sequence.generate_sim_labels import _compute_stable_poses
            poses = _compute_stable_poses(self.assembly_dir, list(parts), 3, 0.9)
            self.cache[parts] = poses or [(None, None)]
        return [
            ("pose-{:03d}".format(index), pose)
            for index, (pose, _probability) in enumerate(self.cache[parts])
        ]


def _normalise_result(result, name):
    graph_contract._require(isinstance(result, dict) and result.get("label") in (True, False, None), "{} result differs".format(name))
    status = result.get("status")
    graph_contract._require(isinstance(status, str) and status, "{} status is absent".format(name))
    normalised = {
        "label": result["label"], "status": status, "reason": result.get("reason"),
        "rollout_count": int(result.get("rollout_count", 1)), "runtime_s": float(result.get("runtime_s", 0.0)),
        "forward_steps": int(result.get("forward_steps", 0)),
    }
    if "error" in result:
        normalised["error"] = result["error"]
    return normalised


def _state_id(parts):
    return hashlib.sha256(production._canonical_json(list(parts)).encode("utf-8")).hexdigest()


def run_action_dfs(parts, pose_provider, ranker, oracle, *, max_action_validations, method, seed,
                   assembly_id="synthetic", wall_timeout_seconds=7200.0, test_rows_loaded=0):
    """Search with model-controlled order and simulator-controlled transitions."""
    root = tuple(parts)
    graph_contract._require(len(root) >= 2 and len(set(root)) == len(root), "root parts differ")
    graph_contract._require(max_action_validations == 8 * (len(root) - 1), "action-validation ceiling differs")
    graph_contract._require(wall_timeout_seconds > 0, "wall timeout must be positive")
    graph_contract._require(type(test_rows_loaded) is int and test_rows_loaded in (0, 1), "test-row count differs")
    stack, attempted, stability_cache, rank_cache = [root], set(), {}, {}
    actions_trace, transitions = [], []
    search_nodes, backtracks, poses_visited = {_state_id(root)}, 0, set()
    solution_at = None
    started = time.perf_counter()
    ranking_seconds = 0.0
    if hasattr(oracle, "set_deadline"):
        oracle.set_deadline(started + wall_timeout_seconds)
    stop_reason = "frontier_exhausted"
    while stack and len(actions_trace) < max_action_validations:
        if time.perf_counter() - started >= wall_timeout_seconds:
            stop_reason = "wall_timeout"
            break
        state = stack[-1]
        if len(state) == 1:
            solution_at, stop_reason = len(actions_trace), "solution_found"
            break
        selected = None
        for pose_id, pose in pose_provider(state):
            graph_contract._require(isinstance(pose_id, str) and pose_id, "pose ID differs")
            poses_visited.add((_state_id(state), pose_id))
            context_id = "{}:{}:{}".format(assembly_id, _state_id(state), pose_id)
            if context_id not in rank_cache:
                ranking_started = time.perf_counter()
                rank_cache[context_id] = list(ranker(state, pose_id, pose, context_id))
                ranking_seconds += time.perf_counter() - ranking_started
            ranked = rank_cache[context_id]
            expected = {(part, direction) for part in state for direction in DIRECTIONS}
            actual = [(row["part"], row["direction"]) for row in ranked]
            graph_contract._require(set(actual) == expected and len(actual) == len(expected), "ranker action list differs")
            order_hash = hashlib.sha256(production._canonical_json(ranked).encode("utf-8")).hexdigest()
            for rank, action in enumerate(ranked):
                key = (_state_id(state), pose_id, action["part"], action["direction"])
                if key in attempted:
                    continue
                cache_key = (_state_id(state), pose_id, action["part"])
                if stability_cache.get(cache_key, {}).get("label") is False:
                    attempted.add(key)
                    continue
                selected = (pose_id, pose, action, rank, order_hash, key, cache_key)
                break
            if selected is not None:
                break
        if selected is None:
            stack.pop()
            backtracks += int(bool(stack))
            continue
        pose_id, pose, action, rank, order_hash, key, cache_key = selected
        attempted.add(key)
        before = len(actions_trace)
        removal = _normalise_result(oracle.evaluate_direction(state, action["part"], action["direction"], pose), "R")
        stability_cache_hit = cache_key in stability_cache
        stability = stability_cache.get(cache_key)
        accepted = False
        child = tuple(part for part in state if part != action["part"])
        if removal["label"] is True:
            if not stability_cache_hit:
                stability = _normalise_result(oracle.evaluate_stability(child, pose), "T")
                stability_cache[cache_key] = stability
            accepted = stability["label"] is True
        actions_trace.append({
            "record_type": "action", "action_validation": before + 1,
            "state_id": _state_id(state), "parts": list(state), "pose_id": pose_id,
            "part": action["part"], "direction": action["direction"], "score": action.get("score"), "rank": rank,
            "ordered_action_list_sha256": order_hash, "R": removal, "T": stability,
            "T_cache_hit": stability_cache_hit, "accepted_transition": accepted,
            "child_state_id": _state_id(child) if accepted else None,
            "redmax_rollouts": removal["rollout_count"],
            "stability_restarts": 0 if stability is None or stability_cache_hit else stability["rollout_count"],
        })
        if accepted:
            transitions.append({"action_validation": len(actions_trace), "parent_state_id": _state_id(state),
                                "child_state_id": _state_id(child), "part": action["part"], "direction": action["direction"]})
            graph_contract._require(removal["label"] is True and stability["label"] is True, "unvalidated transition")
            stack.append(child)
            search_nodes.add(_state_id(child))
    if solution_at is None and stack and len(stack[-1]) == 1:
        solution_at, stop_reason = len(actions_trace), "solution_found"
    elif len(actions_trace) >= max_action_validations:
        stop_reason = "budget_reached"
    budgets = {
        str(multiplier): {
            "absolute_action_validations": multiplier * (len(root) - 1),
            "success": solution_at is not None and solution_at <= multiplier * (len(root) - 1),
        }
        for multiplier in (1, 2, 4, 8)
    }
    wall_seconds = time.perf_counter() - started
    removability_seconds = sum(row["R"]["runtime_s"] for row in actions_trace)
    stability_seconds = sum(
        row["T"]["runtime_s"] for row in actions_trace if row["T"] is not None and not row["T_cache_hit"]
    )
    summary = {
        "record_type": "summary", "schema_version": TRACE_SCHEMA_VERSION, "complete": True,
        "assembly_id": assembly_id, "method": method, "seed": int(seed), "root_parts": list(root),
        "max_action_validations": max_action_validations, "action_validation_count": len(actions_trace),
        "legacy_asap_part_evaluation_count": None, "legacy_counter_applicable": False,
        "solution_action_validation": solution_at, "budgets": budgets,
        "redmax_rollout_count": sum(row["redmax_rollouts"] for row in actions_trace),
        "stability_restart_count": sum(row["stability_restarts"] for row in actions_trace),
        "accepted_transition_count": len(transitions), "transitions": transitions,
        "T_cache_hit_count": sum(row["T_cache_hit"] for row in actions_trace),
        "censored_action_count": sum(
            row["R"]["label"] is None or (row["T"] is not None and row["T"]["label"] is None)
            for row in actions_trace
        ),
        "removability_seconds": removability_seconds, "stability_seconds": stability_seconds,
        "model_ranking_seconds": ranking_seconds,
        "overhead_seconds": max(0.0, wall_seconds - removability_seconds - stability_seconds - ranking_seconds),
        "search_node_count": len(search_nodes), "backtrack_count": backtracks, "pose_count": len(poses_visited),
        "stop_reason": stop_reason, "wall_seconds": wall_seconds,
        "wall_timeout_seconds": float(wall_timeout_seconds), "ranked_context_count": len(rank_cache),
        "mandatory_redmax_validation": True, "unvalidated_leaf_shortcut": False, "test_rows_loaded": test_rows_loaded,
    }
    return actions_trace, summary


def write_trace(path, header, actions, summary):
    path = Path(path)
    digest_path = Path(str(path) + ".sha256")
    graph_contract._require(not path.exists() and not digest_path.exists(), "planner trace cannot be overwritten")
    graph_contract._require(header.get("record_type") == "header" and header.get("schema_version") == TRACE_SCHEMA_VERSION, "trace header differs")
    graph_contract._require(summary.get("record_type") == "summary" and summary.get("complete") is True, "trace summary differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in [header] + list(actions) + [summary]:
                stream.write(production._canonical_json(row) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = production._sha256_file(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + digest_path.name, dir=str(digest_path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(digest + "  " + path.name + "\n")
        os.replace(temporary, digest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


class ASAPDirectionOracle:
    """Thin single-direction adapter over the frozen ASAP RedMax planners."""

    def __init__(self, asset_folder, assembly_dir, removal_timeout=60.0, stability_timeout=900.0, save_sdf=True):
        self.asset_folder, self.assembly_dir = asset_folder, assembly_dir
        self.removal_timeout, self.stability_timeout, self.save_sdf = removal_timeout, stability_timeout, save_sdf
        from assets.load import load_part_ids
        self.all_parts = load_part_ids(assembly_dir)
        self.deadline = None

    def set_deadline(self, deadline):
        self.deadline = float(deadline)

    def _timeout(self, configured):
        if self.deadline is None:
            return configured
        return max(0.0, min(configured, self.deadline - time.perf_counter()))

    def evaluate_direction(self, parts, part, direction, pose):
        from plan_sequence.feasibility_check import get_R3_actions
        from plan_sequence.physics_planner import MultiPartPathPlanner
        action = get_R3_actions()[DIRECTIONS.index(direction)]
        started = time.perf_counter()
        timeout = self._timeout(self.removal_timeout)
        if timeout <= 0:
            return {"label": None, "status": "timeout", "reason": "planning_wall_timeout", "rollout_count": 0,
                    "forward_steps": 0, "runtime_s": 0.0}
        try:
            planner = MultiPartPathPlanner(self.asset_folder, self.assembly_dir, [item for item in parts if item != part], part,
                                           parts_removed=[item for item in self.all_parts if item not in parts], pose=pose, save_sdf=self.save_sdf)
            detail = planner.check_success_detailed(action, timeout=timeout)
            return {"label": detail["success"] if detail["status"] != "timeout" else None, "status": detail["status"],
                    "reason": detail["reason"], "rollout_count": 1, "forward_steps": detail["forward_steps"], "runtime_s": detail["runtime_s"]}
        except Exception as error:
            return {"label": None, "status": "error", "reason": "simulator_error", "rollout_count": 1,
                    "forward_steps": 0, "runtime_s": time.perf_counter() - started,
                    "error": {"type": type(error).__name__, "message": str(error)}}

    def evaluate_stability(self, parts_after, pose):
        from plan_sequence.asap_label_oracle import ASAPLabelOracle
        timeout = self._timeout(self.stability_timeout)
        if timeout <= 0:
            return {"label": None, "status": "timeout", "reason": "planning_wall_timeout", "rollout_count": 0,
                    "forward_steps": 0, "runtime_s": 0.0}
        oracle = ASAPLabelOracle(self.asset_folder, self.assembly_dir, timeout=timeout, save_sdf=self.save_sdf)
        return oracle.evaluate_stability(parts_after, pose)


def freeze_registry(path, registry):
    path = Path(path)
    required = {
        "schema_version", "status", "implementation", "container", "redmax", "dataset_protocol",
        "exclusions", "disclosure", "point_cache", "representation_decision", "models", "released_baseline",
        "direction_prior", "cebulla", "seeds", "planner_contract", "hardware", "trace_schema", "aggregate_outputs",
    }
    graph_contract._require(set(registry) == required and registry["schema_version"] == "fast.study-registry.v1", "registry fields differ")
    graph_contract._require(registry["status"] == "locked" and registry["planner_contract"].get("redmax_validation_required") is True, "registry is not locked")
    graph_contract._require(not path.exists(), "locked registry cannot be overwritten")
    from fast.graph_materialize import _write_json
    _write_json(path, registry)
    return production._sha256_file(path)


def validate_registry(path, expected_sha256):
    graph_contract._require(production._sha256_file(path) == expected_sha256, "locked registry hash differs")
    import json
    with Path(path).open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    graph_contract._require(registry.get("status") == "locked" and registry.get("planner_contract", {}).get("redmax_validation_required") is True, "registry cannot authorize test access")
    return registry
