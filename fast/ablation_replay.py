"""Component-ablation Experiment C: complete-label replay on the 115 small validation assemblies.

The frozen PointNet action-level DFS (`action_planner.run_action_dfs`) is replayed
unchanged. Only its two simulator-facing dependencies are replaced: stable poses come
from the stored `poses.json` order, and every R/T validation returns the stored label.
This is complete-label replay, not live planning or simulator evidence.
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
import torch

from fast import graph_contract, ranker_models, ranker_train, action_planner, planner_integration
from fast import ablation_closure as closure
from fast import ablation_common as common
from fast import ablation_offline as offline


SCHEMA_VERSION = "fast.ablation-replay.v1"
METHODS = ("fast_A0", "fast_A1", "fast_A2", "fast_B1", "fast_B2", "asap_random", "asap_heur_out",
           "heur_out_direction_prior", "cebulla_gcn")
BUDGETS = ("1", "2", "4", "8")
TRACE_METHOD = {"fast": "fast_A0", "asap_random": "asap_random", "asap_heur_out": "asap_heur_out",
                "cebulla_gcn": "cebulla_gcn", "heur_out_direction_prior": "heur_out_direction_prior"}
WALL_TIMEOUT_SECONDS = 7200.0


class LabelStore:
    """Stored exhaustive labels for one assembly, indexed by state, pose, part, and direction."""

    def __init__(self, assembly_id, states, poses, candidates):
        self.assembly_id = assembly_id
        self.state_parts = OrderedDict((state_id, tuple(parts)) for state_id, parts in states.items())
        self.state_by_parts = {frozenset(parts): state_id for state_id, parts in self.state_parts.items()}
        common.require(len(self.state_by_parts) == len(self.state_parts), "duplicate stored state")
        self.poses = poses
        self.candidates = candidates

    @classmethod
    def from_directory(cls, directory, assembly_id):
        directory = Path(directory)
        common.require_development_path(directory.resolve())
        summary = common.read_json(directory / "summary.json")
        common.require(summary.get("complete") is True and summary.get("assembly_id") == assembly_id, "label summary incomplete: " + assembly_id)
        states = OrderedDict((row["state_id"], tuple(row["parts"])) for row in summary["states"])
        poses = {}
        for row in summary["state_pose_files"]:
            payload = common.read_json(directory / row["path"])
            common.require(payload.get("complete") is True and payload["state_id"] == row["state_id"], "stored poses differ")
            pose_ids = [pose["pose_id"] for pose in payload["poses"]]
            common.require(pose_ids == ["pose-{:03d}".format(index) for index in range(len(pose_ids))] and pose_ids,
                           "stored pose order differs: " + row["state_id"])
            poses[row["state_id"]] = pose_ids
        candidates = {}
        for row in summary["candidate_files"]:
            records = common.read_jsonl(directory / row["path"])
            common.require(records[0].get("record_type") == "header" and records[-1].get("record_type") == "summary"
                           and records[-1].get("complete") is True, "candidate file envelope differs: " + row["path"])
            stability = [record for record in records if record.get("record_type") == "stability"]
            actions = [record for record in records if record.get("record_type") == "action"]
            common.require(len(stability) == 1, "candidate stability record differs: " + row["path"])
            directions = {record["direction_id"]: record for record in actions}
            common.require(len(actions) == 6 and set(directions) == set(common.DIRECTIONS), "candidate directions differ: " + row["path"])
            for record in [stability[0]] + actions:
                common.require(record["state_id"] == row["state_id"] and record["pose_id"] == row["pose_id"]
                               and record["part_move"] == row["part_move"], "candidate record identity differs: " + row["path"])
            key = (row["state_id"], row["pose_id"], row["part_move"])
            common.require(key not in candidates, "duplicate candidate file: " + row["path"])
            candidates[key] = {
                "T": {"label": stability[0]["T"], "status": stability[0]["T_status"], "reason": stability[0].get("reason"),
                      "rollout_count": int(stability[0]["oracle_rollout_count"]), "forward_steps": int(stability[0].get("forward_steps") or 0)},
                "R": {direction: {"label": record["R"], "status": record["R_status"], "reason": record.get("R_reason"),
                                  "rollout_count": int(record["oracle_rollout_count"]), "forward_steps": int(record.get("forward_steps") or 0)}
                      for direction, record in directions.items()},
            }
        return cls(assembly_id, states, poses, candidates)

    def state_id(self, parts):
        state_id = self.state_by_parts.get(frozenset(parts))
        common.require(state_id is not None, "replay reached an unrecorded state in " + self.assembly_id)
        return state_id

    def coverage(self):
        """Require every nonterminal state/pose to hold every part with all six directions and T."""
        nonterminal = [state_id for state_id, parts in self.state_parts.items() if len(parts) >= 2]
        contexts = 0
        for state_id in nonterminal:
            common.require(state_id in self.poses, "stored state has no poses: " + state_id)
            for pose_id in self.poses[state_id]:
                contexts += 1
                for part in self.state_parts[state_id]:
                    common.require((state_id, pose_id, part) in self.candidates, "missing candidate {}:{}:{}".format(state_id, pose_id, part))
        return {"nonterminal_states": len(nonterminal), "state_pose_contexts": contexts, "candidates": len(self.candidates)}


class StoredPoseProvider:
    def __init__(self, store):
        self.store = store

    def __call__(self, parts):
        state_id = self.store.state_id(parts)
        return [(pose_id, (state_id, pose_id)) for pose_id in self.store.poses[state_id]]


class LabelReplayOracle:
    """Return stored labels in the live ASAPDirectionOracle result shape; no simulator is used."""

    def __init__(self, store):
        self.store = store

    def _candidate(self, parts, part, pose):
        state_id, pose_id = pose
        common.require(state_id == self.store.state_id(parts), "replay pose belongs to another state")
        return self.store.candidates[(state_id, pose_id, part)]

    def evaluate_direction(self, parts, part, direction, pose):
        return dict(self._candidate(parts, part, pose)["R"][direction], runtime_s=0.0)

    def evaluate_stability(self, parts_after, pose):
        parent = self.store.state_parts[pose[0]]
        removed = [part for part in parent if part not in set(parts_after)]
        common.require(len(removed) == 1 and len(parts_after) == len(parent) - 1, "stability child differs from parent state")
        return dict(self._candidate(parent, removed[0], pose)["T"], runtime_s=0.0)


class ShardFeatureBuilder:
    """Serve the materialized graph record of a stored context to the frozen planner rankers."""

    def __init__(self, records):
        self.records = {}
        for record in records:
            common.require(record["dataset_split"] in graph_contract.DEVELOPMENT_SPLITS, "test row loaded")
            key = (record["state_id"], record["pose_id"])
            common.require(key not in self.records, "duplicate graph context")
            self.records[key] = record

    def record(self, parts, pose_id, pose):
        record = self.records.get((pose[0], pose_id))
        common.require(record is not None, "graph context is absent for {}:{}".format(pose[0], pose_id))
        common.require(record["part_ids"] == sorted(str(part) for part in parts)
                       and np.array_equal(record["candidate_index"], np.arange(len(parts))), "graph context parts differ")
        return {"part_ids": record["part_ids"], "node_x": record["node_x"], "edge_index": record["edge_index"], "action_x": record["action_x"]}


class ComponentRanker(planner_integration.FactorizedRanker):
    """Frozen FAST checkpoint with the A1/A2/B1/B2 score or grouping rule."""

    def __init__(self, variant, direction_rates, *args):
        super().__init__(*args)
        common.require(variant in ("A1", "A2", "B1", "B2"), "unknown component variant")
        self.variant, self.direction_rates = variant, direction_rates

    def __call__(self, parts, pose_id, pose, context_id):
        del context_id
        record = self.features.record(parts, pose_id, pose)
        batch = planner_integration._torch_online_batch(record, self.normalization)
        with torch.no_grad():
            probabilities = {name: value.numpy() for name, value in ranker_models.probabilities(self.model(batch)).items()}
        index = {part: offset for offset, part in enumerate(record["part_ids"])}
        if self.variant in ("A1", "A2"):
            matrix = probabilities["R"] if self.variant == "A1" else np.repeat(probabilities["T"][:, None], 6, axis=1)
            scores = {(part, direction): matrix[index[part], position]
                      for part in parts for position, direction in enumerate(action_planner.DIRECTIONS)}
            return action_planner.scored_actions(parts, scores)
        part_score = {part: float(probabilities["Y"][index[part]].max()) for part in parts}
        order = sorted(parts, key=lambda part: (-part_score[part], str(part)))
        if self.variant == "B1":
            return action_planner.ordered_actions(parts, order)
        return action_planner.direction_prior_actions(parts, order, self.direction_rates)


class ReplayInputs:
    def __init__(self, context):
        self.context = context
        data = context["data"]
        self.normalization = context["audit"]["normalization"]
        prior_path = data / closure.POINTNET_RUN / "planner" / "direction-prior.json"
        common.require(common.sha256_file(prior_path) == context["spec"]["experiments"]["B"]["direction_prior_sha256"], "direction prior differs")
        self.direction_rates = common.read_json(prior_path)["rates"]
        self.checkpoints = {}
        for architecture, run, names in (("graph_transformer", closure.RANKER_RUN, closure.FAST_CHECKPOINTS),
                                         ("cebulla_gcn", closure.POINTNET_RUN, closure.CEBULLA_CHECKPOINTS)):
            for seed, name in names.items():
                path = offline._checkpoint_path(context, run, architecture, seed, name)
                offline._load_frozen(context, architecture, seed)
                self.checkpoints[(architecture, seed)] = (path, closure.FROZEN_SHA256[path.relative_to(data).as_posix()])
        self.inventory = {row["assembly_id"]: row for row in context["audit"]["inventory"]}

    def assembly(self, assembly_id):
        data = self.context["data"]
        row = self.inventory[assembly_id]
        records = ranker_train._load_assembly(self.context["prepared"], row)
        store = LabelStore.from_directory(data / closure.LABEL_ROOT / assembly_id, assembly_id)
        geometry = common.closure_root(self.context["data_root"]) / "asap-data" / closure.GEOMETRY_ROOT / assembly_id
        from assets.load import load_part_ids
        root = tuple(load_part_ids(str(geometry)))
        common.require(frozenset(root) in store.state_by_parts and store.state_parts[store.state_id(root)] is not None, "root state differs")
        return {"row": row, "store": store, "features": ShardFeatureBuilder(records), "geometry": geometry, "root": root}

    def ranker(self, method, seed, item):
        fast_path, fast_sha = self.checkpoints[("graph_transformer", seed)]
        if method.startswith("fast_") and method != "fast_A0":
            return ComponentRanker(method.split("_", 1)[1], self.direction_rates, item["features"], self.normalization, fast_path, fast_sha, seed)
        cebulla_path, cebulla_sha = self.checkpoints[("cebulla_gcn", seed)]
        frozen_method = "fast" if method == "fast_A0" else method
        return planner_integration.make_ranker(
            frozen_method, seed, str(item["geometry"]), item["features"], self.normalization,
            fast_checkpoint=fast_path, fast_sha256=fast_sha, cebulla_checkpoint=cebulla_path, cebulla_sha256=cebulla_sha,
            direction_rates=self.direction_rates,
        )


def replay(item, ranker, method, seed):
    root = item["root"]
    return action_planner.run_action_dfs(
        root, StoredPoseProvider(item["store"]), ranker, LabelReplayOracle(item["store"]),
        max_action_validations=8 * (len(root) - 1), method=method, seed=seed,
        assembly_id=item["row"]["assembly_id"], wall_timeout_seconds=WALL_TIMEOUT_SECONDS, test_rows_loaded=0,
    )


# ---------------------------------------------------------------- live-trace equivalence


GATED_ACTION_FIELDS = ("action_validation", "state_id", "parts", "pose_id", "part", "direction", "rank",
                       "accepted_transition", "child_state_id", "T_cache_hit", "redmax_rollouts", "stability_restarts")
GATED_SUMMARY_FIELDS = ("assembly_id", "root_parts", "max_action_validations", "action_validation_count",
                        "solution_action_validation", "budgets", "redmax_rollout_count", "stability_restart_count",
                        "accepted_transition_count", "transitions", "T_cache_hit_count", "censored_action_count",
                        "search_node_count", "backtrack_count", "pose_count", "stop_reason", "ranked_context_count", "complete")


def _label(result):
    return None if result is None else result["label"]


def compare_trace(live_actions, live_summary, replay_actions, replay_summary):
    mismatches = []
    if len(live_actions) != len(replay_actions):
        mismatches.append({"field": "action_count", "live": len(live_actions), "replay": len(replay_actions)})
    for index, (live, replayed) in enumerate(zip(live_actions, replay_actions)):
        for field in GATED_ACTION_FIELDS:
            if live[field] != replayed[field]:
                mismatches.append({"action": index + 1, "field": field, "live": live[field], "replay": replayed[field]})
        for target in ("R", "T"):
            if _label(live[target]) != _label(replayed[target]):
                mismatches.append({"action": index + 1, "field": target + ".label", "live": _label(live[target]), "replay": _label(replayed[target])})
    for field in GATED_SUMMARY_FIELDS:
        if live_summary[field] != replay_summary[field]:
            mismatches.append({"field": "summary." + field, "live": live_summary[field], "replay": replay_summary[field]})
    reported = OrderedDict([
        ("ordered_action_list_sha256_equal", all(a["ordered_action_list_sha256"] == b["ordered_action_list_sha256"] for a, b in zip(live_actions, replay_actions))),
        ("score_floats_equal", all(a["score"] == b["score"] for a, b in zip(live_actions, replay_actions))),
        ("R_status_equal", all(a["R"]["status"] == b["R"]["status"] for a, b in zip(live_actions, replay_actions))),
        ("T_status_equal", all((a["T"] or {}).get("status") == (b["T"] or {}).get("status") for a, b in zip(live_actions, replay_actions))),
    ])
    return mismatches, reported


def equivalence(inputs):
    item = inputs.assembly(closure.REPLAY_FIXTURE)
    common.require(item["row"]["dataset_split"] == "train", "replay fixture is not training-only")
    rows = []
    for trace in closure.REPLAY_TRACES:
        path = inputs.context["data"] / trace
        common.require(common.sha256_file(path) == closure.FROZEN_SHA256[trace], "live trace hash differs: " + trace)
        records = common.read_jsonl(path)
        header, actions, summary = records[0], records[1:-1], records[-1]
        common.require(header["assembly_id"] == closure.REPLAY_FIXTURE and summary["test_rows_loaded"] == 0, "live trace differs")
        method, seed = TRACE_METHOD[header["method"]], int(header["seed"])
        replay_actions, replay_summary = replay(item, inputs.ranker(method, seed, item), header["method"], seed)
        replay_actions, replay_summary = _roundtrip(replay_actions), _roundtrip(replay_summary)
        mismatches, reported = compare_trace(actions, summary, replay_actions, replay_summary)
        rows.append(OrderedDict([
            ("trace", trace), ("sha256", closure.FROZEN_SHA256[trace]), ("method", header["method"]), ("seed", seed),
            ("action_validations", summary["action_validation_count"]), ("rejected_validations", sum(not row["accepted_transition"] for row in actions)),
            ("T_cache_hits", summary["T_cache_hit_count"]), ("backtracks", summary["backtrack_count"]),
            ("passed", not mismatches), ("mismatches", mismatches), ("reported_not_gated", reported),
        ]))
    return OrderedDict([
        ("fixture_assembly", closure.REPLAY_FIXTURE), ("fixture_split", "train"), ("status", "passed" if all(row["passed"] for row in rows) else "failed"),
        ("gated_action_fields", list(GATED_ACTION_FIELDS) + ["R.label", "T.label"]), ("gated_summary_fields", list(GATED_SUMMARY_FIELDS)),
        ("excluded_from_equality", inputs.context["spec"]["experiments"]["C"]["real_equivalence_gate"]["excluded_from_equality"]),
        ("traces", rows),
        ("disclosure", inputs.context["spec"]["experiments"]["C"]["real_equivalence_gate"]["disclosure"]),
    ])


def _roundtrip(value):
    import json
    return json.loads(common.canonical_json(value))


# ---------------------------------------------------------------- experiment


def coverage(inputs, assembly_ids):
    rows, totals = [], OrderedDict([("nonterminal_states", 0), ("state_pose_contexts", 0), ("candidates", 0)])
    for assembly_id in assembly_ids:
        row = inputs.inventory[assembly_id]
        common.require(row["dataset_split"] == "validation" and row["tier"] == "small", "replay assembly is not validation-small")
        store = LabelStore.from_directory(inputs.context["data"] / closure.LABEL_ROOT / assembly_id, assembly_id)
        counts = store.coverage()
        for name in totals:
            totals[name] += counts[name]
        rows.append(assembly_id)
    common.require(totals["candidates"] == closure.EXPECTED_LABEL_TOTALS["candidate_files"]
                   and totals["nonterminal_states"] == closure.EXPECTED_LABEL_TOTALS["nonterminal_states"], "replay coverage totals differ")
    return OrderedDict([("assemblies", len(rows)), ("graph_audit_status", inputs.context["audit"]["status"]),
                        ("all_rows_validation_small", True), ("label_summaries_complete_and_hash_verified", True),
                        ("every_state_pose_has_every_part_with_six_directions_and_T", True), ("test_paths", 0), ("totals", totals)])


def run(data_root):
    ranker_train.configure_determinism(0)
    context = offline.load_context(data_root)
    environment = offline.compatibility_environment()
    root = context["run_root"]
    for name in ("replay-equivalence.json", "small-label-replay.json"):
        common.require(not (root / name).exists(), "replay output already exists: " + name)
    manifest = common.read_json(root / "transfer-manifest.json")
    assembly_ids = manifest["small_validation_assembly_ids"]
    inputs = ReplayInputs(context)
    provenance = offline._provenance(context, environment)

    gate = equivalence(inputs)
    gate_report = OrderedDict([("schema_version", SCHEMA_VERSION), ("kind", "replay_equivalence")] + list(provenance.items()) + list(gate.items()))
    hashes = OrderedDict([("replay-equivalence.json", common.write_json_once(root / "replay-equivalence.json", gate_report))])
    common.require(gate["status"] == "passed", "real replay-equivalence gate failed; Experiment C is blocked")
    coverage_report = coverage(inputs, assembly_ids)

    per_assembly = []
    for assembly_id in assembly_ids:
        item = inputs.assembly(assembly_id)
        for method in METHODS:
            for seed in offline.SEEDS:
                actions, summary = replay(item, inputs.ranker(method, seed, item), method, seed)
                per_assembly.append(OrderedDict([
                    ("assembly_id", assembly_id), ("part_count", len(item["root"])), ("method", method), ("seed", seed),
                    ("solution_action_validation", summary["solution_action_validation"]),
                    ("success", OrderedDict((budget, summary["budgets"][budget]["success"]) for budget in BUDGETS)),
                    ("action_validation_count", summary["action_validation_count"]),
                    ("censored_action_count", summary["censored_action_count"]),
                    ("T_cache_hit_count", summary["T_cache_hit_count"]), ("backtrack_count", summary["backtrack_count"]),
                    ("stop_reason", summary["stop_reason"]),
                ]))
    per_assembly_path = root / "replay" / "per-assembly.json"
    hashes["replay/per-assembly.json"] = common.write_json_once(per_assembly_path, {"schema_version": SCHEMA_VERSION, "validation_only": True, "rows": per_assembly, "test_rows_loaded": 0})

    tables = {}
    for method in METHODS:
        tables[method] = {}
        for seed in offline.SEEDS:
            rows = [row for row in per_assembly if row["method"] == method and row["seed"] == seed]
            common.require(len(rows) == len(assembly_ids), "replay matrix is incomplete")
            tables[method][seed] = {row["assembly_id"]: {"success_k" + budget: [float(row["success"][budget]), 1] for budget in BUDGETS} for row in rows}
    order, samples, blocks, plan_verification = offline.load_plan(context)
    common.require(plan_verification["tier_draws_per_replicate"]["small"] == 115, "small-tier bootstrap block differs")
    summary_rows = OrderedDict()
    for method in METHODS:
        summary_rows[method] = OrderedDict()
        for budget in BUDGETS:
            seeds = OrderedDict((str(seed), float(np.mean([value["success_k" + budget][0] for value in tables[method][seed].values()]))) for seed in offline.SEEDS)
            summary_rows[method]["k" + budget] = OrderedDict([("success_rate_seed_mean", float(np.mean(list(seeds.values())))), ("seeds", seeds)])
    differences = OrderedDict()
    for method in METHODS[1:]:
        differences[method + "_minus_fast_A0"] = OrderedDict()
        for budget in BUDGETS:
            metric = "success_k" + budget
            point = summary_rows[method]["k" + budget]["success_rate_seed_mean"] - summary_rows["fast_A0"]["k" + budget]["success_rate_seed_mean"]
            differences[method + "_minus_fast_A0"]["k" + budget] = OrderedDict([
                ("point_difference", point),
                ("bootstrap", offline.paired_difference([tables[method][seed] for seed in offline.SEEDS],
                                                        [tables["fast_A0"][seed] for seed in offline.SEEDS], metric, order, samples, blocks["small"])),
            ])
    report = OrderedDict([("schema_version", SCHEMA_VERSION), ("kind", "small_label_replay"), ("experiment", context["spec"]["experiments"]["C"])]
                         + list(provenance.items()) + [
        ("replay_equivalence_sha256", hashes["replay-equivalence.json"]), ("coverage", coverage_report),
        ("bootstrap_plan", plan_verification), ("bootstrap_columns", "small-tier block only (115 draws per replicate)"),
        ("assembly_count", len(assembly_ids)), ("methods", list(METHODS)), ("seeds", list(offline.SEEDS)),
        ("success", summary_rows), ("paired_differences", differences),
        ("per_assembly_sha256", hashes["replay/per-assembly.json"]),
        ("wording", context["spec"]["experiments"]["C"]["wording"]),
    ])
    hashes["small-label-replay.json"] = common.write_json_once(root / "small-label-replay.json", report)
    return OrderedDict([("status", "passed"), ("outputs", hashes), ("test_rows_loaded", 0)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)
    print(common.canonical_json(run(args.data_root)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
