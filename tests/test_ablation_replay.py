"""Synthetic, simulator-free control-flow fixtures for the component-ablation label replay.

Each expected trace below was derived by hand from the frozen action_planner.run_action_dfs
rules: depth-first search, fixed part-ID/direction ranking, one action validation per attempted
action, stability evaluated only after R=True and cached per state/pose/part, the remaining
directions of a part skipped only after a cached T=False, and null labels never accepted.
"""

import unittest

from fast import action_planner
from fast import ablation_replay as replay


DIRS = action_planner.DIRECTIONS


def _candidate(r_true=(), r_null=(), t_label=False, t_rollouts=1):
    def r_label(direction):
        return None if direction in r_null else direction in r_true
    return {
        "T": {"label": t_label, "status": "stored", "reason": None, "rollout_count": t_rollouts, "forward_steps": 0},
        "R": {d: {"label": r_label(d), "status": "stored", "reason": None, "rollout_count": 1, "forward_steps": 0} for d in DIRS},
    }


def _item(states, overrides):
    """states: {state_id: parts}; every part of every nonterminal state gets a default all-false candidate."""
    candidates = {}
    for state_id, parts in states.items():
        if len(parts) >= 2:
            for part in parts:
                candidates[(state_id, "pose-000", part)] = overrides.get((state_id, part), _candidate())
    poses = {state_id: ["pose-000"] for state_id, parts in states.items() if len(parts) >= 2}
    store = replay.LabelStore("synthetic", states, poses, candidates)
    root = next(iter(states.values()))
    return {"root": tuple(root), "store": store, "row": {"assembly_id": "synthetic"}}


def _fixed_ranker(parts, pose_id, pose, context_id):
    return action_planner.ordered_actions(parts, sorted(parts))


def _run(item):
    return replay.replay(item, _fixed_ranker, "synthetic_fixed", 0)


def _label(result):
    return None if result is None else result["label"]


def _observed(actions):
    return [(row["action_validation"], tuple(row["parts"]), row["part"], row["direction"], row["rank"], row["R"]["label"],
             _label(row["T"]), row["T_cache_hit"], row["accepted_transition"], row["redmax_rollouts"], row["stability_restarts"])
            for row in actions]


SUMMARY_FIELDS = ("action_validation_count", "solution_action_validation", "accepted_transition_count", "T_cache_hit_count",
                  "censored_action_count", "redmax_rollout_count", "stability_restart_count", "backtrack_count",
                  "search_node_count", "pose_count", "ranked_context_count", "stop_reason", "max_action_validations")


def _summary(summary):
    values = {name: summary[name] for name in SUMMARY_FIELDS}
    values["budgets"] = {budget: summary["budgets"][budget]["success"] for budget in ("1", "2", "4", "8")}
    values["transitions"] = [(row["action_validation"], row["part"], row["direction"]) for row in summary["transitions"]]
    return values


ABC, BC, AC, AB, C = ("a", "b", "c"), ("b", "c"), ("a", "c"), ("a", "b"), ("c",)


class AblationReplayFixtureTest(unittest.TestCase):
    maxDiff = None

    def test_1_rejected_directions_before_an_accepted_action(self):
        item = _item({"S0": ABC, "S1": BC}, {
            ("S0", "a"): _candidate(r_true=("+X",), t_label=True),
            ("S1", "b"): _candidate(r_true=("+Z",), t_label=True),
        })
        actions, summary = _run(item)
        self.assertEqual(_observed(actions), [
            (1, ABC, "a", "+Z", 0, False, None, False, False, 1, 0),
            (2, ABC, "a", "-Z", 1, False, None, False, False, 1, 0),
            (3, ABC, "a", "+X", 2, True, True, False, True, 1, 1),
            (4, BC, "b", "+Z", 0, True, True, False, True, 1, 1),
        ])
        self.assertEqual(_summary(summary), {
            "action_validation_count": 4, "solution_action_validation": 4, "accepted_transition_count": 2,
            "T_cache_hit_count": 0, "censored_action_count": 0, "redmax_rollout_count": 4, "stability_restart_count": 2,
            "backtrack_count": 0, "search_node_count": 3, "pose_count": 2, "ranked_context_count": 2,
            "stop_reason": "solution_found", "max_action_validations": 16,
            "budgets": {"1": False, "2": True, "4": True, "8": True},
            "transitions": [(3, "a", "+X"), (4, "b", "+Z")],
        })

    def test_2_stability_cache_hit_across_directions_of_one_candidate(self):
        item = _item({"S0": ABC, "S1": AC}, {
            ("S0", "a"): _candidate(r_true=("+Z", "-Z"), t_label=None, t_rollouts=3),
            ("S0", "b"): _candidate(r_true=("+Z",), t_label=True),
            ("S1", "a"): _candidate(r_true=("+Z",), t_label=True),
        })
        actions, summary = _run(item)
        self.assertEqual(_observed(actions), [
            (1, ABC, "a", "+Z", 0, True, None, False, False, 1, 3),
            (2, ABC, "a", "-Z", 1, True, None, True, False, 1, 0),
            (3, ABC, "a", "+X", 2, False, None, True, False, 1, 0),
            (4, ABC, "a", "-X", 3, False, None, True, False, 1, 0),
            (5, ABC, "a", "+Y", 4, False, None, True, False, 1, 0),
            (6, ABC, "a", "-Y", 5, False, None, True, False, 1, 0),
            (7, ABC, "b", "+Z", 6, True, True, False, True, 1, 1),
            (8, AC, "a", "+Z", 0, True, True, False, True, 1, 1),
        ])
        self.assertEqual(_summary(summary), {
            "action_validation_count": 8, "solution_action_validation": 8, "accepted_transition_count": 2,
            "T_cache_hit_count": 5, "censored_action_count": 6, "redmax_rollout_count": 8, "stability_restart_count": 5,
            "backtrack_count": 0, "search_node_count": 3, "pose_count": 2, "ranked_context_count": 2,
            "stop_reason": "solution_found", "max_action_validations": 16,
            "budgets": {"1": False, "2": False, "4": True, "8": True},
            "transitions": [(7, "b", "+Z"), (8, "a", "+Z")],
        })

    def test_3_failed_continuation_backtracks_to_an_alternate_branch(self):
        item = _item({"S0": ABC, "S1": BC, "S2": AC}, {
            ("S0", "a"): _candidate(r_true=("+Z",), t_label=True),
            ("S0", "b"): _candidate(r_true=("+Z",), t_label=True),
            ("S1", "c"): _candidate(r_true=("+Z",), t_label=False),
            ("S2", "a"): _candidate(r_true=("+Z",), t_label=True),
        })
        actions, summary = _run(item)
        expected = [(1, ABC, "a", "+Z", 0, True, True, False, True, 1, 1)]
        expected += [(2 + index, BC, "b", direction, index, False, None, False, False, 1, 0) for index, direction in enumerate(DIRS)]
        expected += [(8, BC, "c", "+Z", 6, True, False, False, False, 1, 1)]
        # c's other directions are skipped after the cached T=False; S1 is exhausted and the search backtracks.
        expected += [(9 + index, ABC, "a", direction, 1 + index, False, True, True, False, 1, 0) for index, direction in enumerate(DIRS[1:])]
        expected += [(14, ABC, "b", "+Z", 6, True, True, False, True, 1, 1), (15, AC, "a", "+Z", 0, True, True, False, True, 1, 1)]
        self.assertEqual(_observed(actions), expected)
        self.assertEqual(_summary(summary), {
            "action_validation_count": 15, "solution_action_validation": 15, "accepted_transition_count": 3,
            "T_cache_hit_count": 5, "censored_action_count": 0, "redmax_rollout_count": 15, "stability_restart_count": 4,
            "backtrack_count": 1, "search_node_count": 4, "pose_count": 3, "ranked_context_count": 3,
            "stop_reason": "solution_found", "max_action_validations": 16,
            "budgets": {"1": False, "2": False, "4": False, "8": True},
            "transitions": [(1, "a", "+Z"), (14, "b", "+Z"), (15, "a", "+Z")],
        })

    def test_4_null_r_and_null_t_are_counted_non_accepted_validations(self):
        item = _item({"S0": AB}, {
            ("S0", "a"): _candidate(r_true=("-Z",), r_null=("+Z",), t_label=None),
            ("S0", "b"): _candidate(r_true=("+Z",), t_label=True),
        })
        actions, summary = _run(item)
        expected = [(1, AB, "a", "+Z", 0, None, None, False, False, 1, 0), (2, AB, "a", "-Z", 1, True, None, False, False, 1, 1)]
        expected += [(3 + index, AB, "a", direction, 2 + index, False, None, True, False, 1, 0) for index, direction in enumerate(DIRS[2:])]
        expected += [(7, AB, "b", "+Z", 6, True, True, False, True, 1, 1)]
        self.assertEqual(_observed(actions), expected)
        self.assertEqual(_summary(summary), {
            "action_validation_count": 7, "solution_action_validation": 7, "accepted_transition_count": 1,
            "T_cache_hit_count": 4, "censored_action_count": 6, "redmax_rollout_count": 7, "stability_restart_count": 2,
            "backtrack_count": 0, "search_node_count": 2, "pose_count": 1, "ranked_context_count": 1,
            "stop_reason": "solution_found", "max_action_validations": 8,
            "budgets": {"1": False, "2": False, "4": False, "8": True},
            "transitions": [(7, "b", "+Z")],
        })

    def test_5a_budget_exhausted_exactly_at_a_solving_transition(self):
        item = _item({"S0": AB}, {("S0", "b"): _candidate(r_true=("-Z",), t_label=True)})
        actions, summary = _run(item)
        expected = [(1 + index, AB, "a", direction, index, False, None, False, False, 1, 0) for index, direction in enumerate(DIRS)]
        expected += [(7, AB, "b", "+Z", 6, False, None, False, False, 1, 0), (8, AB, "b", "-Z", 7, True, True, False, True, 1, 1)]
        self.assertEqual(_observed(actions), expected)
        self.assertEqual(_summary(summary)["solution_action_validation"], 8)
        self.assertEqual(_summary(summary)["stop_reason"], "solution_found")
        self.assertEqual(_summary(summary)["budgets"], {"1": False, "2": False, "4": False, "8": True})

    def test_5b_budget_exhausted_at_a_transition_into_an_unsolved_state(self):
        item = _item({"S0": ABC, "S1": AB}, {("S0", "c"): _candidate(r_true=("-X",), t_label=True)})
        actions, summary = _run(item)
        self.assertEqual(len(actions), 16)
        self.assertEqual(_observed(actions)[-1], (16, ABC, "c", "-X", 15, True, True, False, True, 1, 1))
        self.assertEqual(_summary(summary)["solution_action_validation"], None)
        self.assertEqual(_summary(summary)["stop_reason"], "budget_reached")
        self.assertEqual(_summary(summary)["budgets"], {"1": False, "2": False, "4": False, "8": False})
        self.assertEqual(_summary(summary)["accepted_transition_count"], 1)

    def test_5c_budget_exhausted_inside_a_failed_branch(self):
        item = _item({"S0": ABC, "S1": BC}, {("S0", "a"): _candidate(r_true=("+Z",), t_label=True)})
        actions, summary = _run(item)
        expected = [(1, ABC, "a", "+Z", 0, True, True, False, True, 1, 1)]
        expected += [(2 + index, BC, part, direction, index + 6 * offset, False, None, False, False, 1, 0)
                     for offset, part in enumerate(("b", "c")) for index, direction in enumerate(DIRS)][:12]
        expected = expected[:1] + [(2 + position,) + row[1:] for position, row in enumerate(expected[1:])]
        expected += [(14 + index, ABC, "a", direction, 1 + index, False, True, True, False, 1, 0) for index, direction in enumerate(DIRS[1:4])]
        self.assertEqual(_observed(actions), expected)
        self.assertEqual(_summary(summary), {
            "action_validation_count": 16, "solution_action_validation": None, "accepted_transition_count": 1,
            "T_cache_hit_count": 3, "censored_action_count": 0, "redmax_rollout_count": 16, "stability_restart_count": 1,
            "backtrack_count": 1, "search_node_count": 2, "pose_count": 2, "ranked_context_count": 2,
            "stop_reason": "budget_reached", "max_action_validations": 16,
            "budgets": {"1": False, "2": False, "4": False, "8": False},
            "transitions": [(1, "a", "+Z")],
        })

    def test_unrecorded_state_and_missing_candidate_fail_closed(self):
        item = _item({"S0": ABC}, {("S0", "a"): _candidate(r_true=("+Z",), t_label=True)})
        with self.assertRaisesRegex(ValueError, "unrecorded state"):
            _run(item)
        store = replay.LabelStore("synthetic", {"S0": ABC}, {"S0": ["pose-000"]}, {("S0", "pose-000", "a"): _candidate()})
        with self.assertRaisesRegex(ValueError, "missing candidate"):
            store.coverage()

    def test_live_trace_comparison_reports_label_and_action_mismatches(self):
        item = _item({"S0": AB}, {("S0", "a"): _candidate(r_true=("+Z",), t_label=True)})
        actions, summary = _run(item)
        live = [dict(row) for row in actions]
        live[0] = dict(live[0], R=dict(live[0]["R"], label=None))
        mismatches, _ = replay.compare_trace(live, summary, actions, summary)
        self.assertEqual(mismatches, [{"action": 1, "field": "R.label", "live": None, "replay": True}])
        self.assertEqual(replay.compare_trace(actions, summary, actions, summary)[0], [])


if __name__ == "__main__":
    unittest.main()
