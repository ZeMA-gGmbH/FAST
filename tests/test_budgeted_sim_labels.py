import json
import re
import tempfile
import unittest
from pathlib import Path

from plan_sequence import audit_budgeted_sim_labels as audit
from plan_sequence import audit_adaptive_sim_labels as adaptive_audit
from plan_sequence import audit_greedy_sim_labels as greedy_audit
from plan_sequence import dfs_search
from plan_sequence import generate_budgeted_sim_labels as budgeted
from plan_sequence import generate_adaptive_sim_labels as adaptive
from plan_sequence import generate_greedy_sim_labels as greedy
from plan_sequence import generate_sim_labels as labels
from plan_sequence import mixed_search
from plan_sequence import prepare_sdf_cache as cache_tools


POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FakeOracle:
    def __init__(self, fail=False, always_removable=False):
        self.fail = fail
        self.always_removable = always_removable

    def evaluate_removability(self, parts, part_move, _pose):
        if self.fail:
            raise AssertionError("completed candidate must be reused")
        removable = self.always_removable or part_move == min(parts)
        directions = []
        for index, (name, vector) in enumerate(labels.A6):
            success = removable and index == 0
            directions.append(
                {
                    "direction": name,
                    "action": list(vector),
                    "label": success,
                    "status": "success" if success else "blocked",
                    "reason": "separated" if success else "blocked",
                    "rollout_count": 1,
                    "runtime_s": 0.01,
                    "forward_steps": 1,
                    "path_length": 1,
                }
            )
        return {
            "label": removable,
            "status": "success" if removable else "blocked",
            "reason": "some_direction_succeeded" if removable else "all_directions_blocked",
            "rollout_count": 6,
            "runtime_s": 0.06,
            "forward_steps": 6,
            "directions": directions,
        }

    def evaluate_stability(self, _parts_after, _pose):
        if self.fail:
            raise AssertionError("completed stability query must be reused")
        return {
            "label": True,
            "status": "success",
            "reason": "stable",
            "rollout_count": 1,
            "runtime_s": 0.02,
            "forward_steps": 1,
            "parts_fix": [],
            "rollouts": [],
        }


class MixedSearchTest(unittest.TestCase):
    def test_rank_strata_and_state_lanes(self):
        self.assertEqual(
            mixed_search.stratified_parts(list("abcdefg")),
            ["a", "d", "g", "b", "e", "c", "f"],
        )
        root = tuple("abcd")
        child = tuple("bcd")
        ranker = lambda state: list(state)
        self.assertEqual(
            mixed_search.select_next(root, {root}, {}, root, 0, ranker)[:3],
            ("spine", root, "a"),
        )
        attempted = {root: ["a"], child: ["b"]}
        self.assertEqual(
            mixed_search.select_next(root, {root, child}, attempted, None, 0, ranker)[:3],
            ("breadth", root, "c"),
        )
        self.assertEqual(
            mixed_search.select_next(root, {root, child}, attempted, None, 1, ranker)[:3],
            ("depth", child, "c"),
        )
        adaptive = mixed_search.select_next_adaptive(
            root, {root, child}, attempted, None, 0, ranker, 0
        )
        self.assertEqual(adaptive[0], "breadth")
        self.assertEqual(adaptive[4], "seeded_uniform")
        self.assertEqual(adaptive[5], 1.0 / 3.0)
        path = [root, child]
        self.assertEqual(
            dfs_search.select_next(path, attempted, ranker)[:3],
            ("dfs", child, "c"),
        )


class BudgetedLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "source"
        self.output_root = self.root / "output"
        self.assembly = self.source_root / "training_assembly" / "00001"
        self.assembly.mkdir(parents=True)
        for index in range(7):
            offset = float(index * 3)
            vertices = [
                (offset, 0, 0),
                (offset + 1, 0, 0),
                (offset, 1, 0),
                (offset, 0, 1),
            ]
            text = "o part\n" + "".join(
                "v {} {} {}\n".format(*vertex) for vertex in vertices
            )
            text += "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
            (self.assembly / "{}.obj".format(index)).write_text(text, encoding="utf-8")
        (self.assembly / "contact_graph.json").write_text(
            '{"links": []}\n', encoding="utf-8"
        )
        self.input_sha256 = labels._hash_assembly(self.assembly)[0]
        self.cache_dir = self.output_root / "cache" / "sdf" / self.input_sha256
        cache_tools.prepare_cache(
            self.assembly,
            self.cache_dir,
            simulation_factory=self._fake_simulation,
        )
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": budgeted.MANIFEST_SCHEMA_VERSION,
                    "label_schema_version": "fast.reachable-sim-labels.v1",
                    "phase": "test-budgeted",
                    "protocol": budgeted._PROTOCOL,
                    "parameters": {
                        "max_poses": 3,
                        "pose_probability": 0.9,
                        "removal_timeout_s": 60.0,
                        "stability_timeout_s": 900.0,
                        "save_sdf": True,
                        "seed": 0,
                    },
                    "assembly": {
                        "path": "training_assembly/00001",
                        "part_count": 7,
                        "part_ids": [str(index) for index in range(7)],
                        "input_sha256": self.input_sha256,
                    },
                    "cache": "cache/sdf/{}".format(self.input_sha256),
                    "output_root": "labels/budgeted/training_assembly/00001",
                    "audit_output": "runs/budgeted/audit.json",
                    "expected": {
                        "state_part_attempt_budget": 24,
                        "candidate_evaluation_budget": 72,
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _fake_simulation(model, _asset_folder):
        for obj_path in re.findall(r'filename="([^"]+[.]obj)"', model):
            Path(obj_path).with_suffix(".sdf").write_bytes(
                ("sdf:" + Path(obj_path).stem).encode("ascii")
            )

    @staticmethod
    def _pose_factory(_assembly, _parts, _max_poses, _probability):
        return [(POSE, 1.0)]

    def test_budget_resume_and_audit(self):
        prepared = budgeted._validate_manifest(
            self.manifest_path, self.source_root, self.output_root
        )
        result = budgeted.run(
            prepared, oracle=FakeOracle(), pose_factory=self._pose_factory
        )
        self.assertEqual(result["budget"]["stop_reason"], "budget_reached")
        self.assertEqual(result["accounting"]["state_part_attempt_count"], 24)
        self.assertEqual(result["accounting"]["candidate_count"], 24)
        self.assertTrue(result["coverage"]["complete_sequence_found"])
        self.assertGreater(result["coverage"]["lane_counts"]["breadth"], 0)
        self.assertGreater(result["coverage"]["lane_counts"]["depth"], 0)
        self.assertGreater(result["coverage"]["heuristic_rank_counts"]["non_top"], 0)
        self.assertFalse(any(self.assembly.glob("*.sdf")))

        prepared["summary_path"].unlink()
        resumed = budgeted.run(
            prepared,
            oracle=FakeOracle(fail=True),
            pose_factory=lambda *_: (_ for _ in ()).throw(
                AssertionError("state poses must be reused")
            ),
        )
        self.assertEqual(resumed["current_run"]["candidate_checkpoints_reused"], 24)
        self.assertEqual(resumed["current_run"]["candidate_checkpoints_created"], 0)

        report = audit.audit(self.manifest_path, self.source_root, self.output_root)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["integrity_status"], "passed")
        self.assertEqual(report["observed"]["state_part_attempt_count"], 24)

    def test_budget_formula_is_frozen(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["expected"]["candidate_evaluation_budget"] = 71
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"12[*]"):
            budgeted._validate_manifest(
                self.manifest_path, self.source_root, self.output_root
            )

    def test_audit_uses_recorded_rank_and_reports_orphan_checkpoints(self):
        prepared = budgeted._validate_manifest(
            self.manifest_path, self.source_root, self.output_root
        )
        default_ranker = budgeted._heur_out_ranker(self.assembly, 0)

        def shifted_ranker(state):
            ranked = list(default_ranker(state))
            return ranked[1:] + ranked[:1]

        result = budgeted.run(
            prepared,
            oracle=FakeOracle(),
            pose_factory=self._pose_factory,
            ranker=shifted_ranker,
        )
        self.assertNotEqual(
            result["schedule"][0]["part_move"],
            list(default_ranker(tuple(prepared["part_ids"])))[0],
        )
        orphan = prepared["labels_root"] / "state-interrupted-orphan"
        orphan.mkdir()

        report = audit.audit(self.manifest_path, self.source_root, self.output_root)
        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["integrity_status"], "passed")
        self.assertFalse(report["policy_checks"]["no_orphan_state_checkpoints"])
        self.assertEqual(report["orphan_state_checkpoints"], [orphan.name])

    def test_adaptive_worker_stops_on_time_and_first_feasible_pose(self):
        adaptive_manifest = self.root / "adaptive.json"
        adaptive_manifest.write_text(
            json.dumps(
                {
                    "schema_version": adaptive.MANIFEST_SCHEMA_VERSION,
                    "label_schema_version": "fast.reachable-sim-labels.v1",
                    "phase": "test-adaptive",
                    "protocol": adaptive._PROTOCOL,
                    "parameters": {
                        "max_poses": 3,
                        "pose_probability": 0.9,
                        "removal_timeout_s": 60.0,
                        "stability_timeout_s": 900.0,
                        "candidate_time_budget_s": 1e-12,
                        "save_sdf": True,
                        "seed": 0,
                    },
                    "assembly": {
                        "path": "training_assembly/00001",
                        "part_count": 7,
                        "part_ids": [str(index) for index in range(7)],
                        "input_sha256": self.input_sha256,
                    },
                    "cache": "cache/sdf/{}".format(self.input_sha256),
                    "output_root": "labels/adaptive/training_assembly/00001",
                    "audit_output": "runs/adaptive/audit.json",
                    "expected": {
                        "state_part_attempt_budget": 12,
                        "candidate_evaluation_budget": 36,
                    },
                }
            ),
            encoding="utf-8",
        )

        def three_poses(_assembly, _parts, _max_poses, _probability):
            return [(POSE, 1.0 / 3.0)] * 3

        prepared = adaptive._validate_manifest(
            adaptive_manifest, self.source_root, self.output_root
        )
        result = budgeted.run(
            prepared,
            oracle=FakeOracle(always_removable=True),
            pose_factory=three_poses,
        )
        self.assertEqual(
            result["budget"]["stop_reason"], "candidate_time_budget_reached"
        )
        self.assertEqual(result["accounting"]["candidate_count"], 1)
        self.assertEqual(result["schedule"][0]["evaluated_pose_count"], 1)
        self.assertTrue(result["schedule"][0]["part_attempt_complete"])
        self.assertEqual(
            result["provenance"]["scheduler_sha256"], adaptive._scheduler_sha256()
        )
        self.assertEqual(result["complexity"]["part_count"], 7)

        report = adaptive_audit.audit(
            adaptive_manifest, self.source_root, self.output_root
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["integrity_status"], "passed")
        self.assertTrue(report["outcomes"]["complete_sequence_found"] is False)


class GreedyLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "source"
        self.output_root = self.root / "output"
        self.assembly = self.source_root / "training_assembly" / "00002"
        self.assembly.mkdir(parents=True)
        for index in range(21):
            offset = float(index * 3)
            vertices = [
                (offset, 0, 0),
                (offset + 1, 0, 0),
                (offset, 1, 0),
                (offset, 0, 1),
            ]
            text = "o part\n" + "".join(
                "v {} {} {}\n".format(*vertex) for vertex in vertices
            )
            text += "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
            (self.assembly / "{}.obj".format(index)).write_text(text, encoding="utf-8")
        (self.assembly / "contact_graph.json").write_text("{}\n", encoding="utf-8")
        self.input_sha256 = labels._hash_assembly(self.assembly)[0]
        self.cache_dir = self.output_root / "cache" / "sdf" / self.input_sha256
        cache_tools.prepare_cache(
            self.assembly,
            self.cache_dir,
            simulation_factory=BudgetedLabelsTest._fake_simulation,
        )
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": greedy.MANIFEST_SCHEMA_VERSION,
                    "label_schema_version": "fast.reachable-sim-labels.v1",
                    "phase": "test-greedy",
                    "protocol": greedy._PROTOCOL,
                    "parameters": {
                        "max_poses": 3,
                        "pose_probability": 0.9,
                        "removal_timeout_s": 60.0,
                        "stability_timeout_s": 900.0,
                        "save_sdf": True,
                        "seed": 0,
                    },
                    "assembly": {
                        "path": "training_assembly/00002",
                        "part_count": 21,
                        "part_ids": sorted(str(index) for index in range(21)),
                        "input_sha256": self.input_sha256,
                    },
                    "cache": "cache/sdf/{}".format(self.input_sha256),
                    "output_root": "labels/greedy/training_assembly/00002",
                    "audit_output": "runs/greedy/audit.json",
                    "expected": {
                        "state_part_attempt_budget": 400,
                        "candidate_evaluation_budget": 400,
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _pose_factory(_assembly, _parts, _max_poses, _probability):
        poses = []
        for translation in (0.0, 1.0, 2.0):
            pose = [list(row) for row in POSE]
            pose[0][3] = translation
            poses.append((pose, 1.0 / 3.0))
        return poses

    def test_dfs_stops_after_first_feasible_pose_and_audits(self):
        prepared = greedy._validate_manifest(
            self.manifest_path, self.source_root, self.output_root
        )
        result = budgeted.run(
            prepared, oracle=FakeOracle(), pose_factory=self._pose_factory
        )
        self.assertEqual(result["budget"]["stop_reason"], "solution_found")
        self.assertTrue(result["coverage"]["complete_sequence_found"])
        self.assertEqual(
            result["coverage"]["lane_counts"]["dfs"],
            result["accounting"]["state_part_attempt_count"],
        )
        self.assertLessEqual(result["accounting"]["candidate_count"], 400)
        self.assertTrue(
            any(
                step["feasible"] and step["evaluated_pose_count"] == 1
                for step in result["schedule"]
            )
        )
        report = greedy_audit.audit(
            self.manifest_path, self.source_root, self.output_root
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["integrity_status"], "passed")


if __name__ == "__main__":
    unittest.main()
