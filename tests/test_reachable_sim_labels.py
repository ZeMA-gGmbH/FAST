import json
import re
import tempfile
import unittest
from pathlib import Path

from plan_sequence import audit_reachable_sim_labels as audit
from plan_sequence import generate_reachable_sim_labels as reachable
from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_sdf_cache as cache_tools


POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FakeOracle:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def evaluate_removability(self, parts, part_move, _pose):
        if self.fail:
            raise AssertionError("completed candidate must be reused")
        self.calls.append(("R", tuple(parts), part_move))
        removable = (tuple(parts), part_move) in {
            (("0", "1", "2"), "0"),
            (("1", "2"), "1"),
        }
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

    def evaluate_stability(self, parts_after, _pose):
        if self.fail:
            raise AssertionError("completed stability query must be reused")
        self.calls.append(("T", tuple(parts_after), None))
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


class ReachableLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "source"
        self.output_root = self.root / "output"
        self.assembly = self.source_root / "training_assembly" / "00001"
        self.assembly.mkdir(parents=True)
        for part in ("0", "1", "2"):
            (self.assembly / (part + ".obj")).write_text(
                "o part\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
        (self.assembly / "contact_graph.json").write_text("{}\n", encoding="utf-8")
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
                    "schema_version": reachable.MANIFEST_SCHEMA_VERSION,
                    "label_schema_version": reachable.SCHEMA_VERSION,
                    "phase": "test-reachable",
                    "protocol": reachable._PROTOCOL,
                    "parameters": {
                        "max_poses": 3,
                        "pose_probability": 0.9,
                        "removal_timeout_s": 60.0,
                        "stability_timeout_s": 900.0,
                        "save_sdf": True,
                        "max_states": 7,
                    },
                    "assembly": {
                        "path": "training_assembly/00001",
                        "part_count": 3,
                        "part_ids": ["0", "1", "2"],
                        "input_sha256": self.input_sha256,
                    },
                    "cache": "cache/sdf/{}".format(self.input_sha256),
                    "root_reference": "labels/root-reference/training_assembly/00001",
                    "output_root": "labels/reachable/training_assembly/00001",
                    "audit_output": "runs/reachable/audit.json",
                    "expected": {
                        "root_candidate_count": 9,
                        "root_action_count": 54,
                        "theoretical_max_nonempty_states": 7,
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
    def _pose_factory(_assembly, parts, _max_poses, _probability):
        return [(POSE, 1.0)]

    def _copy_root_reference(self, prepared):
        root = tuple(prepared["part_ids"])
        for part_move in root:
            source = reachable._candidate_path(prepared, root, 0, part_move)
            target = prepared["reference_root"] / "pose-000" / ("part-{}.jsonl".format(part_move))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

    def test_bfs_resume_and_root_audit(self):
        prepared = reachable._validate_manifest(
            self.manifest_path, self.source_root, self.output_root
        )
        oracle = FakeOracle()
        pose_calls = []

        def pose_factory(*args):
            pose_calls.append(tuple(args[1]))
            return self._pose_factory(*args)

        result = reachable.run(prepared, oracle=oracle, pose_factory=pose_factory)
        self.assertEqual(result["accounting"]["state_count"], 3)
        self.assertEqual(result["accounting"]["evaluated_state_count"], 2)
        self.assertEqual(result["accounting"]["terminal_state_count"], 1)
        self.assertEqual(result["accounting"]["candidate_count"], 5)
        self.assertEqual(result["accounting"]["candidate_actions"], 30)
        self.assertEqual(result["accounting"]["stability_queries"], 2)
        self.assertEqual(result["accounting"]["stability_queries_avoided"], 3)
        self.assertEqual(result["accounting"]["oracle_rollouts_total"], 32)
        self.assertEqual(pose_calls, [("0", "1", "2"), ("1", "2")])
        self.assertEqual(len([call for call in oracle.calls if call[0] == "R"]), 5)
        self.assertEqual(len([call for call in oracle.calls if call[0] == "T"]), 2)
        self.assertFalse(any(self.assembly.glob("*.sdf")))

        prepared["summary_path"].unlink()
        resumed = reachable.run(
            prepared,
            oracle=FakeOracle(fail=True),
            pose_factory=lambda *_: (_ for _ in ()).throw(
                AssertionError("state poses must be reused")
            ),
        )
        self.assertEqual(resumed["current_run"]["state_pose_checkpoints_reused"], 2)
        self.assertEqual(resumed["current_run"]["candidate_checkpoints_reused"], 5)
        self.assertEqual(resumed["current_run"]["candidate_checkpoints_created"], 0)

        self._copy_root_reference(prepared)
        report = audit.audit(self.manifest_path, self.source_root, self.output_root)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["integrity_status"], "passed")
        self.assertEqual(report["root_reference"]["mismatch_count"], 0)
        self.assertEqual(report["observed"]["candidate_count"], 5)

    def test_production_audit_does_not_require_a_pilot_root_reference(self):
        prepared = reachable._validate_manifest(
            self.manifest_path, self.source_root, self.output_root
        )
        reachable.run(prepared, oracle=FakeOracle(), pose_factory=self._pose_factory)
        report = audit.audit(
            self.manifest_path,
            self.source_root,
            self.output_root,
            check_root_reference=False,
        )
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["root_reference"]["checked"])


if __name__ == "__main__":
    unittest.main()
