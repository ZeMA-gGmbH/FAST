import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plan_sequence import audit_conditional_sim_labels as audit
from plan_sequence import generate_conditional_sim_labels as conditional
from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_sdf_cache as cache_tools
from plan_sequence import prepare_pose_cache as pose_cache_tools


POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FakeOracle:
    def __init__(self, removable_parts=("0",), timeout_part=None):
        self.removable_parts = set(removable_parts)
        self.timeout_part = timeout_part
        self.calls = []

    def evaluate_removability(self, _parts, part_move, _pose):
        self.calls.append(("R", part_move))
        directions = []
        for index, (name, vector) in enumerate(labels.A6):
            if part_move == self.timeout_part and index == 0:
                label, status, reason = None, "timeout", "direction_timeout"
            elif part_move in self.removable_parts and index == 0:
                label, status, reason = True, "success", "separated"
            else:
                label, status, reason = False, "blocked", "blocked"
            directions.append(
                {
                    "direction": name,
                    "action": list(vector),
                    "label": label,
                    "status": status,
                    "reason": reason,
                    "rollout_count": 1,
                    "runtime_s": 1.0,
                    "forward_steps": 10,
                    "path_length": 2,
                }
            )
        any_true = part_move in self.removable_parts
        return {
            "label": True if any_true else None if part_move == self.timeout_part else False,
            "status": "success" if any_true else "timeout" if part_move == self.timeout_part else "blocked",
            "reason": "some_direction_succeeded" if any_true else "no_direction_succeeded",
            "rollout_count": 6,
            "runtime_s": 6.0,
            "forward_steps": 60,
            "directions": directions,
        }

    def evaluate_stability(self, _parts_after, _pose):
        self.calls.append(("T", None))
        return {
            "label": True,
            "status": "success",
            "reason": "stable",
            "rollout_count": 1,
            "runtime_s": 2.0,
            "forward_steps": 1000,
            "parts_fix": [],
            "rollouts": [],
        }


class ConditionalLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "source"
        self.output_root = self.root / "output"
        self.assembly = self.source_root / "training_assembly" / "00001"
        self.assembly.mkdir(parents=True)
        for part in ("0", "1"):
            (self.assembly / (part + ".obj")).write_text(
                "o part\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
        (self.assembly / "contact_graph.json").write_text("{}\n", encoding="utf-8")
        self.input_sha256 = labels._hash_assembly(self.assembly)[0]
        self.cache_dir = self.output_root / "cache" / "sdf" / self.input_sha256
        cache_tools.prepare_cache(
            self.assembly,
            self.cache_dir,
            simulation_factory=self._fake_simulation,
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _fake_simulation(model, _asset_folder):
        for obj_path in re.findall(r'filename="([^"]+[.]obj)"', model):
            Path(obj_path).with_suffix(".sdf").write_bytes(
                ("sdf:" + Path(obj_path).stem).encode("ascii")
            )

    def _manifest(self, expected=None, cached_poses=False):
        expected = expected or {"candidate_count": 2, "action_count": 12}
        value = {
            "schema_version": (
                conditional.MANIFEST_SCHEMA_VERSION_V2
                if cached_poses
                else conditional.MANIFEST_SCHEMA_VERSION
            ),
            "label_schema_version": (
                conditional.SCHEMA_VERSION_V2
                if cached_poses
                else conditional.SCHEMA_VERSION
            ),
            "phase": "test-conditional",
            "protocol": conditional._PROTOCOL_V2 if cached_poses else conditional._PROTOCOL,
            "parameters": {
                "max_poses": 1,
                "pose_probability": 0.9,
                "removal_timeout_s": 60.0,
                "stability_timeout_s": 900.0,
                "save_sdf": True,
            },
            "assembly": {
                "path": "training_assembly/00001",
                "part_count": 2,
                "part_ids": ["0", "1"],
                "input_sha256": self.input_sha256,
            },
            "cache": "cache/sdf/{}".format(self.input_sha256),
            "baseline": "labels/baseline.jsonl",
            "output_root": "labels/conditional/training_assembly/00001",
            "audit_output": "runs/conditional/audit.json",
            "expected": expected,
        }
        if cached_poses:
            value["pose_cache"] = "cache/stable-poses/{}".format(self.input_sha256)
            value["reference_labels"] = "labels/reference/training_assembly/00001"
        path = self.root / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _prepared(self, expected=None):
        return conditional._validate_manifest(
            self._manifest(expected), self.source_root, self.output_root
        )

    @staticmethod
    def _records(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_candidate_mapping_and_conditional_stability(self):
        prepared = self._prepared()
        self.assertEqual(
            [(item["pose_index"], item["part_move"]) for item in prepared["candidates"]],
            [(0, "0"), (0, "1")],
        )
        oracle = FakeOracle()
        with patch.object(conditional.labels, "_compute_stable_poses", return_value=[(POSE, 1.0)]):
            conditional.run_candidate(prepared, 0, oracle=oracle)
        records = self._records(prepared["candidates"][0]["output_path"])
        stability = records[1]
        actions = records[2:8]
        summary = records[-1]
        self.assertEqual(oracle.calls, [("R", "0"), ("T", None)])
        self.assertTrue(stability["queried"])
        self.assertTrue(stability["T"])
        self.assertEqual([action["Y"] for action in actions], [True] + [False] * 5)
        self.assertEqual(summary["accounting"]["stability_queries"], 1)
        self.assertEqual(summary["accounting"]["oracle_rollouts_total"], 7)
        self.assertFalse(any(self.assembly.glob("*.sdf")))

    def test_no_proven_removal_skips_stability_and_preserves_unknown(self):
        prepared = self._prepared()
        oracle = FakeOracle(removable_parts=(), timeout_part="1")
        with patch.object(conditional.labels, "_compute_stable_poses", return_value=[(POSE, 1.0)]):
            conditional.run_candidate(prepared, 1, oracle=oracle)
        records = self._records(prepared["candidates"][1]["output_path"])
        stability = records[1]
        actions = records[2:8]
        self.assertEqual(oracle.calls, [("R", "1")])
        self.assertFalse(stability["queried"])
        self.assertIsNone(stability["T"])
        self.assertEqual(stability["T_status"], "not_evaluated")
        self.assertIsNone(actions[0]["Y"])
        self.assertTrue(all(action["Y"] is False for action in actions[1:]))
        self.assertEqual(records[-1]["accounting"]["stability_queries"], 0)

    def test_v2_candidate_uses_validated_pose_cache_without_recomputing(self):
        pose_cache_tools.prepare_cache(
            self.assembly,
            self.output_root / "cache" / "stable-poses" / self.input_sha256,
            max_poses=1,
            probability_threshold=0.9,
            pose_factory=lambda *_: [(POSE, 1.0)],
        )
        manifest_path = self._manifest(cached_poses=True)
        prepared = conditional._validate_manifest(
            manifest_path, self.source_root, self.output_root
        )
        with patch.object(
            conditional.labels,
            "_compute_stable_poses",
            side_effect=AssertionError("worker must use the pose cache"),
        ):
            conditional.run_candidate(prepared, 0, oracle=FakeOracle())
        records = self._records(prepared["candidates"][0]["output_path"])
        self.assertEqual(records[0]["schema_version"], conditional.SCHEMA_VERSION_V2)
        self.assertEqual(records[0]["pose"], [list(row) for row in POSE])
        self.assertIn("stable_pose_cache", records[0])
        self.assertEqual(records[-1]["timings_s"]["stable_pose_search_wall_s"], 0.0)

    def test_audit_compares_all_candidates_with_baseline(self):
        expected = {
            "candidate_count": 2,
            "action_count": 12,
            "candidates_with_removable_direction": 1,
            "stability_queries": 1,
            "stability_queries_avoided": 1,
            "stability_rollouts": 1,
            "oracle_rollouts_total": 13,
            "label_counts": {
                "R": {"true": 1, "false": 11, "null": 0},
                "T": {"true": 1, "false": 0, "null": 1},
                "Y": {"true": 1, "false": 11, "null": 0},
            },
        }
        manifest_path = self._manifest(expected)
        prepared = conditional._validate_manifest(
            manifest_path, self.source_root, self.output_root
        )
        for index in range(2):
            with patch.object(
                conditional.labels, "_compute_stable_poses", return_value=[(POSE, 1.0)]
            ):
                conditional.run_candidate(prepared, index, oracle=FakeOracle())

        generated = [
            self._records(candidate["output_path"]) for candidate in prepared["candidates"]
        ]
        generated[1][0]["provenance"]["runtime"]["platform"] = "different-worker-kernel"
        prepared["candidates"][1]["output_path"].write_text(
            "".join(
                labels._canonical_json(record) + "\n" for record in generated[1]
            ),
            encoding="utf-8",
        )
        baseline = [
            {
                "record_type": "header",
                "provenance": generated[0][0]["provenance"],
            }
        ]
        for records in generated:
            header, stability = records[0], records[1]
            baseline_t = dict(stability)
            baseline_t.update(
                {
                    "record_type": "stability",
                    "removed_part": header["part_move"],
                    "pose_id": header["pose_id"],
                    "pose": header["pose"],
                    "pose_probability": header["pose_probability"] - 3e-15,
                }
            )
            baseline.append(baseline_t)
            baseline.extend(records[2:8])
        baseline.append({"record_type": "summary", "complete": True})
        prepared["baseline_path"].parent.mkdir(parents=True, exist_ok=True)
        prepared["baseline_path"].write_text(
            "".join(labels._canonical_json(record) + "\n" for record in baseline),
            encoding="utf-8",
        )

        report = audit.audit(manifest_path, self.source_root, self.output_root)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["schema_version"], audit.SCHEMA_VERSION)
        self.assertEqual(
            report["pose_comparison"]["probability_abs_tolerance"], 1e-12
        )
        self.assertEqual(report["mismatch_count"], 0)
        self.assertEqual(report["observed"], expected)
        self.assertEqual(report["timings_s"]["total_wall_s"]["count"], 2)
        self.assertEqual(report["savings_vs_baseline"]["stability_queries"], 1)
        override = "runs/conditional/audit-v2.json"
        repeated = audit.audit(
            manifest_path,
            self.source_root,
            self.output_root,
            audit_output=override,
        )
        self.assertEqual(repeated["spec_sha256"], report["spec_sha256"])
        self.assertTrue((self.output_root / override).is_file())
        with self.assertRaisesRegex(ValueError, "must differ"):
            audit.audit(
                manifest_path,
                self.source_root,
                self.output_root,
                audit_output="runs/conditional/audit.json",
            )

if __name__ == "__main__":
    unittest.main()
