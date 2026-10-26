import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plan_sequence import generate_sim_labels as labels


POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FakeOracle:
    def __init__(self):
        self.stability_calls = []
        self.removability_calls = []

    def evaluate_stability(self, parts_after, pose):
        self.stability_calls.append((tuple(parts_after), pose))
        removed = ({"0", "1", "2", "3"} - set(parts_after)).pop()
        label, status, rollouts = {
            "0": (True, "success", 2),
            "1": (None, "timeout", 3),
            "2": (None, "error", 1),
            "3": (False, "failure", 2),
        }[removed]
        result = {
            "label": label,
            "status": status,
            "rollout_count": rollouts,
            "runtime_s": 0.01,
        }
        if status == "error":
            result["error"] = {"type": "FakeError", "message": "stability failed"}
        return result

    def evaluate_removability(self, parts, part_move, pose):
        self.removability_calls.append((tuple(parts), part_move, pose))
        values = {
            "0": ((True, "success"),) + ((False, "failure"),) * 5,
            "1": ((False, "failure"),) * 6,
            "2": (
                (True, "success"),
                (None, "timeout"),
                (None, "error"),
                (False, "failure"),
                (True, "success"),
                (False, "failure"),
            ),
            "3": (
                (True, "success"),
                (None, "timeout"),
                (None, "error"),
                (False, "failure"),
                (True, "success"),
                (False, "failure"),
            ),
        }[part_move]
        directions = []
        for (_, vector), (label, status) in zip(labels.A6, values):
            result = {
                "action": list(vector),
                "label": label,
                "status": status,
                "rollout_count": 1,
                "runtime_s": 0.02,
            }
            if status == "error":
                result["error"] = {"type": "FakeError", "message": "removal failed"}
            directions.append(result)
        return {"directions": list(reversed(directions))}


class GenerateSimulationLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.assembly = root / "training_assembly" / "00001"
        self.assembly.mkdir(parents=True)
        for part in ("0", "1", "2", "3"):
            (self.assembly / (part + ".obj")).write_text(
                "o part\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
        self.output = root / "labels" / "00001.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def run_worker(self, oracle=None):
        oracle = oracle or FakeOracle()
        with patch.object(labels, "_compute_stable_poses", return_value=[(POSE, 1.0)]):
            summary = labels.run_job(
                self.assembly, self.output, max_poses=1, oracle=oracle
            )
        return oracle, summary

    def read_records(self):
        return [
            json.loads(line)
            for line in self.output.read_text(encoding="utf-8").splitlines()
        ]

    def test_complete_labels_order_references_algebra_and_accounting(self):
        oracle, summary = self.run_worker()
        records = self.read_records()
        stability = {
            record["removed_part"]: record
            for record in records
            if record["record_type"] == "stability"
        }
        actions = [record for record in records if record["record_type"] == "action"]

        self.assertEqual([record["record_type"] for record in records[:2]], ["header", "stability"])
        self.assertFalse(records[0]["input"]["config_present"])
        self.assertEqual(records[0]["input"]["transform_source"], "baked_obj_identity")
        self.assertEqual(records[0]["protocol"]["support"]["mode"], "ground_plane")
        self.assertFalse((self.assembly / "config.json").exists())
        self.assertEqual(list(self.assembly.glob("*.sdf")), [])
        self.assertEqual(records[-1]["record_type"], "summary")
        self.assertEqual(len(oracle.stability_calls), 4)
        self.assertEqual(len(oracle.removability_calls), 4)
        self.assertEqual(len(stability), 4)
        self.assertEqual(len(actions), 24)

        expected_ids = [direction_id for direction_id, _ in labels.A6]
        expected_vectors = [list(vector) for _, vector in labels.A6]
        for offset in range(0, len(actions), 6):
            group = actions[offset : offset + 6]
            self.assertEqual([record["direction_id"] for record in group], expected_ids)
            self.assertEqual([record["direction"] for record in group], expected_vectors)

        by_part = {}
        for action in actions:
            by_part.setdefault(action["part_move"], []).append(action)
            stable = stability[action["part_move"]]
            self.assertEqual(action["stability_ref"], stable["stability_id"])
            self.assertIs(action["T"], stable["T"])
            expected_y = (
                False
                if action["R"] is False or action["T"] is False
                else True
                if action["R"] is True and action["T"] is True
                else None
            )
            self.assertIs(action["Y"], expected_y)

        self.assertTrue(all(action["R"] is False for action in by_part["1"]))
        self.assertIsNone(stability["1"]["T"])
        self.assertEqual(stability["1"]["T_status"], "timeout")
        self.assertTrue(all(action["Y"] is False for action in by_part["1"]))
        self.assertIsNone(stability["2"]["T"])
        self.assertEqual(stability["2"]["T_status"], "error")
        self.assertIsNone(by_part["2"][0]["Y"])
        self.assertEqual(by_part["2"][0]["Y_status"], "error")
        self.assertIsNone(by_part["2"][1]["R"])
        self.assertEqual(by_part["2"][1]["R_status"], "timeout")
        self.assertIsNone(by_part["2"][1]["Y"])
        self.assertTrue(all(action["Y"] is False for action in by_part["3"]))

        self.assertEqual(
            summary["accounting"],
            {
                "candidate_actions": 24,
                "removability_batches": 4,
                "removability_rollouts": 24,
                "stability_queries": 4,
                "stability_rollouts": 8,
                "oracle_rollouts_total": 32,
                "legacy_asap_part_evaluations": 4,
            },
        )
        self.assertEqual(summary["label_counts"]["R"], {"true": 5, "false": 15, "null": 4})
        self.assertEqual(summary["label_counts"]["T"], {"true": 1, "false": 1, "null": 2})
        self.assertEqual(summary["label_counts"]["Y"], {"true": 1, "false": 19, "null": 4})
        self.assertEqual(summary["record_count"], 30)

    def test_atomic_idempotent_output_and_mismatch_rejection(self):
        with self.assertRaisesRegex(ValueError, "authoritative source split"):
            with patch.object(labels, "_compute_stable_poses", return_value=[(POSE, 1.0)]):
                labels.run_job(
                    self.assembly,
                    self.assembly.parent / "labels.jsonl",
                    max_poses=1,
                    oracle=FakeOracle(),
                )

        with self.assertRaisesRegex(ValueError, "label=None"):
            labels._normalise_result({"label": False, "status": "timeout"})

        first_oracle, _ = self.run_worker()
        original = self.output.read_bytes()
        second_oracle, second = self.run_worker()

        self.assertEqual(len(first_oracle.stability_calls), 4)
        self.assertTrue(second["idempotent"])
        self.assertEqual(second_oracle.stability_calls, [])
        self.assertEqual(second_oracle.removability_calls, [])
        self.assertEqual(self.output.read_bytes(), original)

        (self.assembly / "0.obj").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_worker()
        self.assertEqual(self.output.read_bytes(), original)

        atomic_target = self.output.parent / "atomic.jsonl"
        atomic_target.write_text("sentinel\n", encoding="utf-8")

        def broken_records():
            yield {"record_type": "header"}
            raise RuntimeError("interrupted")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            labels._write_atomic(atomic_target, broken_records())
        self.assertEqual(atomic_target.read_text(encoding="utf-8"), "sentinel\n")

        with self.assertRaises(FileExistsError):
            labels._write_atomic(atomic_target, [{"record_type": "header"}])
        self.assertEqual(atomic_target.read_text(encoding="utf-8"), "sentinel\n")
        self.assertEqual(list(atomic_target.parent.glob(".atomic.jsonl-*.tmp")), [])

    def test_stale_sdf_is_neither_hashed_nor_copied_to_scratch(self):
        stale = self.assembly / "0.sdf"
        stale.write_text("stale cache", encoding="utf-8")
        scratch_sdfs = []

        def inspect_scratch(assembly_dir, parts, max_poses, probability_threshold):
            scratch_sdfs.extend(Path(assembly_dir).glob("*.sdf"))
            return [(POSE, 1.0)]

        with patch.object(labels, "_compute_stable_poses", side_effect=inspect_scratch):
            labels.run_job(self.assembly, self.output, max_poses=1, oracle=FakeOracle())

        header = self.read_records()[0]
        self.assertEqual(scratch_sdfs, [])
        self.assertTrue(stale.exists())
        self.assertFalse(any(item["path"].endswith(".sdf") for item in header["input"]["files"]))

    def test_frozen_input_hash_is_rechecked_by_the_worker(self):
        with self.assertRaisesRegex(ValueError, "input SHA-256"):
            labels.run_job(
                self.assembly,
                self.output,
                max_poses=1,
                oracle=FakeOracle(),
                expected_input_sha256="0" * 64,
            )
        self.assertFalse(self.output.exists())

    def test_manifest_identity_is_persisted_in_the_shard(self):
        context = {
            "schema_version": "fast.sim-label-manifest.v1",
            "sha256": "1" * 64,
            "phase": "test-phase",
            "job_id": "n004-00001",
            "index": 0,
        }
        with patch.object(labels, "_compute_stable_poses", return_value=[(POSE, 1.0)]):
            labels.run_job(
                self.assembly,
                self.output,
                max_poses=1,
                oracle=FakeOracle(),
                manifest_context=context,
            )
        self.assertEqual(self.read_records()[0]["manifest"], context)


if __name__ == "__main__":
    unittest.main()
