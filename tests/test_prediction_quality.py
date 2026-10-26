"""Synthetic pre-access tests for the test-set prediction-quality evaluator."""

import copy
import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fast import graph_contract
from fast import ablation_common as common
from fast import prediction_quality as test_quality


SPEC = common.read_json(test_quality.SPEC_PATH)


def _record(split="test", source="test_assembly"):
    rng = np.random.RandomState(4)
    action_x = rng.rand(2, 6, 16).astype(np.float32)
    action_x[:, :, :3] = graph_contract.DIRECTION_VECTORS
    r = np.ones((2, 6), dtype=np.float32)
    t = np.asarray([1.0, 0.0], dtype=np.float32)
    return {
        "schema_version": graph_contract.RECORD_SCHEMA_VERSION, "query_id": "q", "dataset_split": split, "source_split": source,
        "assembly_id": "a", "tier": "small", "state_id": "s", "pose_id": "p", "input_sha256": "0" * 64,
        "feature_shard_sha256": "0" * 64, "feature_extractor_sha256": "0" * 64, "part_ids": ["0", "1"],
        "node_x": rng.rand(2, 41).astype(np.float32), "edge_index": [[0, 1], [1, 0]], "candidate_index": [0, 1],
        "candidate_mask": [True, True], "base_mask": [False, False], "action_x": action_x,
        "targets": {"R": r, "R_valid": np.ones((2, 6), dtype=bool), "T": t, "T_valid": np.ones(2, dtype=bool),
                    "Y": r * t[:, None], "Y_valid": np.ones((2, 6), dtype=bool)},
    }


def _entries():
    entries, index = [], 0
    for tier, count in SPEC["population"]["tier_counts"].items():
        for _ in range(count):
            entries.append({"index": index, "assembly_id": "id{}".format(index), "dataset_split": "test", "source_split": "test_assembly",
                            "tier": tier, "candidate_count": 0, "action_count": 0})
            index += 1
    entries[0]["candidate_count"] = SPEC["population"]["feature_candidate_count"]
    entries[0]["action_count"] = SPEC["population"]["feature_action_count"]
    return entries


class TestQualityGuardTest(unittest.TestCase):
    def test_aggregate_only_rejects_identifiers_probabilities_and_vectors(self):
        self.assertTrue(test_quality.aggregate_only({"R": {"all": {"counts": {"TP": 3}, "rates": {"accuracy_percent": 50.0}}},
                                                     "sha256": "a" * 64, "commit": "5817ad03976a17ee1ada7c1680bbbd9185d75a7c",
                                                     "created_at": "2026-10-09T05:00:00+00:00", "capability": [12, 0]}))
        for bad in ({"assembly_id": "x"}, {"note": "assembly 01234"}, {"rows": ["state-abc"]}, {"p": "test_assembly/x"},
                    {"scores": [0.1]}, {"vector": [0.1] * 17}, {"nested": [{"pose_id": "pose-000"}]}):
            with self.assertRaises(ValueError):
                test_quality.aggregate_only(bad)

    def test_development_loader_refuses_and_test_evaluator_requires_test_rows(self):
        with self.assertRaisesRegex(ValueError, "rejects test"):
            graph_contract.validate_development_record(_record())
        validated = test_quality.validate_test_record(_record())
        self.assertEqual((validated["dataset_split"], validated["source_split"]), ("test", "test_assembly"))
        for split, source in (("validation", "training_assembly"), ("train", "training_assembly"), ("test", "training_assembly")):
            with self.assertRaisesRegex(ValueError, "accepts only test rows"):
                test_quality.validate_test_record(_record(split, source))

    def test_test_validator_keeps_every_frozen_structural_check(self):
        broken = _record()
        broken["action_x"] = broken["action_x"][:, :, :5]
        with self.assertRaisesRegex(ValueError, "action_x shape differs"):
            test_quality.validate_test_record(broken)

    def test_registry_rejects_missing_duplicated_foreign_and_tier_drift(self):
        self.assertTrue(test_quality.validate_registry(_entries(), SPEC))
        missing = _entries()[:-1]
        with self.assertRaisesRegex(ValueError, "missing|tier"):
            test_quality.validate_registry(missing, SPEC)
        duplicated = _entries()
        duplicated[1]["assembly_id"] = duplicated[0]["assembly_id"]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            test_quality.validate_registry(duplicated, SPEC)
        for split, source in (("validation", "training_assembly"), ("train", "training_assembly"), ("test", "other")):
            foreign = _entries()
            foreign[5].update(dataset_split=split, source_split=source)
            with self.assertRaisesRegex(ValueError, "foreign"):
                test_quality.validate_registry(foreign, SPEC)
        drift = _entries()
        drift[0]["tier"] = "large"
        with self.assertRaisesRegex(ValueError, "tier counts"):
            test_quality.validate_registry(drift, SPEC)

    def test_registry_must_equal_the_included_test_index(self):
        rows = {("test_assembly", entry["assembly_id"]): {"dataset_split": "test", "included": True} for entry in _entries()}
        rows[("test_assembly", "extra")] = {"dataset_split": "test", "included": True}
        with self.assertRaisesRegex(ValueError, "included test assemblies"):
            test_quality.validate_registry(_entries(), SPEC, rows)

    def test_publication_is_registered_atomic_write_once_and_confined(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unregistered"):
                test_quality.publish(directory, "per-assembly.json", {})
            test_quality.publish(directory, "input-audit.json", {"status": "passed"})
            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                test_quality.publish(directory, "input-audit.json", {"status": "passed"})
            with self.assertRaisesRegex(ValueError, "aggregate-only"):
                test_quality.publish(directory, "final-audit.json", {"assembly_id": "x"})
            with self.assertRaisesRegex(ValueError, "escapes"):
                test_quality.safe_output(directory, "../outside.json")
            with self.assertRaisesRegex(ValueError, "source closure"):
                test_quality.safe_output(directory, "closure/fast-data/x.json")

    def test_read_only_sources_refuse_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            (source / "a").mkdir(parents=True)
            target = source / "a" / "labels.json"
            target.write_text("{}", encoding="utf-8")
            test_quality.make_read_only(source)
            if os.geteuid() != 0:
                self.assertTrue(test_quality.write_refused(target))
            for path in (source, source / "a"):
                path.chmod(path.stat().st_mode | stat.S_IWUSR)

    def test_undefined_metrics_get_mechanical_reasons(self):
        rows = {"T": {"large": {"counts": {"TP": 0, "TN": 5, "FP": 0, "FN": 0}}}}
        reasons = test_quality.undefined_reasons(rows)["T"]["large"]["undefined_reasons"]
        self.assertEqual(set(reasons), {"precision_percent", "recall_percent", "average_precision", "auroc"})

    def test_d_decision_is_required_and_must_precede_unblinding(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "statement"):
                test_quality.record_d_decision(directory, "stop", " ")
            with self.assertRaises(ValueError):
                test_quality.record_d_decision(directory, "maybe", "x")
            test_quality.run_root(directory).mkdir(parents=True)
            (test_quality.run_root(directory) / "test-prediction-quality.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "precede"):
                test_quality.record_d_decision(directory, "continue", "continue")

    def test_timing_gate_requires_the_d_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            test_quality.run_root(directory).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "decision"):
                test_quality.timing_gate(directory)
            test_quality.record_d_decision(directory, "stop", "stop after seed 0")
            gate = test_quality.timing_gate(directory)
            self.assertEqual((gate["mode"], gate["d_jobs_running"]), ("d_decision_recorded", 0))

if __name__ == "__main__":
    unittest.main()
