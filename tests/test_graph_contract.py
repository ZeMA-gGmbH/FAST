import math
import unittest
from pathlib import Path

import numpy as np

from fast import graph_contract


ROOT = Path(__file__).resolve().parents[1]


def _record(assembly_id="00001", partial=False):
    candidate_count = 2
    action_x = np.zeros((candidate_count, 6, 16), dtype=np.float32)
    action_x[:, :, :3] = graph_contract.DIRECTION_VECTORS
    r = np.zeros((candidate_count, 6), dtype=np.float32)
    r_valid = np.ones((candidate_count, 6), dtype=bool)
    t = np.ones(candidate_count, dtype=np.float32)
    t_valid = np.ones(candidate_count, dtype=bool)
    if partial:
        t[:] = 0.0
        t_valid[:] = False
    else:
        r[0, 0] = 1.0
    y_valid = (r_valid & (r == 0.0)) | (t_valid[:, None] & (t[:, None] == 0.0)) | (
        r_valid & (r == 1.0) & t_valid[:, None] & (t[:, None] == 1.0)
    )
    y = (r_valid & (r == 1.0) & t_valid[:, None] & (t[:, None] == 1.0)).astype(np.float32)
    return {
        "schema_version": graph_contract.RECORD_SCHEMA_VERSION,
        "query_id": "training_assembly:{}:state-root:pose-000".format(assembly_id),
        "dataset_split": "train",
        "source_split": "training_assembly",
        "assembly_id": assembly_id,
        "tier": "small",
        "state_id": "state-root",
        "pose_id": "pose-000",
        "input_sha256": "a" * 64,
        "feature_shard_sha256": "b" * 64,
        "feature_extractor_sha256": "c" * 64,
        "part_ids": ["0", "1", "2"],
        "node_x": np.arange(3 * 41, dtype=np.float32).reshape(3, 41) / 100.0,
        "edge_index": np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        "candidate_index": np.asarray([0, 1], dtype=np.int64),
        "candidate_mask": np.asarray([True, True, False]),
        "base_mask": np.asarray([False, False, False]),
        "action_x": action_x,
        "targets": {
            "R": r,
            "R_valid": r_valid,
            "T": t,
            "T_valid": t_valid,
            "Y": y,
            "Y_valid": y_valid,
        },
    }


class _ExplodesOnArray:
    def __array__(self):
        raise AssertionError("test payload was accessed")


class RankerContractTest(unittest.TestCase):
    def test_frozen_spec_and_graph_record(self):
        spec = graph_contract.load_spec(ROOT / "configs" / "ranker_models.json")
        self.assertEqual(spec["training"]["seeds"], [0, 1, 2])
        record = graph_contract.validate_development_record(_record())
        self.assertEqual(record["node_x"].shape, (3, 41))
        self.assertEqual(record["action_x"].shape, (2, 6, 16))
        self.assertEqual(record["candidate_index"].tolist(), [0, 1])
        self.assertFalse(record["candidate_mask"][2])

    def test_batch_offsets_and_ownership(self):
        batch = graph_contract.batch_development_records([_record(), _record("00002", partial=True)])
        self.assertEqual(batch["node_x"].shape, (6, 41))
        self.assertEqual(batch["candidate_index"].tolist(), [0, 1, 3, 4])
        self.assertEqual(batch["edge_index"][:, 4:].tolist(), [[3, 4, 4, 5], [4, 3, 5, 4]])
        self.assertEqual(batch["context_assembly"].tolist(), [0, 1])
        self.assertEqual(batch["candidate_context"].tolist(), [0, 0, 1, 1])

    def test_empty_integer_array_survives_json_representation(self):
        record = _record()
        record["edge_index"] = [[], []]
        validated = graph_contract.validate_development_record(record)
        self.assertEqual(validated["edge_index"].shape, (2, 0))
        self.assertEqual(validated["edge_index"].dtype, np.dtype(np.int64))

    def test_test_split_is_rejected_before_payload_access(self):
        record = _record()
        record["dataset_split"] = "test"
        record["node_x"] = _ExplodesOnArray()
        with self.assertRaisesRegex(ValueError, "rejects test"):
            graph_contract.validate_development_record(record)

    def test_direction_mask_and_conjunction_fail_closed(self):
        record = _record()
        record["action_x"][0, 0, 0] = 9.0
        with self.assertRaisesRegex(ValueError, "direction"):
            graph_contract.validate_development_record(record)
        record = _record("00002", partial=True)
        record["targets"]["Y_valid"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "Y validity"):
            graph_contract.validate_development_record(record)

    def test_stable_masked_loss_and_equal_assembly_weighting(self):
        batch = graph_contract.batch_development_records([_record(), _record("00002", partial=True)])
        r_logits = np.zeros_like(batch["R"], dtype=np.float64)
        t_logits = np.zeros_like(batch["T"], dtype=np.float64)
        counts = {"total": 2, "rank": 1, "R": 2, "T": 1, "Y": 1}
        result = graph_contract.factorized_reference_loss(batch, r_logits, t_logits, counts)

        expected_rank = math.log(12.0)
        expected_r = math.log(2.0)
        expected_t = math.log(2.0)
        expected_y = (math.log(4.0) + 11.0 * -math.log(0.75)) / 12.0
        expected = expected_rank + (expected_r + expected_t + expected_y) / 3.0
        self.assertAlmostEqual(result["components"]["rank"], expected_rank)
        self.assertAlmostEqual(result["components"]["Y"], expected_y)
        self.assertAlmostEqual(result["loss"], expected)
        self.assertEqual(result["strict_y_valid_count"], 12)
        self.assertEqual(result["denominators"]["rank"].tolist(), [1, 0])

        changed_t = t_logits.copy()
        changed_t[2:] = 1000.0
        self.assertAlmostEqual(
            graph_contract.factorized_reference_loss(batch, r_logits, changed_t, counts)["loss"],
            result["loss"],
        )
        extreme_r = np.where(batch["R"] == 1.0, 1000.0, -1000.0)
        extreme_t = np.full_like(t_logits, 1000.0)
        self.assertTrue(math.isfinite(graph_contract.factorized_reference_loss(batch, extreme_r, extreme_t, counts)["loss"]))


if __name__ == "__main__":
    unittest.main()
