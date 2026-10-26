import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from fast import graph_contract, graph_materialize, ranker_models, ranker_train, ranker_validate


def _record():
    action_x = np.zeros((2, 6, 16), dtype=np.float32)
    action_x[:, :, :3] = graph_contract.DIRECTION_VECTORS
    r = np.zeros((2, 6), dtype=np.float32)
    r[0, 0] = 1
    valid = np.ones((2, 6), dtype=bool)
    t = np.ones(2, dtype=np.float32)
    return {
        "schema_version": graph_contract.RECORD_SCHEMA_VERSION,
        "query_id": "training_assembly:00001:state-root:pose-000",
        "dataset_split": "train", "source_split": "training_assembly",
        "assembly_id": "00001", "tier": "small", "state_id": "state-root", "pose_id": "pose-000",
        "input_sha256": "a" * 64, "feature_shard_sha256": "b" * 64, "feature_extractor_sha256": "c" * 64,
        "part_ids": ["0", "1", "2"],
        "node_x": np.arange(123, dtype=np.float32).reshape(3, 41) / 100,
        "edge_index": np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        "candidate_index": np.asarray([0, 1]),
        "candidate_mask": np.asarray([True, True, False]),
        "base_mask": np.zeros(3, dtype=bool),
        "action_x": action_x,
        "targets": {"R": r, "R_valid": valid, "T": t, "T_valid": np.ones(2, dtype=bool),
                    "Y": r.copy(), "Y_valid": valid.copy()},
    }


def _normalization():
    return {
        "node": {"mean": [0.0] * 41, "std": [1.0] * 41},
        "action": {"mean": [0.0] * 16, "std": [1.0] * 16},
    }


class RankerRuntimeTest(unittest.TestCase):
    def test_repository_identity_uses_submitted_clean_commit_without_git(self):
        root = Path(ranker_train.__file__).resolve().parents[1]
        commit = ranker_train._git_head(root)
        with mock.patch.dict(os.environ, {
            "FAST_IMPLEMENTATION_COMMIT": commit,
            "FAST_IMPLEMENTATION_CLEAN": "1",
        }):
            self.assertEqual(
                ranker_train._repository_identity(require_clean=True),
                {"commit": commit, "dirty": False},
            )
        with mock.patch.dict(os.environ, {
            "FAST_IMPLEMENTATION_COMMIT": "0" * 40,
            "FAST_IMPLEMENTATION_CLEAN": "1",
        }):
            with self.assertRaisesRegex(ValueError, "submitted implementation commit differs"):
                ranker_train._repository_identity(require_clean=True)

    def test_all_models_have_frozen_counts_shapes_and_permutation_behavior(self):
        torch.manual_seed(0)
        batch = ranker_models.torch_batch([_record()], _normalization(), torch.device("cpu"))
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(3)
        permuted = dict(batch)
        permuted["node_x"] = batch["node_x"][permutation]
        permuted["edge_index"] = inverse[batch["edge_index"]]
        permuted["candidate_index"] = inverse[batch["candidate_index"]]
        for architecture in ranker_models.ARCHITECTURES:
            model = ranker_models.build_model(architecture).eval()
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), ranker_models.EXPECTED_PARAMETERS[architecture])
            with torch.no_grad():
                output = model(batch)
                changed = model(permuted)
            self.assertEqual(next(iter(output.values())).shape, (2, 6))
            for name in output:
                torch.testing.assert_close(output[name], changed[name], rtol=1e-5, atol=1e-6)

    def test_torch_factorized_loss_matches_numpy_reference(self):
        record = _record()
        numpy_batch = graph_contract.batch_development_records([record])
        expected = graph_contract.factorized_reference_loss(
            numpy_batch, np.zeros((2, 6)), np.zeros(2), {"total": 1, "rank": 1, "R": 1, "T": 1, "Y": 1}
        )
        batch = ranker_models.torch_batch([record], _normalization(), torch.device("cpu"))
        actual, details = ranker_models.model_loss(
            batch, {"R": torch.zeros((2, 6)), "T": torch.zeros(2)},
            {"total": 1, "rank": 1, "R": 1, "T": 1, "Y": 1},
        )
        self.assertAlmostEqual(float(actual), expected["loss"], places=6)
        self.assertEqual(details["denominators"]["Y"], 12)

    def test_checkpoint_round_trip_and_hash_guard(self):
        model = ranker_models.build_model("mlp")
        optimizer = torch.optim.AdamW(model.parameters())
        provenance = {"configuration": "synthetic"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            ranker_train.configure_determinism(0)
            original = ranker_train._state_digest(model)
            ranker_train.save_checkpoint(path, model, optimizer, provenance, 1, 1, {"epoch": 1}, {"epoch": 1})
            with torch.no_grad():
                next(model.parameters()).zero_()
            ranker_train.load_checkpoint(path, model, optimizer, provenance)
            self.assertEqual(ranker_train._state_digest(model), original)
            with path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                ranker_train.load_checkpoint(path, model, optimizer, provenance)

    def test_normalization_calibration_and_bootstrap_are_deterministic(self):
        state = {"count": 2, "sum": np.asarray([2.0, 4.0]), "sum_sq": np.asarray([4.0, 8.0])}
        normalization = graph_materialize._normalization_finish(state)
        self.assertEqual(normalization["zero_variance_columns"], [1])
        self.assertEqual(normalization["std"][1], 1.0)

        entries, folds = [], {}
        validation_rows = []
        for index in range(6):
            assembly = "{:05d}".format(index)
            tier = ("small", "medium", "large")[index % 3]
            folds["training_assembly:" + assembly] = index % 5
            validation_rows.append({"assembly_id": assembly, "tier": tier})
            entries.append({"assembly_id": assembly, "tier": tier, "direction": 0,
                            "target": float(index % 2), "valid": True, "logit": float(index - 2)})
        temperatures, assignments, final = ranker_validate._cross_fit(entries, folds)
        self.assertTrue(all(value > 0 for value in temperatures.values()))
        self.assertTrue(all(value > 0 for value in assignments.values()))
        self.assertGreater(final, 0)
        order1, plan1 = ranker_validate._bootstrap_plan(validation_rows, 100, 20260916)
        order2, plan2 = ranker_validate._bootstrap_plan(validation_rows, 100, 20260916)
        self.assertEqual(order1, order2)
        np.testing.assert_array_equal(plan1, plan2)


if __name__ == "__main__":
    unittest.main()
