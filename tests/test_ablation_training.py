import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fast import graph_contract, ranker_models, ranker_train
from fast import ablation_common as common
from fast import ablation_masks as masks
from fast import ablation_replay as replay
from fast import ablation_runner as runner
from fast import ablation_smoke as smoke
from fast import ablation_telemetry as telemetry
from fast.cad_features import ACTION_FEATURES


SPEC = common.read_json(common.REPOSITORY_ROOT / "configs" / "component_ablation.json")


def _record(seed, split="train"):
    rng = np.random.RandomState(seed)
    action_x = rng.rand(3, 6, 16).astype(np.float32)
    action_x[:, :, :3] = graph_contract.DIRECTION_VECTORS
    r = (rng.rand(3, 6) > 0.5).astype(np.float32)
    t = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    valid = np.ones((3, 6), dtype=bool)
    return {
        "schema_version": graph_contract.RECORD_SCHEMA_VERSION, "query_id": "q{}".format(seed), "dataset_split": split,
        "source_split": "training_assembly", "assembly_id": "a{}".format(seed), "tier": "small", "state_id": "s", "pose_id": "pose-000",
        "input_sha256": "0" * 64, "feature_shard_sha256": "0" * 64, "feature_extractor_sha256": "0" * 64,
        "part_ids": ["0", "1", "2"], "node_x": rng.rand(3, 41).astype(np.float32), "edge_index": [[0, 1, 1, 2], [1, 0, 2, 1]],
        "candidate_index": [0, 1, 2], "candidate_mask": [True, True, True], "base_mask": [False, False, False], "action_x": action_x,
        "targets": {"R": r, "R_valid": valid, "T": t, "T_valid": np.ones(3, dtype=bool), "Y": r * t[:, None], "Y_valid": valid},
    }


def _normalization():
    return {"node": {"mean": [0.1] * 41, "std": [2.0] * 41}, "action": {"mean": [0.05 * index for index in range(16)], "std": [1.0 + index for index in range(16)]}}


COUNTS = {"total": 2, "rank": 2, "R": 2, "T": 2, "Y": 2}


class AblationMaskTest(unittest.TestCase):
    def test_mask_membership_widths_and_order(self):
        expected = {"D0": list(range(16)), "D1": list(range(5)), "D2": list(range(5, 16))}
        for variant, indices in expected.items():
            mask, names = masks.action_mask(SPEC, variant)
            self.assertEqual(mask.tolist(), indices)
            self.assertEqual(names, [ACTION_FEATURES[index] for index in indices])
        self.assertTrue(all(name.startswith("direction__") for name in masks.action_mask(SPEC, "D1")[1]))
        self.assertTrue(all(name.startswith("directional_clearance__") for name in masks.action_mask(SPEC, "D2")[1]))

    def test_mask_fails_closed_when_schema_names_or_order_differ(self):
        changed = copy.deepcopy(SPEC)
        groups = changed["experiments"]["D"]["action_feature_groups"]["directional_clearance"]
        groups[0], groups[1] = groups[1], groups[0]
        with self.assertRaisesRegex(ValueError, "schema differs"):
            masks.action_mask(changed, "D2")
        changed = copy.deepcopy(SPEC)
        changed["experiments"]["D"]["variants"]["D1"]["input_width"] = 6
        with self.assertRaisesRegex(ValueError, "width differs"):
            masks.action_mask(changed, "D1")

    def test_sliced_normalization_equals_slicing_the_full_normalized_batch(self):
        records = [graph_contract.validate_development_record(_record(seed)) for seed in (1, 2)]
        full = ranker_models.torch_batch(records, _normalization(), torch.device("cpu"))
        for variant in masks.VARIANTS:
            indices, _ = masks.action_mask(SPEC, variant)
            masked = masks.masked_torch_batch(records, _normalization(), indices, torch.device("cpu"))
            torch.testing.assert_close(masked["action_x"], full["action_x"][..., indices], rtol=0, atol=1e-6)
            self.assertTrue(torch.equal(masked["node_x"], full["node_x"]))

    def test_forward_backward_and_parameter_accounting_for_every_variant(self):
        records = [graph_contract.validate_development_record(_record(seed)) for seed in (1, 2)]
        for variant, expected in (("D0", 90142), ("D1", 89086), ("D2", 89662)):
            model = masks.build_model(SPEC, variant)
            self.assertEqual(masks.parameter_count(model), expected)
            indices, _ = masks.action_mask(SPEC, variant)
            batch = masks.masked_torch_batch(records, _normalization(), indices, torch.device("cpu"))
            loss, _ = ranker_models.model_loss(batch, model(batch), COUNTS)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(all(parameter.grad is not None for parameter in model.action_head.parameters()))

    def test_d0_initialization_equals_the_frozen_graph_transformer(self):
        self.assertTrue(smoke.initialization_parity(SPEC))

    def test_deterministic_repeat_and_resume(self):
        records = [graph_contract.validate_development_record(_record(seed)) for seed in (1, 2)]
        indices, _ = masks.action_mask(SPEC, "D1")
        batch = masks.masked_torch_batch(records, _normalization(), indices, torch.device("cpu"))
        device = torch.device("cpu")
        first, digest, initial, _ = smoke._pilot_trace(SPEC, "D1", 0, batch, COUNTS, 2, device)
        second, second_digest, _, _ = smoke._pilot_trace(SPEC, "D1", 0, batch, COUNTS, 2, device, initial)
        self.assertEqual((first, digest), (second, second_digest))
        one, _, initial, interrupted = smoke._pilot_trace(SPEC, "D1", 0, batch, COUNTS, 1, device)
        resumed, resumed_digest, _, _ = smoke._pilot_trace(SPEC, "D1", 0, batch, COUNTS, 1, device, interrupted)
        straight, straight_digest, _, _ = smoke._pilot_trace(SPEC, "D1", 0, batch, COUNTS, 2, device, initial)
        self.assertEqual((one + resumed, resumed_digest), (straight, straight_digest))


class AblationRegistryTest(unittest.TestCase):
    def test_registry_has_nine_unique_identities_bound_to_the_spec(self):
        registry = runner.build_registry(SPEC, "a" * 64)
        self.assertEqual([row["run_name"] for row in registry["matrix"][:3]], ["D0-seed-0", "D1-seed-0", "D2-seed-0"])
        self.assertEqual(len({row["identity_sha256"] for row in registry["matrix"]}), 9)
        other = runner.build_registry(SPEC, "b" * 64)
        self.assertNotEqual(registry["matrix"][0]["identity_sha256"], other["matrix"][0]["identity_sha256"])
        duplicate = copy.deepcopy(SPEC)
        duplicate["experiments"]["D"]["matrix"][1] = dict(duplicate["experiments"]["D"]["matrix"][0], process=1)
        with self.assertRaisesRegex(ValueError, "nine unique"):
            runner.build_registry(duplicate, "a" * 64)

    def test_completed_identity_is_unique_and_order_requires_smoke_and_d0(self):
        registry = runner.build_registry(SPEC, "a" * 64)
        row = registry["matrix"][1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(runner.completed(root, row))
            for attempt in (1, 2):
                path = root / "training" / row["variant"] / "seed-0" / "attempt-{}".format(attempt)
                path.mkdir(parents=True)
                common.write_json_once(path / "result.json", {"status": "passed"})
            with self.assertRaisesRegex(ValueError, "duplicate completed"):
                runner.completed(root, row)
            with self.assertRaisesRegex(ValueError, "smoke has not passed"):
                runner.require_order(root, registry, registry["matrix"][0], False)
            (root / "smoke").mkdir()
            common.write_json_once(root / "smoke" / "d0-seed-0-smoke.json", {"status": "passed"})
            with self.assertRaisesRegex(ValueError, "D0-seed-0 must complete"):
                runner.require_order(root, registry, registry["matrix"][2], False)

    def test_seed_one_requires_a_recorded_continuation(self):
        registry = runner.build_registry(SPEC, "a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "smoke").mkdir()
            common.write_json_once(root / "smoke" / "d0-seed-0-smoke.json", {"status": "passed"})
            path = root / "training" / "D0" / "seed-0" / "attempt-1"
            path.mkdir(parents=True)
            common.write_json_once(path / "result.json", {"status": "passed"})
            with self.assertRaisesRegex(ValueError, "recorded continuation"):
                runner.require_order(root, registry, registry["matrix"][3], False)


class AblationTelemetryAndGuardTest(unittest.TestCase):
    def _attempt(self, directory):
        root = Path(directory)
        config = {"run_name": "x", "variant": "D0", "seed": 0, "epochs": 1, "provenance": {
            "run_id": common.RUN_ID, "implementation": {"commit": "c" * 40}, "environment": {"lock_sha256": "l" * 64, "torch": "t", "device": "d"},
            "dataset_manifest_sha256": "1", "graph_audit_sha256": "2", "normalization_sha256": "3", "sliced_normalization_sha256": "4",
            "spec_sha256": "5", "action_feature_names": ["direction__x"], "input_width": 1, "parameter_count": 1}}
        common.write_json_once(root / "telemetry-config.json", config)
        row = {"epoch": 1, "mean_batch_loss": 1.0, "loss_components": {"rank": 0.5, "R": None}, "R_average_precision": 0.5,
               "validation_selection": {"offline_context_success_at_1": 0.5, "first_feasible_mrr": 0.6, "Y_average_precision": 0.7},
               "epoch_seconds": 1.0, "validation_inference_seconds": 0.1, "peak_host_kib": 1, "peak_cuda_allocated_bytes": 1, "best_epoch_so_far": 1}
        (root / "epochs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        common.write_json_once(root / "result.json", {"status": "passed", "best": {"epoch": 1}, "best_checkpoint_sha256": "f" * 64})
        return root

    def test_disabled_mirror_records_without_service(self):
        with tempfile.TemporaryDirectory() as directory:
            record = telemetry.mirror(Path(directory), "x", "disabled")
            self.assertEqual((record["mode"], record["status"], record["authoritative"]), ("disabled", "disabled", False))

    @unittest.skipUnless(importlib.util.find_spec("wandb"), "W&B is only installed in the training runtime")
    def test_offline_mirror_logs_scalars_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["WANDB_MODE"] = "offline"
            try:
                record = telemetry.mirror(self._attempt(directory), "x", "offline", poll_seconds=0.1, max_wait_seconds=60)
            finally:
                os.environ.pop("WANDB_MODE", None)
            self.assertEqual((record["mode"], record["status"], record["logged_epochs"], record["url"]), ("offline", "completed", 1, None))

    def test_zero_test_row_guards(self):
        with self.assertRaisesRegex(ValueError, "rejects test"):
            graph_contract.validate_development_record(_record(1, split="test"))
        with self.assertRaisesRegex(ValueError, "test row loaded"):
            replay.ShardFeatureBuilder([{"dataset_split": "test", "state_id": "s", "pose_id": "p"}])
        with self.assertRaisesRegex(ValueError, "test-bearing"):
            common.require_development_path("closure/asap-data/test_assembly/00000/0.obj")

    def test_input_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shards" / "a.jsonl"
            shard.parent.mkdir()
            shard.write_text("{}\n", encoding="utf-8")
            prepared = {"output_root": root}
            with self.assertRaisesRegex(ValueError, "hash differs"):
                ranker_train._load_assembly(prepared, {"dataset_split": "train", "path": "shards/a.jsonl", "sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
