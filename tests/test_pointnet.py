import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import trimesh

from fast import graph_contract, ranker_models, pointnet_models, action_planner, planner_integration, pointnet_points, pointnet_train, pointnet_validate, planning_study
from plan_sequence import production_sim_labels as production


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "pointnet.json"


def _record(query="q1"):
    action_x = np.zeros((2, 6, 16), dtype=np.float32)
    action_x[:, :, :3] = graph_contract.DIRECTION_VECTORS
    r = np.zeros((2, 6), dtype=np.float32)
    r[0, 0] = 1
    valid = np.ones((2, 6), dtype=bool)
    return {
        "schema_version": graph_contract.RECORD_SCHEMA_VERSION, "query_id": query,
        "dataset_split": "train", "source_split": "training_assembly", "assembly_id": "00001", "tier": "small",
        "state_id": "state-root", "pose_id": "pose-000", "input_sha256": "a" * 64,
        "feature_shard_sha256": "b" * 64, "feature_extractor_sha256": "c" * 64,
        "part_ids": ["0", "1", "2"], "node_x": np.arange(123, dtype=np.float32).reshape(3, 41) / 100,
        "edge_index": np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        "candidate_index": np.asarray([0, 1]), "candidate_mask": np.asarray([True, True, False]),
        "base_mask": np.zeros(3, dtype=bool), "action_x": action_x,
        "targets": {"R": r, "R_valid": valid, "T": np.ones(2, dtype=np.float32), "T_valid": np.ones(2, dtype=bool),
                    "Y": r.copy(), "Y_valid": valid.copy()},
    }


def _normalization():
    return {"node": {"mean": [0.0] * 41, "std": [1.0] * 41}, "action": {"mean": [0.0] * 16, "std": [1.0] * 16}}


class _FakeOracle:
    def __init__(self):
        self.t_calls = {}

    def evaluate_direction(self, parts, part, direction, pose):
        del pose
        if tuple(parts) == ("A", "B", "C"):
            values = {("A", "+Z"): True, ("B", "+Z"): None, ("B", "-Z"): True,
                      ("B", "+X"): True, ("C", "+Z"): True}
            value = values.get((part, direction), False)
        else:
            value = part == "A" and direction == "+Z"
        return {"label": value, "status": "success" if value is True else ("timeout" if value is None else "blocked"),
                "reason": "fake", "rollout_count": 1, "forward_steps": 1, "runtime_s": 0.1}

    def evaluate_stability(self, parts_after, pose):
        del pose
        key = tuple(parts_after)
        self.t_calls[key] = self.t_calls.get(key, 0) + 1
        value = False if key == ("B", "C") else (None if key == ("A", "C") else True)
        return {"label": value, "status": "success" if value is True else ("timeout" if value is None else "unstable"),
                "reason": "fake", "rollout_count": 2, "forward_steps": 2, "runtime_s": 0.2}


class PointNetTest(unittest.TestCase):
    def test_prepare_configures_determinism_before_environment_capture(self):
        source = inspect.getsource(pointnet_train._prepare)
        self.assertLess(source.index("configure_determinism"), source.index("ranker_train._prepare"))

    def test_gpu_provenance_allows_cpu_auditor_but_rejects_wrong_gpu(self):
        prepared = {
            "spec_sha256": "spec", "ranker_spec_sha256": "graph_contract", "audit_sha256": "graph",
            "point_audit_sha256": "points", "audit": {"normalization": {}, "validation_folds_sha256": "folds"},
            "implementation": {"commit": "b" * 40, "dirty": False},
            "environment": {"cuda_available": False, "device": "cpu", "deterministic_algorithms": True, "tf32_cudnn": False},
            "spec": {"resources": {"gpu": {"model": "NVIDIA A100-PCIE-40GB"}}},
        }
        producer = {"commit": "a" * 40, "dirty": False}
        provenance = pointnet_train._provenance(prepared, "pointnet_graph_transformer", 0, producer)
        provenance["environment"] = {**provenance["environment"], "cuda_available": True, "device": "NVIDIA A100-PCIE-40GB"}
        pointnet_train._validate_gpu_provenance(prepared, provenance, "pointnet_graph_transformer", 0, producer)
        wrong = {**provenance, "environment": {**provenance["environment"], "device": "different GPU"}}
        with self.assertRaisesRegex(ValueError, "GPU artifact environment differs"):
            pointnet_train._validate_gpu_provenance(prepared, wrong, "pointnet_graph_transformer", 0, producer)

    def test_frozen_spec_and_exact_model_counts(self):
        spec = pointnet_points.load_spec(SPEC)
        self.assertEqual(spec["points"]["root_seed"], 20260924)
        self.assertEqual(sum(p.numel() for p in pointnet_models.build_pointnet_model().parameters()), 93902)
        self.assertEqual(sum(p.numel() for p in pointnet_train._model("cebulla_gcn").parameters()), 72577)
        self.assertEqual(sum(p.numel() for p in ranker_models.build_model("graph_transformer").parameters()), 90142)

    def test_sampling_is_byte_identical_normalized_and_rng_isolated(self):
        spec = pointnet_points.load_spec(SPEC)["points"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = trimesh.creation.box(extents=(1, 2, 3)); first.export(root / "0.obj")
            second = trimesh.creation.box(extents=(2, 1, 1)); second.apply_translation((4, 0, 0)); second.export(root / "1.obj")
            np.random.seed(17)
            expected = np.random.random(4)
            np.random.seed(17)
            one = pointnet_points.sample_assembly(root, "a" * 64, spec)
            actual = np.random.random(4)
            two = pointnet_points.sample_assembly(root, "a" * 64, spec)
            np.testing.assert_array_equal(expected, actual)
            self.assertEqual(one[0].tobytes(), two[0].tobytes())
            self.assertEqual(one[2], two[2])
            self.assertEqual(one[0].dtype.str, "<f4")
            self.assertLessEqual(float(np.abs(one[0]).max()), 1.0 + 1e-6)

    def test_point_gather_is_unique_and_point_order_invariant(self):
        tensor = np.arange(9000, dtype=np.float32).reshape(3, 1000, 3) / 9000
        records = [_record("q1"), _record("q2")]
        batch = pointnet_models.torch_batch(records, _normalization(), {("training_assembly", "00001"): (["0", "1", "2"], tensor)}, torch.device("cpu"))
        self.assertEqual(batch["point_x"].shape, (3, 1000, 3))
        self.assertEqual(batch["node_point_index"].tolist(), [0, 1, 2, 0, 1, 2])
        model = pointnet_models.build_pointnet_model().eval()
        permuted = dict(batch)
        permuted["point_x"] = batch["point_x"][:, torch.arange(999, -1, -1)]
        with torch.no_grad():
            one, two = model(batch), model(permuted)
        torch.testing.assert_close(one["R"], two["R"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(one["T"], two["T"], rtol=1e-5, atol=1e-6)

    def test_pointnet_pool_preserves_asap_values_and_tie_gradients_without_adaptive_cuda_kernel(self):
        self.assertNotIn("adaptive_max_pool", inspect.getsource(pointnet_models.PointNet.forward))
        for values in ([1.0, 3.0, 2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 1.0]):
            frozen = torch.tensor(values, requires_grad=True).reshape(1, 1, -1)
            frozen.retain_grad()
            torch.nn.functional.adaptive_max_pool1d(frozen, 1).sum().backward()
            replacement = torch.tensor(values, requires_grad=True).reshape(1, 1, -1)
            replacement.retain_grad()
            replacement.max(dim=-1).values.sum().backward()
            torch.testing.assert_close(frozen.detach().max(dim=-1).values, replacement.detach().max(dim=-1).values)
            torch.testing.assert_close(frozen.grad, replacement.grad)

    def test_point_cache_rejects_tampering_and_test_metadata(self):
        spec = pointnet_points.load_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source" / "training_assembly" / "00001"; source.mkdir(parents=True)
            (source / "0.obj").write_text("v 0 0 0\n", encoding="utf-8")
            tensor_path, sidecar_path = root / "0.npy", root / "0.json"
            tensor = np.zeros((1, 1000, 3), dtype="<f4"); np.save(tensor_path, tensor, allow_pickle=False)
            job = {"dataset_split": "train", "source_split": "training_assembly", "assembly_id": "00001", "tier": "small", "input_sha256": "a" * 64}
            contract = spec["points"]
            sidecar = {
                "schema_version": pointnet_points.SIDECAR_SCHEMA_VERSION, "complete": True, "spec_sha256": "b" * 64,
                "identity": job, "cache_key": pointnet_points._canonical_hash({"contract": contract, "input_sha256": job["input_sha256"]}),
                "contract_sha256": pointnet_points._canonical_hash(contract), "part_ids": ["0"],
                "part_seeds": {"0": pointnet_points.part_seed(contract["contract"], contract["root_seed"], job["input_sha256"], "0")},
                "normalization": {"formula": contract["normalization"]}, "source_obj_sha256": {"0": production._sha256_file(source / "0.obj")},
                "sampler_source_sha256": production._sha256_file(Path(pointnet_points.__file__)), "numpy_version": np.__version__,
                "trimesh_version": trimesh.__version__, "shape": [1, 1000, 3], "dtype": "<f4", "tensor": "0.npy",
                "tensor_sha256": production._sha256_file(tensor_path), "test_rows_loaded": 0,
            }
            sidecar_path.write_text(production._canonical_json(sidecar), encoding="utf-8")
            prepared = {"spec": spec, "spec_sha256": "b" * 64, "source_root": root / "source"}
            pointnet_points._validate_shard(tensor_path, sidecar_path, prepared, job)
            with tensor_path.open("ab") as stream: stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "tensor hash differs"):
                pointnet_points._validate_shard(tensor_path, sidecar_path, prepared, job)
        with self.assertRaisesRegex(ValueError, "rejects non-development"):
            pointnet_points._jobs({"graph_audit": {"inventory": [{"dataset_split": "test", "source_split": "test_assembly"}]}})

    def test_deterministic_pointnet_resume_and_advancement_rule(self):
        tensor = np.arange(9000, dtype=np.float32).reshape(3, 1000, 3) / 9000
        batch = pointnet_models.torch_batch([_record()], _normalization(), {("training_assembly", "00001"): (["0", "1", "2"], tensor)}, torch.device("cpu"))
        counts = {"total": 1, "rank": 1, "R": 1, "T": 1, "Y": 1}
        one, _, initial, interrupted = pointnet_train._pilot_trace(batch, counts, 1, torch.device("cpu"))
        resumed, resumed_digest, _, _ = pointnet_train._pilot_trace(batch, counts, 1, torch.device("cpu"), interrupted)
        full, full_digest, _, _ = pointnet_train._pilot_trace(batch, counts, 2, torch.device("cpu"), initial)
        self.assertEqual(one + resumed, full)
        self.assertEqual(resumed_digest, full_digest)
        self.assertTrue(pointnet_validate.advancement_decision(True, 1e-9)["pointnet_advanced"])
        self.assertFalse(pointnet_validate.advancement_decision(True, 0.0)["pointnet_advanced"])
        self.assertFalse(pointnet_validate.advancement_decision(False, 1.0)["pointnet_advanced"])

    def test_cebulla_targets_follow_any_true_all_false_null_rule(self):
        batch = ranker_models.torch_batch([_record()], _normalization(), torch.device("cpu"))
        batch["R_valid"][1, 0] = False
        targets, valid = pointnet_models.cebulla_targets(batch)
        self.assertEqual(targets[0].item(), 1.0); self.assertTrue(valid[0].item())
        self.assertFalse(valid[1].item()); self.assertFalse(valid[2].item())

    def test_action_planner_validates_every_transition_and_never_retries_censored_t(self):
        oracle = _FakeOracle()
        poses = lambda parts: [("pose-000", None)]
        calls = []
        def ranker(parts, pose_id, pose, context):
            calls.append(context)
            return action_planner.ordered_actions(parts, parts)
        actions, summary = action_planner.run_action_dfs(("A", "B", "C"), poses, ranker, oracle,
                                                          max_action_validations=16, method="fake", seed=0)
        self.assertEqual(oracle.t_calls[("A", "C")], 1)
        self.assertTrue(any(row["part"] == "B" and row["direction"] == "+X" and row["T_cache_hit"] for row in actions))
        self.assertEqual(summary["accepted_transition_count"], 2)
        self.assertTrue(all(row["R"]["label"] is True and row["T"]["label"] is True for row in actions if row["accepted_transition"]))
        self.assertFalse(summary["budgets"]["4"]["success"]); self.assertTrue(summary["budgets"]["8"]["success"])
        self.assertEqual(summary["action_validation_count"], len(actions))
        self.assertEqual(len(calls), summary["ranked_context_count"])
        self.assertGreaterEqual(summary["model_ranking_seconds"], 0.0)
        self.assertGreaterEqual(summary["overhead_seconds"], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            header = {"record_type": "header", "schema_version": action_planner.TRACE_SCHEMA_VERSION}
            digest = action_planner.write_trace(path, header, actions, summary)
            self.assertEqual(digest, production._sha256_file(path))
            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                action_planner.write_trace(path, header, actions, summary)

    def test_online_features_and_train_only_direction_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, offset in enumerate((0.0, 1.0, 2.0)):
                mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
                mesh.apply_translation((offset, 0.0, 0.4))
                mesh.export(root / "{}.obj".format(index))
            (root / "contact_graph.json").write_text(json.dumps({
                "directed": False, "multigraph": False, "graph": {},
                "nodes": [{"id": str(index)} for index in range(3)],
                "links": [{"source": "0", "target": "1"}, {"source": "1", "target": "2"}],
            }), encoding="utf-8")
            builder = planner_integration.OnlineFeatureBuilder(root, {"support_tolerance_fraction": 0.001})
            record = builder.record(("0", "1", "2"), "pose-000", None)
            self.assertEqual(record["node_x"].shape, (3, 41))
            self.assertEqual(record["action_x"].shape, (3, 6, 16))
            np.testing.assert_array_equal(record["action_x"][:, :, :3], np.broadcast_to(graph_contract.DIRECTION_VECTORS, (3, 6, 3)))
            batch = planner_integration._torch_online_batch(record, _normalization())
            self.assertEqual(tuple(batch["node_ptr"].tolist()), (0, 3))
        record = graph_contract.validate_development_record(_record())
        positive, valid = planner_integration._direction_counts(record)
        self.assertEqual(positive.tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(valid.tolist(), [2, 2, 2, 2, 2, 2])

    def test_planner_wall_timeout_precedes_any_simulator_access(self):
        class Oracle:
            def evaluate_direction(self, *args):
                raise AssertionError("oracle should not be called")
        actions, summary = action_planner.run_action_dfs(
            ("A", "B"), lambda parts: [("pose-000", None)],
            lambda *args: action_planner.ordered_actions(("A", "B"), ("A", "B")), Oracle(),
            max_action_validations=8, method="timeout", seed=0, wall_timeout_seconds=1e-12,
        )
        self.assertEqual(actions, [])
        self.assertEqual(summary["stop_reason"], "wall_timeout")

    def test_study_validates_registry_before_opening_test_metadata_or_payload(self):
        setup_source = inspect.getsource(planning_study.setup)
        self.assertLess(setup_source.index("_locked("), setup_source.index("assemblies_path.open"))
        evaluate_source = inspect.getsource(planning_study.evaluate)
        self.assertLess(evaluate_source.index("_locked("), evaluate_source.index("_job("))
        self.assertIn('import_module("redmax_py")', inspect.getsource(planning_study._locked))

    def test_study_paired_inference_aggregates_seeds_and_stratifies_assemblies(self):
        rows = []
        for index in range(237):
            tier = ("small", "medium", "large")[min(index // 79, 2)]
            for method in planner_integration.METHODS:
                for seed in (0, 1, 2):
                    rows.append({
                        "method": method, "seed": seed, "assembly_id": "{:05d}".format(index), "tier": tier,
                        "success": {str(budget): method == "fast" for budget in (1, 2, 4, 8)},
                    })
        with mock.patch.object(planning_study, "BOOTSTRAP_REPLICATES", 100):
            result = planning_study._paired_statistics(rows)
        comparison = result["comparisons"]["asap_random"]["1"]
        self.assertEqual(comparison["fast_minus_baseline"], 1.0)
        self.assertEqual((comparison["percentile_2_5"], comparison["percentile_97_5"]), (1.0, 1.0))
        self.assertTrue(comparison["reject_at_0_05"])
        with mock.patch.object(planning_study, "BOOTSTRAP_REPLICATES", 100):
            complete_case = planning_study._paired_statistics(
                [row for row in rows if int(row["assembly_id"]) < 234], 234
            )
        self.assertEqual(complete_case["comparisons"]["asap_random"]["1"]["fast_minus_baseline"], 1.0)

    def test_registry_records_each_models_available_evidence_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs" / "model"
            checkpoint = root / "training" / "seed-0.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            Path(str(checkpoint) + ".json").write_text("{}\n", encoding="utf-8")
            rows = [{"seed": 0, "best_checkpoint": "training/seed-0.pt",
                     "best_checkpoint_sha256": production._sha256_file(checkpoint), "parameter_count": 7}]
            for name in ("calibration", "validation"):
                evidence_root = root / name
                evidence_root.mkdir()
                (evidence_root / "seed-0.json").write_text('{"test_rows_loaded":0}\n', encoding="utf-8")
                entry = planner_integration._registry_model_entries(root, rows, evidence_root, name)[0]
                self.assertIn(name, entry)
                self.assertIn(name + "_sha256", entry)
                self.assertNotIn("calibration" if name == "validation" else "validation", entry)


if __name__ == "__main__":
    unittest.main()
