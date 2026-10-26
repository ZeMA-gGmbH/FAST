import copy
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("locked Torch runtime is unavailable")

import torch
import trimesh

from fast import ranker_models, action_planner
from fast import planner_integration as integration
from fast import mlp_check as mlp


def _registry():
    return {
        "schema_version": mlp.REGISTRY_SCHEMA, "status": "locked", "models": {"architecture": "mlp", "seeds": [{
            "seed": 0, "checkpoint_sha256": "a" * 64, "calibration_sha256": "b" * 64}]},
        "planner_contract": copy.deepcopy(mlp.CONTRACT), "population": {"assembly_count": 234},
        "matrix": {"row_count": 234}, "output_root": mlp.OUTPUT_ROOT,
        "aggregate_outputs": {"per_assembly_test_payload_export": False},
    }


class MLPCheckTest(unittest.TestCase):
    def test_mlp_architecture(self):
        self.assertEqual(sum(p.numel() for p in ranker_models.build_model("mlp").parameters()), 91266)

    def test_ranker_matches_direct_mlp_inference_and_fast_default_is_unchanged(self):
        self.assertEqual(inspect.signature(integration.FactorizedRanker).parameters["architecture"].default,
                         "graph_transformer")
        self.assertEqual(integration.METHODS, ("asap_random", "asap_heur_out", "asap_released_gatv2",
                                               "cebulla_gcn", "heur_out_direction_prior", "fast"))
        normalization = {"node": {"mean": [0.0] * 41, "std": [1.0] * 41}, "action": {"mean": [0.0] * 16, "std": [1.0] * 16}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(0)
            checkpoint = root / "mlp.pth"
            torch.save(ranker_models.build_model("mlp").state_dict(), checkpoint)  # weights-only release format
            for index, offset in enumerate((0.0, 1.0, 2.0)):
                mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
                mesh.apply_translation((offset, 0.0, 0.4))
                mesh.export(root / "{}.obj".format(index))
            (root / "contact_graph.json").write_text(json.dumps({
                "directed": False, "multigraph": False, "graph": {}, "nodes": [{"id": str(i)} for i in range(3)],
                "links": [{"source": "0", "target": "1"}, {"source": "1", "target": "2"}]}), encoding="utf-8")
            builder = integration.OnlineFeatureBuilder(root, {"support_tolerance_fraction": 0.001})
            ranker = integration.make_ranker(mlp.METHOD, 0, root, builder, normalization, mlp_checkpoint=checkpoint)
            parts = ("0", "1", "2")
            ranked = ranker(parts, "pose-000", None, "ctx")
            self.assertEqual(ranked, ranker(parts, "pose-000", None, "ctx"))
            model = ranker_models.build_model("mlp")
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            batch = integration._torch_online_batch(builder.record(parts, "pose-000", None), normalization)
            with torch.no_grad():
                expected = ranker_models.probabilities(model.eval()(batch))["Y"].numpy()
            for row in ranked:
                index = parts.index(row["part"])
                self.assertAlmostEqual(row["score"], float(expected[index, action_planner.DIRECTIONS.index(row["direction"])]), places=6)

    def test_contract_rejects_every_changed_field(self):
        mlp.check_contract(_registry())
        mutations = [
            ("models", "architecture", "graph_transformer"), ("planner_contract", "h_max", 2),
            ("planner_contract", "seeds", [1]), ("planner_contract", "directions", ["+Z"]),
            ("planner_contract", "budgets", [1, 2]), ("planner_contract", "max_stable_poses", 4),
            ("planner_contract", "wall_timeout_seconds", 3600), ("population", "assembly_count", 233),
            ("matrix", "row_count", 233), ("aggregate_outputs", "per_assembly_test_payload_export", True),
        ]
        for section, key, value in mutations:
            registry = _registry()
            registry[section][key] = value
            with self.assertRaisesRegex(Exception, "MLP planning-check contract"):
                mlp.check_contract(registry)
        for key, value in (("checkpoint_sha256", "missing"), ("calibration_sha256", None), ("seed", 1)):
            registry = _registry()
            registry["models"]["seeds"][0][key] = value
            with self.assertRaises(Exception):
                mlp.check_contract(registry)
        registry = _registry()
        registry["output_root"] = "runs/planning-study-v4"
        with self.assertRaises(Exception):
            mlp.check_contract(registry)

    def test_registry_is_required_before_any_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "registry.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "registry hash differs"):
                mlp.evaluate("spec", directory, "/nonexistent/test/source", "assets", directory, "0" * 64,
                             "released", "prior", 0)

    def test_paired_intervals_are_deterministic_and_carry_no_p_values(self):
        tiers = {"a{}".format(i): ("small", "medium", "large")[i % 3] for i in range(30)}
        fast = {a: {str(b): (i % 2 == 0) for b in mlp.BUDGETS} for i, a in enumerate(sorted(tiers))}
        same = mlp.paired_intervals(fast, fast, tiers)
        self.assertEqual(same["8"]["fast_minus_mlp"], 0.0)
        self.assertEqual((same["8"]["percentile_2_5"], same["8"]["percentile_97_5"]), (0.0, 0.0))
        worse = {a: {str(b): False for b in mlp.BUDGETS} for a in tiers}
        first = mlp.paired_intervals(fast, worse, tiers)
        self.assertEqual(first, mlp.paired_intervals(fast, worse, tiers))
        self.assertEqual(first["1"]["fast_minus_mlp"], 0.5)
        self.assertLessEqual(first["1"]["percentile_2_5"], 0.5)
        self.assertNotIn("p_value", json.dumps(first))
        with self.assertRaisesRegex(Exception, "paired assemblies differ"):
            mlp.paired_intervals(fast, {"a0": worse["a0"]}, tiers)

    def test_trace_audit_accounts_rows_without_reading_outcomes(self):
        from plan_sequence import production_sim_labels as production
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_dir = root / "registry"
            registry_dir.mkdir()
            rows = ["\t".join(mlp.MATRIX_FIELDS)]
            rows += ["{}\tmlp_check\t0\tA{:03d}\tsmall\t3\t{}".format(i, i, "0" * 64) for i in range(234)]
            (registry_dir / "matrix.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            registry = _registry()
            registry["matrix"]["sha256"] = production._sha256_file(registry_dir / "matrix.tsv")
            (registry_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
            sha = production._sha256_file(registry_dir / "registry.json")
            result = mlp.audit_traces(root, registry_dir, sha, root / "audit.json")
            self.assertEqual((result["valid_traces"], result["technical_failures"]), (0, 234))
            self.assertNotIn("success", (root / "audit.json").read_text(encoding="utf-8"))
            removal = {"label": True, "status": "success", "reason": None, "rollout_count": 1, "runtime_s": 0.1}
            action_planner.write_trace(
                mlp._mlp_trace(root, "A007"),
                {"record_type": "header", "schema_version": action_planner.TRACE_SCHEMA_VERSION,
                 "dataset_scope": "locked_test", "registry_sha256": sha, "method": mlp.METHOD, "seed": 0,
                 "assembly_id": "A007", "input_sha256": "0" * 64, "architecture": "mlp",
                 "checkpoint_sha256": "a" * 64, "calibration_sha256": "b" * 64,
                 "matrix_process": 7, "test_rows_loaded": 1},
                [{"accepted_transition": True, "R": removal, "T": dict(removal)}],
                {"record_type": "summary", "complete": True, "test_rows_loaded": 1, "mandatory_redmax_validation": True,
                 "unvalidated_leaf_shortcut": False, "action_validation_count": 1,
                 "budgets": {str(b): {"absolute_action_validations": 2 * b, "success": False} for b in mlp.BUDGETS}})
            result = mlp.audit_traces(root, registry_dir, sha, root / "audit-ok.json")
            self.assertEqual((result["valid_traces"], result["technical_failures"]), (1, 233))


if __name__ == "__main__":
    unittest.main()
