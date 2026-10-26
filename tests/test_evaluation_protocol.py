import json
import unittest
from pathlib import Path


class DatasetEvaluationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.protocol = json.loads(
            (
                cls.root
                / "configs"
                / "evaluation_protocol.json"
            ).read_text(encoding="utf-8")
        )

    def test_dataset_freeze_matches_dataset_index_config(self):
        index_config = json.loads(
            (self.root / "configs" / "dataset_index.json").read_text(
                encoding="utf-8"
            )
        )
        dataset = self.protocol["dataset"]
        self.assertEqual(dataset["phase"], index_config["phase"])
        self.assertEqual(dataset["external_root"], index_config["output_root"])
        self.assertEqual(dataset["splits"], index_config["expected_assemblies"])
        self.assertEqual(
            sum(dataset["splits"][name]["total"] for name in ("train", "validation", "test")),
            dataset["eligible_assembly_count"],
        )
        self.assertEqual(
            dataset["eligible_assembly_count"] + dataset["splits"]["excluded"]["total"],
            dataset["artifacts"]["assemblies"]["row_count"],
        )

    def test_action_table_and_target_contract(self):
        dataset = self.protocol["dataset"]
        self.assertEqual(
            dataset["artifacts"]["actions"]["row_count"],
            6 * dataset["artifacts"]["candidates"]["row_count"],
        )
        problem = self.protocol["prediction_problem"]
        self.assertEqual(len(problem["directions"]), 6)
        self.assertEqual(
            problem["directions"], ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]
        )
        self.assertEqual(problem["targets"]["R"]["unit"], "(state, pose, part, direction)")
        self.assertEqual(problem["targets"]["T"]["unit"], "(state, pose, part)")
        self.assertTrue(problem["targets"]["T"]["direction_invariant_within_candidate"])
        self.assertIn("never a negative", problem["missing_target_policy"]["null_meaning"])

    def test_budget_units_cannot_be_conflated(self):
        costs = self.protocol["cost_accounting"]
        self.assertEqual(costs["primary_counter"]["name"], "action_validation")
        self.assertEqual(costs["secondary_counter"]["name"], "physics_rollout")
        self.assertEqual(costs["legacy_counter"]["name"], "asap_legacy_evaluation")
        self.assertEqual(costs["ideal_successful_transition_count"], "part_count - 1")
        budgets = self.protocol["budgets"]
        self.assertEqual(budgets["primary_normalized_action_validation"], [1, 2, 4, 8])
        self.assertEqual(budgets["asap_legacy_reproduction"], [50, 400])
        self.assertIn("never compare", costs["legacy_counter"]["purpose"])

    def test_primary_test_and_leakage_are_locked(self):
        self.assertTrue(
            self.protocol["dataset"]["split_policy"][
                "test_is_never_used_for_training_model_selection_calibration_or_threshold_selection"
            ]
        )
        self.assertEqual(
            self.protocol["outcomes"]["primary"]["population"],
            "all 237 eligible unseen test assemblies",
        )
        forbidden = " ".join(self.protocol["leakage_boundary"]["forbidden_model_inputs"])
        self.assertIn("simulator status", forbidden)
        self.assertIn("test labels", forbidden)
        self.assertFalse(
            self.protocol["planning_protocol"]["primary_condition"]["plan_grasp"]
        )
        self.assertFalse(
            self.protocol["planning_protocol"]["primary_condition"]["plan_arm"]
        )


if __name__ == "__main__":
    unittest.main()
