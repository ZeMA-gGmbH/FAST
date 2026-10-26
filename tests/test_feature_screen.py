import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fast import cad_features as features
from fast import feature_screen as screen
from plan_sequence import production_sim_labels as production
from tests.test_cad_features import _record


class ScalarFeatureScreenTest(unittest.TestCase):
    def test_feature_group_selection(self):
        candidate_columns, action_columns, groups = screen._feature_set_columns(
            "only:direction",
            list(features.CANDIDATE_FEATURES),
            list(features.ACTION_FEATURES),
        )
        self.assertEqual(candidate_columns, [])
        self.assertEqual(groups, ["direction"])
        self.assertTrue(action_columns)
        self.assertTrue(
            all(
                features.ACTION_FEATURES[index].startswith("direction__")
                for index in action_columns
            )
        )

    def test_ranking_reports_positive_and_all_negative_contexts(self):
        data = {
            "action_candidate": np.array([0, 1, 2, 3]),
            "candidate_split": np.array([1, 1, 1, 1]),
            "candidate_tier": np.array([0, 0, 0, 0]),
            "candidate_query": np.array([10, 10, 20, 20]),
            "action_y_valid": np.array([True, True, True, True]),
            "action_y": np.array([0, 1, 0, 0]),
        }
        result = screen._ranking_metrics(
            data, np.array([0.1, 0.9, 0.8, 0.2]), tier_id=0
        )
        self.assertEqual(result["positive_context_count"], 1)
        self.assertEqual(result["all_negative_context_count"], 1)
        self.assertEqual(result["censored_context_count"], 0)
        self.assertEqual(result["success_at_1"], 1.0)
        self.assertEqual(result["recall_at_1"], 1.0)
        self.assertEqual(result["recall_at_3"], 1.0)
        self.assertEqual(result["first_feasible_mrr"], 1.0)

    def test_ranking_excludes_and_counts_partially_censored_contexts(self):
        data = {
            "action_candidate": np.array([0, 1]),
            "candidate_split": np.array([1, 1]),
            "candidate_tier": np.array([0, 0]),
            "candidate_query": np.array([10, 10]),
            "action_y_valid": np.array([True, False]),
            "action_y": np.array([1, 0]),
        }
        result = screen._ranking_metrics(data, np.array([0.1, 0.9]))
        self.assertEqual(result["context_count"], 1)
        self.assertEqual(result["censored_context_count"], 1)
        self.assertEqual(result["positive_context_count"], 0)
        self.assertIsNone(result["success_at_1"])

    def test_equal_mass_ece(self):
        targets = np.array([0, 0, 1, 1])
        calibrated = np.array([0.0, 0.0, 1.0, 1.0])
        uncalibrated = np.full(4, 0.5)
        self.assertEqual(screen._equal_mass_ece(targets, calibrated), 0.0)
        self.assertEqual(screen._equal_mass_ece(targets, uncalibrated), 0.5)
        self.assertIn(
            "expected_calibration_error_15_equal_mass",
            screen._binary_metrics(targets, calibrated),
        )

    def test_combined_reference_keeps_part_order_primary(self):
        data = {
            "candidate_names": ["pose_support__candidate_outside_distance"],
            "candidate_x": np.array([[2.0], [1.0]]),
            "candidate_query": np.array([0, 0]),
            "candidate_split": np.array([0, 0]),
            "candidate_t_valid": np.array([True, True]),
            "candidate_t": np.array([1, 1]),
            "action_candidate": np.repeat(np.array([0, 1]), 6),
            "action_direction": np.tile(np.arange(6), 2),
            "action_r_valid": np.ones(12, dtype=bool),
            "action_r": np.tile(np.array([0, 0, 0, 0, 0, 1]), 2),
        }
        probability = screen._heur_out_direction_prior_probability(data)
        self.assertGreater(probability[:6].min(), probability[6:].max())
        self.assertEqual(int(np.argmax(probability[:6])), 5)

    def test_fitted_screen_uses_train_and_validation_without_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            shards = []
            scopes = {}
            for index, split in enumerate(("train", "validation")):
                candidate = copy.deepcopy(_record())
                candidate["dataset_split"] = split
                candidate["assembly_id"] = "{:05d}".format(index + 1)
                candidate["candidate_id"] = "{}-candidate".format(split)
                candidate["query_id"] = "{}-query".format(split)
                for action_index, action in enumerate(candidate["actions"]):
                    action["action_id"] = "{}-{}".format(split, action_index)
                path = output_root / "shards" / split / "00001.jsonl"
                features._write_atomic(
                    path,
                    [
                        {
                            "schema_version": features.SHARD_SCHEMA_VERSION,
                            "record_type": "header",
                        },
                        candidate,
                        {
                            "schema_version": features.SHARD_SCHEMA_VERSION,
                            "record_type": "summary",
                            "complete": True,
                        },
                    ],
                )
                shards.append(
                    {
                        "path": path.relative_to(output_root).as_posix(),
                        "sha256": production._sha256_file(path),
                        "dataset_split": split,
                    }
                )
                scopes[split + "/small"] = {
                    "candidate_count": 1,
                    "action_count": 6,
                }
            build = {
                "schema_version": features.BUILD_SCHEMA_VERSION,
                "status": "passed",
                "feature_manifest_sha256": "a" * 64,
                "extractor_sha256": "b" * 64,
                "candidate_features": list(features.CANDIDATE_FEATURES),
                "action_features": list(features.ACTION_FEATURES),
                "per_split_tier": scopes,
                "shards": shards,
            }
            features._write_atomic(output_root / "manifest.json", [build])
            data = screen._load_data(
                {
                    "output_root": output_root,
                    "manifest_sha256": "a" * 64,
                    "extractor_sha256": "b" * 64,
                }
            )
            candidate_columns, action_columns, _ = screen._feature_set_columns(
                "full", data["candidate_names"], data["action_names"]
            )
            r_probability, t_probability = screen._fitted_probabilities(
                data, "logistic", candidate_columns, action_columns
            )
            result = screen._evaluate(data, r_probability, t_probability)
            self.assertEqual(len(data["candidate_x"]), 2)
            self.assertEqual(result["overall"]["R"]["count"], 6)
            self.assertEqual(result["overall"]["ranking"]["positive_context_count"], 1)

if __name__ == "__main__":
    unittest.main()
