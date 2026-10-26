import tempfile
import unittest
from pathlib import Path

import numpy as np

from fast import posthoc


class PosthocPosthocTest(unittest.TestCase):
    def test_threshold_inversion_censoring_and_undefined_target(self):
        events = np.asarray([[1.0, 2.0, 3.0], [np.inf, np.inf, np.inf]])
        self.assertEqual(posthoc._empirical_threshold(events, 0.5), 3.0)
        self.assertIsNone(posthoc._empirical_threshold(events, 0.6))

    def test_cumulative_rollouts_stop_at_solution_action(self):
        actions = [
            {"redmax_rollouts": 2, "stability_restarts": 1},
            {"redmax_rollouts": 3, "stability_restarts": 2},
            {"redmax_rollouts": 100, "stability_restarts": 100},
        ]
        self.assertEqual(
            posthoc._resource_event(actions, {"solution_action_validation": 2}, 3),
            (1.0, 4.0),
        )
        self.assertEqual(
            posthoc._resource_event(actions, {"solution_action_validation": None}, 3),
            (None, None),
        )

    def test_tier_stratified_resampling_preserves_strata(self):
        tiers = ["small", "small", "medium", "large", "large"]
        samples = posthoc._resample_indices(tiers, replicates=20, seed=7)
        self.assertTrue(np.all(np.isin(samples[:, :2], [0, 1])))
        self.assertTrue(np.all(samples[:, 2:3] == 2))
        self.assertTrue(np.all(np.isin(samples[:, 3:], [3, 4])))
        np.testing.assert_array_equal(samples, posthoc._resample_indices(tiers, 20, 7))

    def test_paired_bootstrap_carries_methods_and_seeds_together(self):
        records = []
        for assembly, tier, value in (("a", "small", 1.0), ("b", "medium", 2.0),
                                      ("c", "large", 3.0)):
            for method in posthoc.METHODS:
                for seed in (0, 1, 2):
                    offset = 0.0 if method == "fast" else 1.0
                    records.append({"assembly_id": assembly, "tier": tier, "method": method,
                                    "seed": seed, "validations": value + offset,
                                    "rollouts": 2 * value + offset})
        result = posthoc._matched_success(records, replicates=100, seed=9)
        row = result["validations"]["comparisons"]["asap_random"]["0.3"]
        self.assertEqual(row["fast_minus_baseline"], -1.0)
        self.assertEqual((row["percentile_2_5"], row["percentile_97_5"]), (-1.0, -1.0))
        self.assertEqual(row["undefined_replicates"], 0)

    def test_bootstrap_reports_undefined_replicates(self):
        values = np.asarray([[1.0, np.inf, np.inf], [np.inf, np.inf, np.inf],
                             [np.inf, np.inf, np.inf]])
        samples = posthoc._resample_indices(["small", "medium", "large"], 50, 3)
        thresholds = posthoc._bootstrap_thresholds(values, samples, (0.3,))
        self.assertTrue(np.isnan(thresholds).any())

    def test_rejects_mutated_immutable_input_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable differs"):
                posthoc._verify_hash(path, "0" * 64, "immutable")
            output = Path(directory) / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                posthoc._require(not output.exists(), "post-hoc output root cannot be overwritten")

    def test_feature_report_paths_are_artifact_root_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "screening" / "histgb__full.json"
            report.parent.mkdir()
            report.write_text("{}\n", encoding="utf-8")
            posthoc._verify_hash(
                root / "screening/histgb__full.json",
                posthoc._sha256(report),
                "scalar-feature screen report",
            )


if __name__ == "__main__":
    unittest.main()
