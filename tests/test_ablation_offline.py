import unittest

import numpy as np

from fast import ablation_offline as offline


def _prediction(r, t, y_targets, fully_observed=True, part_ids=None):
    r = np.asarray(r, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    return {"R_score": r, "T_score": t, "Y_score": r * t[:, None], "Y": np.asarray(y_targets, dtype=np.float32),
            "fully_observed": fully_observed, "part_ids": part_ids or [str(index) for index in range(len(t))]}


class AblationOfflineTest(unittest.TestCase):
    def test_a2_ties_resolve_by_part_then_frozen_direction_order(self):
        prediction = _prediction(np.full((2, 6), 0.5), [0.7, 0.7], np.zeros((2, 6)))
        order = offline.variant_orders(prediction, list(range(6)))["A2"]
        np.testing.assert_array_equal(order, np.arange(12))

    def test_ties_use_part_id_string_order_not_stored_candidate_order(self):
        prediction = _prediction(np.full((3, 6), 0.5), [0.7, 0.7, 0.7], np.zeros((3, 6)), part_ids=["7", "10", "2"])
        orders = offline.variant_orders(prediction, list(range(6)))
        # String order is "10" < "2" < "7": stored candidates 1, 2, 0.
        np.testing.assert_array_equal(orders["A0"][:7], [6, 7, 8, 9, 10, 11, 12])
        np.testing.assert_array_equal(orders["A2"][12:], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(orders["B1"][::6], [6, 12, 0])

    def test_a0_a1_and_b_orders(self):
        r = [[0.1, 0.9, 0.2, 0.2, 0.2, 0.2], [0.8, 0.1, 0.1, 0.1, 0.1, 0.1]]
        t = [0.5, 1.0]
        orders = offline.variant_orders(_prediction(r, t, np.zeros((2, 6))), [5, 4, 3, 2, 1, 0])
        self.assertEqual(int(orders["A0"][0]), 6)   # 0.8*1.0 beats 0.9*0.5
        self.assertEqual(int(orders["A1"][0]), 1)   # removability alone prefers part 0 direction -Z
        self.assertEqual(int(orders["A2"][0]), 6)   # stability alone prefers part 1
        np.testing.assert_array_equal(orders["B1"], [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(orders["B2"], [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    def test_context_values_and_eligibility(self):
        targets = np.zeros((2, 6), dtype=np.float32)
        targets[1, 2] = 1.0
        values = offline.context_values(targets, np.arange(12))
        self.assertEqual(values["mean_first_feasible_rank"], 9.0)
        self.assertEqual(values["offline_context_success_at_1"], 0.0)
        self.assertEqual(values["offline_context_success_at_3"], 0.0)
        self.assertAlmostEqual(values["first_feasible_mrr"], 1.0 / 9.0)
        self.assertTrue(offline.eligible({"fully_observed": True, "Y": targets}))
        self.assertFalse(offline.eligible({"fully_observed": False, "Y": targets}))
        self.assertFalse(offline.eligible({"fully_observed": True, "Y": np.zeros(3)}))

    def test_prior_direction_order_breaks_ties_by_frozen_order(self):
        rates = {"+Z": 0.2, "-Z": 0.5, "+X": 0.5, "-X": 0.1, "+Y": 0.2, "-Y": 0.0}
        self.assertEqual(offline.prior_direction_order(rates), [1, 2, 0, 4, 3, 5])

    def test_threshold_table_uses_all_labels_and_resolved_denominators(self):
        values = np.asarray([1, 1, 0, 0, 0, 1], dtype=np.float32)
        valid = np.asarray([True, True, True, True, False, False])
        probabilities = np.asarray([0.9, 0.2, 0.5, 0.1, 0.8, 0.3], dtype=np.float32)
        counts = offline.threshold_counts(values, valid, probabilities)
        self.assertEqual([counts[name] for name in ("TP", "TN", "FP", "FN", "unresolved", "total")], [1, 1, 1, 1, 2, 6])
        self.assertEqual(counts["probability_exactly_threshold"], 1)
        rates = offline.quality_rates(counts)
        self.assertAlmostEqual(rates["TP_percent"], 100.0 / 6.0)
        self.assertAlmostEqual(rates["unresolved_percent"], 200.0 / 6.0)
        self.assertEqual(rates["accuracy_percent"], 50.0)
        self.assertEqual(rates["precision_percent"], 50.0)
        self.assertEqual(rates["recall_percent"], 50.0)

    def test_ranking_tables_count_rows_and_assembly_sums(self):
        rows = [("a", "small", True, {metric: 1.0 for metric in offline.METRICS}),
                ("a", "small", False, None),
                ("b", "large", True, {metric: 0.0 for metric in offline.METRICS})]
        tables, assemblies = offline.ranking_tables(rows)
        self.assertEqual(tables["all"]["stored_context_count"], 3)
        self.assertEqual(tables["all"]["ranking_eligible_context_count"], 2)
        self.assertEqual(tables["small"]["assembly_count"], 1)
        self.assertIsNone(tables["medium"]["offline_context_success_at_1"])
        self.assertEqual(assemblies["a"]["offline_context_success_at_1"], [1.0, 1])

    def test_paired_bootstrap_difference_is_seed_averaged_and_paired(self):
        order = [{"assembly_id": "a", "tier": "small"}, {"assembly_id": "b", "tier": "small"}]
        samples = np.asarray([[0, 1], [0, 0], [1, 1]])
        metric = "offline_context_success_at_1"
        variant = [{"a": {metric: [1.0, 1]}, "b": {metric: [0.0, 1]}}] * 3
        control = [{"a": {metric: [1.0, 1]}, "b": {metric: [1.0, 1]}}] * 3
        result = offline.paired_difference(variant, control, metric, order, samples, np.arange(2))
        self.assertAlmostEqual(result["replicate_mean"], (-0.5 + 0.0 - 1.0) / 3)
        self.assertEqual(result["undefined_replicates"], 0)


if __name__ == "__main__":
    unittest.main()
