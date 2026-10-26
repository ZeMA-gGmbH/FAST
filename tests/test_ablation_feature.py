import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from fast import ablation_feature as feature


class AblationFeatureTest(unittest.TestCase):
    def test_weighted_average_precision_matches_sklearn_with_ties_and_multiplicities(self):
        rng = np.random.RandomState(3)
        targets = (rng.rand(400) > 0.6).astype(float)
        scores = np.round(rng.rand(400), 2)
        weights = rng.randint(0, 4, size=(5, 400)).astype(float)
        values = feature.weighted_average_precision(targets, scores, weights)
        for row, value in zip(weights, values):
            self.assertAlmostEqual(value, average_precision_score(targets, scores, sample_weight=row), places=12)
            repeated = np.repeat(np.arange(400), row.astype(int))
            self.assertAlmostEqual(value, average_precision_score(targets[repeated], scores[repeated]), places=12)
        self.assertAlmostEqual(feature.weighted_average_precision(targets, scores, np.ones(400))[0],
                               average_precision_score(targets, scores), places=12)

    def test_weighted_average_precision_is_undefined_without_positive_weight(self):
        self.assertTrue(np.isnan(feature.weighted_average_precision(np.asarray([0.0, 1.0]), np.asarray([0.2, 0.9]), np.asarray([[1.0, 0.0]]))[0]))


if __name__ == "__main__":
    unittest.main()
