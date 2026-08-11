import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from metrics import silhouette_score


class SilhouetteScoreTests(unittest.TestCase):
    def test_matches_sklearn(self):
        X, labels = make_blobs(n_samples=80, centers=3, random_state=4)
        self.assertAlmostEqual(
            silhouette_score(X, labels), sklearn_silhouette_score(X, labels)
        )

    def test_singleton_cluster_has_zero_sample_coefficient(self):
        X = np.array([[0.0], [10.0], [11.0]])
        labels = np.array([0, 1, 1])
        expected = sklearn_silhouette_score(X, labels)
        self.assertAlmostEqual(silhouette_score(X, labels), expected)

    def test_rejects_invalid_cluster_counts(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            silhouette_score([[0], [1]], [0, 0])
        with self.assertRaisesRegex(ValueError, "requires"):
            silhouette_score([[0], [1]], [0, 1])


if __name__ == "__main__":
    unittest.main()
