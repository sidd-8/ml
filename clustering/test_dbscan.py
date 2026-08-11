import unittest

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from clustering import DBSCAN


class DBSCANTests(unittest.TestCase):
    def setUp(self):
        X, _ = make_moons(n_samples=250, noise=0.06, random_state=8)
        self.X = StandardScaler().fit_transform(X)

    def test_euclidean_labels_and_core_samples_match_sklearn(self):
        ours = DBSCAN(eps=0.28, min_samples=5).fit(self.X)
        reference = SklearnDBSCAN(eps=0.28, min_samples=5).fit(self.X)

        np.testing.assert_array_equal(ours.labels_, reference.labels_)
        np.testing.assert_array_equal(
            ours.core_sample_indices_, reference.core_sample_indices_
        )
        np.testing.assert_allclose(ours.components_, reference.components_)
        self.assertEqual(ours.n_clusters_, 2)

    def test_manhattan_and_minkowski_match_sklearn(self):
        for metric, p in (("manhattan", 2), ("minkowski", 3)):
            ours = DBSCAN(eps=0.38, min_samples=5, metric=metric, p=p).fit(self.X)
            reference = SklearnDBSCAN(
                eps=0.38, min_samples=5, metric=metric, p=p
            ).fit(self.X)
            np.testing.assert_array_equal(ours.labels_, reference.labels_)
            np.testing.assert_array_equal(
                ours.core_sample_indices_, reference.core_sample_indices_
            )

    def test_noise_and_border_points(self):
        X = np.array([[0.0], [0.1], [0.2], [0.3], [2.0]])
        model = DBSCAN(eps=0.11, min_samples=3).fit(X)
        np.testing.assert_array_equal(model.labels_, [0, 0, 0, 0, -1])
        np.testing.assert_array_equal(model.core_sample_indices_, [1, 2])

    def test_all_noise_and_no_noise_cases(self):
        all_noise = DBSCAN(eps=0.1, min_samples=2).fit([[0], [1], [2]])
        np.testing.assert_array_equal(all_noise.labels_, [-1, -1, -1])
        self.assertEqual(all_noise.n_clusters_, 0)

        one_cluster = DBSCAN(eps=2, min_samples=1).fit([[0], [1], [2]])
        np.testing.assert_array_equal(one_cluster.labels_, [0, 0, 0])
        self.assertEqual(one_cluster.n_clusters_, 1)

    def test_fit_predict_returns_independent_labels(self):
        model = DBSCAN(eps=0.28, min_samples=5)
        labels = model.fit_predict(self.X)
        np.testing.assert_array_equal(labels, model.labels_)
        labels[0] = 99
        self.assertNotEqual(model.labels_[0], 99)


class DBSCANValidationTests(unittest.TestCase):
    def test_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "eps"):
            DBSCAN(eps=0)
        with self.assertRaisesRegex(ValueError, "min_samples"):
            DBSCAN(min_samples=0)
        with self.assertRaisesRegex(ValueError, "metric"):
            DBSCAN(metric="cosine")
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            DBSCAN(metric="minkowski", p=0.5)

    def test_rejects_non_finite_data(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            DBSCAN().fit([[0], [np.nan]])


if __name__ == "__main__":
    unittest.main()
