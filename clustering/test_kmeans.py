import unittest

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering import KMeans


class KMeansTests(unittest.TestCase):
    def setUp(self):
        self.X, self.true_labels = make_blobs(
            n_samples=240,
            centers=[[-4, -2], [0, 4], [4, -1]],
            cluster_std=0.65,
            random_state=7,
        )

    def test_kmeans_plus_plus_matches_sklearn_quality(self):
        ours = KMeans(n_clusters=3, n_init=10, random_state=42).fit(self.X)
        reference = SklearnKMeans(
            n_clusters=3, n_init=10, random_state=42
        ).fit(self.X)

        self.assertGreater(adjusted_rand_score(self.true_labels, ours.labels_), 0.99)
        self.assertAlmostEqual(ours.inertia_, reference.inertia_, places=6)
        self.assertLessEqual(ours.n_iter_, ours.max_iter)

    def test_random_initialization_and_multiple_runs_are_reproducible(self):
        settings = dict(n_clusters=3, init="random", n_init=5, random_state=11)
        first = KMeans(**settings).fit(self.X)
        second = KMeans(**settings).fit(self.X)
        np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
        np.testing.assert_array_equal(first.labels_, second.labels_)
        self.assertEqual(first.inertia_, second.inertia_)

    def test_explicit_initial_centers_match_sklearn(self):
        initial = np.array([[-4, -2], [0, 4], [4, -1]], dtype=float)
        ours = KMeans(n_clusters=3, init=initial, n_init=20).fit(self.X)
        reference = SklearnKMeans(n_clusters=3, init=initial, n_init=1).fit(self.X)
        np.testing.assert_allclose(ours.cluster_centers_, reference.cluster_centers_)
        np.testing.assert_array_equal(ours.labels_, reference.labels_)
        self.assertAlmostEqual(ours.inertia_, reference.inertia_)

    def test_transform_predict_and_score_match_sklearn(self):
        initial = np.array([[-4, -2], [0, 4], [4, -1]], dtype=float)
        ours = KMeans(n_clusters=3, init=initial).fit(self.X)
        reference = SklearnKMeans(n_clusters=3, init=initial, n_init=1).fit(self.X)
        query = np.array([[-3, -2], [0, 3], [3, -1]])
        np.testing.assert_allclose(ours.transform(query), reference.transform(query))
        np.testing.assert_array_equal(ours.predict(query), reference.predict(query))
        self.assertAlmostEqual(ours.score(self.X), reference.score(self.X))

    def test_fit_predict_returns_independent_labels(self):
        model = KMeans(n_clusters=3, random_state=2)
        labels = model.fit_predict(self.X)
        np.testing.assert_array_equal(labels, model.labels_)
        labels[0] = 99
        self.assertNotEqual(model.labels_[0], 99)

    def test_empty_cluster_recovery_remains_finite(self):
        X = np.array([[0.0], [0.0], [10.0], [10.0]])
        initial = np.array([[0.0], [0.0], [10.0]])
        model = KMeans(n_clusters=3, init=initial, max_iter=10).fit(X)
        self.assertTrue(np.all(np.isfinite(model.cluster_centers_)))
        self.assertTrue(np.isfinite(model.inertia_))


class KMeansValidationTests(unittest.TestCase):
    def test_predict_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            KMeans().predict([[0]])

    def test_rejects_too_many_clusters(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            KMeans(n_clusters=3).fit([[0], [1]])

    def test_rejects_invalid_parameters_and_centers(self):
        with self.assertRaisesRegex(ValueError, "n_clusters"):
            KMeans(n_clusters=0)
        with self.assertRaisesRegex(ValueError, "init"):
            KMeans(init="bad")
        with self.assertRaisesRegex(ValueError, "shape"):
            KMeans(n_clusters=2, init=[[0]]).fit([[0], [1]])

    def test_rejects_non_finite_data(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            KMeans(n_clusters=2).fit([[0], [np.nan]])


if __name__ == "__main__":
    unittest.main()
