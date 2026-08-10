import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SklearnKNNClassifier
from sklearn.neighbors import KNeighborsRegressor as SklearnKNNRegressor

from knn import KNeighborsClassifier, KNeighborsRegressor


class KNeighborsClassifierTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [4, 4], [4, 5], [5, 4]])
        self.y = np.array(["near", "near", "near", "far", "far", "far"])
        self.query = np.array([[0.2, 0.1], [4.5, 4.2]])

    def test_uniform_predictions_and_probabilities_match_sklearn(self):
        ours = KNeighborsClassifier(n_neighbors=3).fit(self.X, self.y)
        reference = SklearnKNNClassifier(n_neighbors=3).fit(self.X, self.y)

        np.testing.assert_array_equal(ours.predict(self.query), reference.predict(self.query))
        np.testing.assert_allclose(
            ours.predict_proba(self.query), reference.predict_proba(self.query)
        )

    def test_distance_weighting_matches_sklearn(self):
        ours = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(
            self.X, self.y
        )
        reference = SklearnKNNClassifier(n_neighbors=3, weights="distance").fit(
            self.X, self.y
        )
        np.testing.assert_allclose(
            ours.predict_proba(self.query), reference.predict_proba(self.query)
        )

    def test_exact_match_ignores_nonzero_distance_neighbors(self):
        model = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(
            [[0], [1], [2]], ["exact", "other", "other"]
        )
        probabilities = model.predict_proba([[0]])[0]
        self.assertEqual(probabilities[np.where(model.classes_ == "exact")[0][0]], 1.0)

    def test_multiclass_and_manhattan_distance(self):
        X = [[0, 0], [3, 0], [0, 3]]
        y = ["a", "b", "c"]
        model = KNeighborsClassifier(n_neighbors=1, metric="manhattan").fit(X, y)
        np.testing.assert_array_equal(model.predict([[2.5, 0], [0, 2.5]]), ["b", "c"])


class KNeighborsRegressorTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(8, dtype=float).reshape(-1, 1)
        self.y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
        self.query = np.array([[1.5], [4.5]])

    def test_uniform_predictions_match_sklearn(self):
        ours = KNeighborsRegressor(n_neighbors=2).fit(self.X, self.y)
        reference = SklearnKNNRegressor(n_neighbors=2).fit(self.X, self.y)
        np.testing.assert_allclose(ours.predict(self.query), reference.predict(self.query))

    def test_distance_predictions_match_sklearn(self):
        ours = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(
            self.X, self.y
        )
        reference = SklearnKNNRegressor(n_neighbors=3, weights="distance").fit(
            self.X, self.y
        )
        np.testing.assert_allclose(ours.predict(self.query), reference.predict(self.query))

    def test_minkowski_distance_and_neighbor_lookup(self):
        model = KNeighborsRegressor(n_neighbors=2, metric="minkowski", p=3).fit(
            [[0, 0], [1, 1], [3, 3]], [0, 1, 3]
        )
        distances, indices = model.kneighbors([[0.8, 0.9]])
        np.testing.assert_array_equal(indices, [[1, 0]])
        self.assertTrue(np.all(np.diff(distances[0]) >= 0))


class KNNValidationTests(unittest.TestCase):
    def test_predict_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            KNeighborsClassifier().predict([[0]])

    def test_rejects_too_many_neighbors(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            KNeighborsRegressor(n_neighbors=3).fit([[0], [1]], [0, 1])

    def test_rejects_invalid_parameters_and_data(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            KNeighborsClassifier(n_neighbors=0)
        with self.assertRaisesRegex(ValueError, "metric"):
            KNeighborsClassifier(metric="cosine")
        with self.assertRaisesRegex(ValueError, "finite"):
            KNeighborsRegressor().fit([[0], [np.nan]], [0, 1])


if __name__ == "__main__":
    unittest.main()
