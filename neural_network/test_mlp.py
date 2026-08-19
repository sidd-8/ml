import unittest

import numpy as np
from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network import MLPClassifier, MLPRegressor


class MLPClassifierTests(unittest.TestCase):
    def test_nonlinear_classification_and_probabilities(self):
        X, y = make_moons(n_samples=240, noise=0.12, random_state=2)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=4)
        model = MLPClassifier((16, 8), learning_rate=0.01, n_iters=300, batch_size=32, tol=1e-5, n_iter_no_change=30, random_state=3).fit(X_train, y_train)
        self.assertGreater(model.score(X_test, y_test), 0.9)
        np.testing.assert_allclose(model.predict_proba(X_test).sum(axis=1), 1.0)

    def test_multiclass_arbitrary_labels_and_reproducibility(self):
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]], dtype=float)
        y = np.array(["a", "b", "c", "c", "a"])
        settings = dict(hidden_layer_sizes=(6,), n_iters=20, random_state=8)
        first, second = MLPClassifier(**settings).fit(X, y), MLPClassifier(**settings).fit(X, y)
        for left, right in zip(first.coefs_, second.coefs_):
            np.testing.assert_allclose(left, right)


class MLPRegressorTests(unittest.TestCase):
    def test_regression_learns_smooth_mapping(self):
        X, y = make_regression(n_samples=220, n_features=4, noise=2, random_state=5)
        X = StandardScaler().fit_transform(X)
        y = (y - y.mean()) / y.std()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)
        model = MLPRegressor((16,), activation="tanh", learning_rate=0.01, n_iters=300, batch_size=32, n_iter_no_change=40, random_state=2).fit(X_train, y_train)
        self.assertGreater(model.score(X_test, y_test), 0.9)
        self.assertEqual(model.predict(X_test).ndim, 1)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "activation"):
            MLPRegressor(activation="bad")
        with self.assertRaisesRegex(ValueError, "fit"):
            MLPRegressor().predict([[0]])


if __name__ == "__main__":
    unittest.main()
