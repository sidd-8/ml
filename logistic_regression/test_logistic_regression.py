import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression


class LogisticRegressionTests(unittest.TestCase):
    def test_learns_linearly_separable_data_and_converges(self):
        X = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.2, n_iters=10_000, tolerance=1e-8, l2=0.01
        ).fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
        self.assertGreater(model.score(X, y), 0.99)
        self.assertLess(model.loss_history[-1], model.loss_history[0])
        self.assertTrue(model.converged_)
        self.assertLess(model.n_iters_, model.n_iters)

    def test_matches_sklearn_probabilities(self):
        X, y = make_classification(
            n_samples=300,
            n_features=4,
            n_redundant=0,
            class_sep=1.2,
            random_state=11,
        )
        X = StandardScaler().fit_transform(X)
        ours = LogisticRegression(
            learning_rate=0.1, n_iters=5000, tolerance=1e-11
        ).fit(X, y)
        reference = SklearnLogisticRegression(C=1e8, tol=1e-11).fit(X, y)

        np.testing.assert_allclose(
            ours.predict_proba(X)[:, 1],
            reference.predict_proba(X)[:, 1],
            atol=2e-4,
        )
        self.assertEqual(ours.score(X, y), reference.score(X, y))

    def test_predict_proba_and_decision_function(self):
        X = np.array([[0, 0], [1, 1], [2, 1], [-1, -2]], dtype=float)
        y = np.array([0, 1, 1, 0])
        model = LogisticRegression(n_iters=100).fit(X, y)

        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities.shape, (4, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4))
        np.testing.assert_allclose(
            probabilities[:, 1], model._sigmoid(model.decision_function(X))
        )
        self.assertEqual(model.coef_.shape, (2,))
        self.assertIsInstance(model.intercept_, float)

    def test_supports_arbitrary_binary_labels(self):
        X = np.array([[-2], [-1], [1], [2]], dtype=float)
        y = np.array(["no", "no", "yes", "yes"])
        model = LogisticRegression(n_iters=500).fit(X, y)

        np.testing.assert_array_equal(model.classes_, ["no", "yes"])
        np.testing.assert_array_equal(model.predict(X), y)

    def test_minibatch_training_is_reproducible(self):
        X, y = make_classification(
            n_samples=100, n_features=5, class_sep=2.0, random_state=3
        )
        X = StandardScaler().fit_transform(X)
        settings = dict(
            learning_rate=0.02,
            n_iters=200,
            batch_size=16,
            random_state=42,
        )
        first = LogisticRegression(**settings).fit(X, y)
        second = LogisticRegression(**settings).fit(X, y)

        np.testing.assert_allclose(first.get_theta(), second.get_theta())
        self.assertGreater(first.score(X, y), 0.9)

    def test_l2_shrinks_coefficients(self):
        X, y = make_classification(n_samples=200, n_features=4, random_state=5)
        X = StandardScaler().fit_transform(X)
        unregularized = LogisticRegression(n_iters=1000).fit(X, y)
        regularized = LogisticRegression(n_iters=1000, l2=1.0).fit(X, y)

        self.assertLess(
            np.linalg.norm(regularized.coef_), np.linalg.norm(unregularized.coef_)
        )

    def test_balanced_class_weights(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.array([0] * 8 + [1] * 2)
        model = LogisticRegression(class_weight="balanced", n_iters=10).fit(X, y)

        self.assertAlmostEqual(model.class_weight_[0], 10 / 16)
        self.assertAlmostEqual(model.class_weight_[1], 10 / 4)

    def test_gradient_matches_finite_difference(self):
        X = np.array([[-1.0, 0.5], [0.5, 1.0], [2.0, -1.0]])
        y = np.array([0.0, 1.0, 1.0])
        model = LogisticRegression(l2=0.1)
        model.weights = np.array([0.2, -0.3])
        model.bias = 0.1
        sample_weight = np.ones(3)
        _, gradient, bias_gradient = model._loss_and_gradients(X, y, sample_weight)

        epsilon = 1e-6
        numerical = np.empty(2)
        for index in range(2):
            model.weights[index] += epsilon
            plus = model._loss_and_gradients(X, y, sample_weight)[0]
            model.weights[index] -= 2 * epsilon
            minus = model._loss_and_gradients(X, y, sample_weight)[0]
            model.weights[index] += epsilon
            numerical[index] = (plus - minus) / (2 * epsilon)
        model.bias += epsilon
        plus = model._loss_and_gradients(X, y, sample_weight)[0]
        model.bias -= 2 * epsilon
        minus = model._loss_and_gradients(X, y, sample_weight)[0]
        model.bias += epsilon

        np.testing.assert_allclose(gradient, numerical, atol=1e-7)
        self.assertAlmostEqual(bias_gradient, (plus - minus) / (2 * epsilon), places=7)

    def test_rejects_non_binary_targets(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            LogisticRegression().fit([[0], [1], [2]], [0, 1, 2])

    def test_predict_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            LogisticRegression().predict([[0]])


if __name__ == "__main__":
    unittest.main()
