import unittest

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from linear_regression import GradientDescentLR, NormalEquationLR


class NormalEquationTests(unittest.TestCase):
    def test_recovers_exact_line(self):
        X = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
        y = 3.0 + 2.5 * X[:, 0]

        model = NormalEquationLR().fit(X, y)

        self.assertAlmostEqual(model.intercept_, 3.0)
        self.assertAlmostEqual(model.coef_[0], 2.5)
        np.testing.assert_allclose(model.predict(X), y)
        self.assertAlmostEqual(model.score(X, y), 1.0)

    def test_matches_sklearn_on_rank_deficient_data(self):
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)

        ours = NormalEquationLR().fit(X, y)
        reference = LinearRegression().fit(X, y)

        np.testing.assert_allclose(ours.predict(X), reference.predict(X), atol=1e-10)
        self.assertLess(ours.rank_, X.shape[1] + 1)


class GradientDescentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        raw_X = rng.normal(size=(150, 3))
        self.X = StandardScaler().fit_transform(raw_X)
        self.y = 1.25 + self.X @ np.array([2.0, -3.0, 0.75])

    def test_matches_sklearn(self):
        ours = GradientDescentLR(lr=0.1, n_iters=1000, tolerance=1e-12).fit(
            self.X, self.y
        )
        reference = LinearRegression().fit(self.X, self.y)

        np.testing.assert_allclose(ours.coef_, reference.coef_, atol=1e-5)
        self.assertAlmostEqual(ours.intercept_, reference.intercept_, places=5)
        self.assertGreater(ours.score(self.X, self.y), 0.99999)
        self.assertTrue(ours.converged_)
        self.assertLess(ours.n_iters_, ours.n_iters)

    def test_minibatch_training_is_reproducible(self):
        settings = dict(lr=0.02, n_iters=200, batch_size=16, random_state=42)
        first = GradientDescentLR(**settings).fit(self.X, self.y)
        second = GradientDescentLR(**settings).fit(self.X, self.y)

        np.testing.assert_allclose(first.get_theta(), second.get_theta())
        self.assertGreater(first.score(self.X, self.y), 0.999)

    def test_l2_matches_ridge_objective(self):
        strength = 0.2
        ours = GradientDescentLR(lr=0.05, n_iters=2000, l2=strength).fit(
            self.X, self.y
        )
        # sklearn minimizes ||error||^2 + alpha * ||coef||^2, while this model
        # minimizes mean(error^2) + l2 * ||coef||^2.
        reference = Ridge(alpha=len(self.y) * strength).fit(self.X, self.y)

        np.testing.assert_allclose(ours.coef_, reference.coef_, atol=1e-5)


class ValidationTests(unittest.TestCase):
    def test_predict_before_fit_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            NormalEquationLR().predict([[1]])

    def test_rejects_mismatched_sample_counts(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            NormalEquationLR().fit([[1], [2]], [1])

    def test_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            GradientDescentLR().fit([[1], [np.nan]], [1, 2])


if __name__ == "__main__":
    unittest.main()
