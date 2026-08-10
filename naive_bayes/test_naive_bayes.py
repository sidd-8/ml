import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

from naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


class GaussianNBTests(unittest.TestCase):
    def test_multiclass_predictions_and_probabilities_match_sklearn(self):
        X, y = load_iris(return_X_y=True)
        ours = GaussianNB().fit(X, y)
        reference = SklearnGaussianNB().fit(X, y)

        np.testing.assert_array_equal(ours.predict(X), reference.predict(X))
        np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X))
        np.testing.assert_allclose(ours.theta_, reference.theta_)
        np.testing.assert_allclose(ours.var_, reference.var_)

    def test_supports_string_labels_and_custom_priors(self):
        X = [[0], [0.2], [2], [2.2]]
        y = ["cold", "cold", "hot", "hot"]
        model = GaussianNB(priors=[0.25, 0.75]).fit(X, y)
        np.testing.assert_array_equal(model.classes_, ["cold", "hot"])
        np.testing.assert_allclose(model.class_prior_, [0.25, 0.75])

    def test_constant_features_still_produce_finite_probabilities(self):
        model = GaussianNB().fit([[1], [1], [1], [1]], [0, 0, 1, 1])
        probabilities = model.predict_proba([[1]])
        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(probabilities, [[0.5, 0.5]])


class MultinomialNBTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[2, 1, 0], [3, 0, 1], [0, 1, 4], [0, 2, 3]])
        self.y = np.array(["sports", "sports", "tech", "tech"])
        self.query = np.array([[1, 0, 0], [0, 0, 2]])

    def test_matches_sklearn(self):
        ours = MultinomialNB(alpha=0.5).fit(self.X, self.y)
        reference = SklearnMultinomialNB(alpha=0.5).fit(self.X, self.y)
        np.testing.assert_allclose(ours.feature_count_, reference.feature_count_)
        np.testing.assert_allclose(ours.feature_log_prob_, reference.feature_log_prob_)
        np.testing.assert_allclose(
            ours.predict_log_proba(self.query), reference.predict_log_proba(self.query)
        )
        np.testing.assert_array_equal(ours.predict(self.query), reference.predict(self.query))

    def test_rejects_negative_features(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MultinomialNB().fit([[1, -1], [0, 2]], [0, 1])


class BernoulliNBTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0]])
        self.y = np.array([0, 0, 1, 1])
        self.query = np.array([[0, 2, -1], [3, 0, 1]])

    def test_binarization_matches_sklearn(self):
        ours = BernoulliNB(alpha=0.5, binarize=0).fit(self.X, self.y)
        reference = SklearnBernoulliNB(alpha=0.5, binarize=0).fit(self.X, self.y)
        np.testing.assert_allclose(ours.feature_count_, reference.feature_count_)
        np.testing.assert_allclose(ours.feature_log_prob_, reference.feature_log_prob_)
        np.testing.assert_allclose(
            ours.predict_proba(self.query), reference.predict_proba(self.query)
        )

    def test_none_binarization_requires_binary_data(self):
        with self.assertRaisesRegex(ValueError, "binary features"):
            BernoulliNB(binarize=None).fit([[0, 2], [1, 0]], [0, 1])


class NaiveBayesValidationTests(unittest.TestCase):
    def test_probabilities_are_normalized(self):
        model = GaussianNB().fit([[0], [1], [4], [5]], [0, 0, 1, 1])
        probabilities = model.predict_proba([[0.5], [4.5]])
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
        np.testing.assert_allclose(np.exp(model.predict_log_proba([[0.5]])), probabilities[:1])

    def test_predict_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            GaussianNB().predict([[0]])

    def test_rejects_invalid_priors(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            GaussianNB(priors=[0.2, 0.2]).fit([[0], [1]], [0, 1])

    def test_rejects_one_class(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            GaussianNB().fit([[0], [1]], [0, 0])


if __name__ == "__main__":
    unittest.main()
