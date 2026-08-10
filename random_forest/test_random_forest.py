import unittest

import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.ensemble import RandomForestClassifier as SklearnForestClassifier
from sklearn.ensemble import RandomForestRegressor as SklearnForestRegressor
from sklearn.model_selection import train_test_split

from random_forest import RandomForestClassifier, RandomForestRegressor


class RandomForestClassifierTests(unittest.TestCase):
    def test_without_bootstrap_matches_a_single_deterministic_tree(self):
        X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array(["low", "low", "low", "high", "high", "high"])
        ours = RandomForestClassifier(
            n_estimators=3, bootstrap=False, max_features=None, max_depth=2
        ).fit(X, y)
        reference = SklearnForestClassifier(
            n_estimators=3,
            bootstrap=False,
            max_features=None,
            max_depth=2,
            random_state=0,
        ).fit(X, y)
        query = [[0.5], [2.5], [4.5]]
        np.testing.assert_array_equal(ours.predict(query), reference.predict(query))
        np.testing.assert_allclose(ours.predict_proba(query), reference.predict_proba(query))

    def test_wine_performance_is_comparable_to_sklearn(self):
        X, y = load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=9, stratify=y
        )
        ours = RandomForestClassifier(
            n_estimators=25, max_depth=6, random_state=42
        ).fit(X_train, y_train)
        reference = SklearnForestClassifier(
            n_estimators=25, max_depth=6, random_state=42
        ).fit(X_train, y_train)
        self.assertGreaterEqual(ours.score(X_test, y_test), 0.9)
        self.assertAlmostEqual(
            ours.score(X_test, y_test), reference.score(X_test, y_test), delta=0.08
        )
        np.testing.assert_allclose(ours.predict_proba(X_test).sum(axis=1), 1.0)

    def test_oob_probabilities_and_score(self):
        X, y = load_wine(return_X_y=True)
        model = RandomForestClassifier(
            n_estimators=30, oob_score=True, random_state=4
        ).fit(X, y)
        valid = model.oob_counts_ > 0
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(
            model.oob_decision_function_[valid].sum(axis=1), 1.0
        )
        self.assertGreater(model.oob_score_, 0.85)

    def test_bootstrap_missing_class_is_aligned(self):
        X = np.arange(12, dtype=float).reshape(-1, 1)
        y = np.array(["common"] * 11 + ["rare"])
        model = RandomForestClassifier(n_estimators=15, random_state=1).fit(X, y)
        self.assertTrue(any(len(tree.classes_) == 1 for tree in model.estimators_))
        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities.shape, (12, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


class RandomForestRegressorTests(unittest.TestCase):
    def test_diabetes_performance_is_comparable_to_sklearn(self):
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        ours = RandomForestRegressor(
            n_estimators=25,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
        ).fit(X_train, y_train)
        reference = SklearnForestRegressor(
            n_estimators=25,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
        ).fit(X_train, y_train)
        self.assertGreater(ours.score(X_test, y_test), 0.35)
        self.assertAlmostEqual(
            ours.score(X_test, y_test), reference.score(X_test, y_test), delta=0.08
        )

    def test_oob_predictions_and_feature_importances(self):
        X, y = load_diabetes(return_X_y=True)
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=6,
            min_samples_leaf=3,
            oob_score=True,
            random_state=7,
        ).fit(X, y)
        self.assertTrue(np.all(model.oob_counts_ > 0))
        self.assertTrue(np.all(np.isfinite(model.oob_prediction_)))
        self.assertGreater(model.oob_score_, 0.3)
        np.testing.assert_allclose(np.sum(model.feature_importances_), 1.0)


class RandomForestReproducibilityTests(unittest.TestCase):
    def test_seeds_and_bootstrap_samples_are_reproducible(self):
        X, y = load_wine(return_X_y=True)
        settings = dict(n_estimators=8, max_samples=0.7, random_state=12)
        first = RandomForestClassifier(**settings).fit(X, y)
        second = RandomForestClassifier(**settings).fit(X, y)
        for first_samples, second_samples in zip(
            first.estimators_samples_, second.estimators_samples_
        ):
            np.testing.assert_array_equal(first_samples, second_samples)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))


class RandomForestValidationTests(unittest.TestCase):
    def test_predict_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            RandomForestClassifier().predict([[0]])

    def test_rejects_invalid_bootstrap_configuration(self):
        with self.assertRaisesRegex(ValueError, "requires bootstrap"):
            RandomForestRegressor(bootstrap=False, oob_score=True)
        with self.assertRaisesRegex(ValueError, "requires bootstrap"):
            RandomForestClassifier(bootstrap=False, max_samples=0.5)

    def test_rejects_invalid_sample_size(self):
        with self.assertRaisesRegex(ValueError, "max_samples"):
            RandomForestRegressor(max_samples=2.0).fit([[0], [1]], [0, 1])


if __name__ == "__main__":
    unittest.main()
