import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBClassifier
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBRegressor
from sklearn.model_selection import train_test_split

from ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


class AdaBoostClassifierTests(unittest.TestCase):
    def test_perfect_stump_stops_after_one_estimator(self):
        X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array(["low", "low", "low", "high", "high", "high"])
        model = AdaBoostClassifier(n_estimators=20).fit(X, y)
        self.assertEqual(model.n_estimators_, 1)
        np.testing.assert_array_equal(model.predict(X), y)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)

    def test_multiclass_iris_performance_is_comparable(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=7, stratify=y
        )
        ours = AdaBoostClassifier(n_estimators=40, learning_rate=0.8).fit(
            X_train, y_train
        )
        reference = SklearnAdaBoostClassifier(
            n_estimators=40, learning_rate=0.8, random_state=0
        ).fit(X_train, y_train)
        self.assertGreaterEqual(ours.score(X_test, y_test), 0.85)
        self.assertAlmostEqual(
            ours.score(X_test, y_test), reference.score(X_test, y_test), delta=0.12
        )
        self.assertEqual(len(list(ours.staged_predict(X_test))), ours.n_estimators_)

    def test_estimator_diagnostics_and_feature_importances(self):
        X, y = load_iris(return_X_y=True)
        model = AdaBoostClassifier(n_estimators=10).fit(X, y)
        self.assertEqual(len(model.estimator_weights_), model.n_estimators_)
        self.assertEqual(len(model.estimator_errors_), model.n_estimators_)
        self.assertTrue(np.all(model.estimator_errors_ < 1 - 1 / 3))
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0)


class GradientBoostingRegressorTests(unittest.TestCase):
    def test_diabetes_performance_is_comparable(self):
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        ours = GradientBoostingRegressor(
            n_estimators=60,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        ).fit(X_train, y_train)
        reference = SklearnGBRegressor(
            n_estimators=60,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        ).fit(X_train, y_train)
        self.assertGreater(ours.score(X_test, y_test), 0.4)
        self.assertAlmostEqual(
            ours.score(X_test, y_test), reference.score(X_test, y_test), delta=0.06
        )
        self.assertLess(ours.train_score_[-1], ours.train_score_[0])

    def test_staged_predictions_end_at_final_prediction(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.sin(X[:, 0])
        model = GradientBoostingRegressor(n_estimators=12, max_depth=2).fit(X, y)
        staged = list(model.staged_predict(X))
        self.assertEqual(len(staged), 12)
        np.testing.assert_allclose(staged[-1], model.predict(X))

    def test_training_early_stopping(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.ones(20)
        model = GradientBoostingRegressor(
            n_estimators=100, n_iter_no_change=3, tol=1e-10
        ).fit(X, y)
        self.assertLess(model.n_estimators_, model.n_estimators)


class GradientBoostingClassifierTests(unittest.TestCase):
    def test_breast_cancer_performance_is_comparable(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        ours = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        ).fit(X_train, y_train)
        reference = SklearnGBClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        ).fit(X_train, y_train)
        self.assertGreater(ours.score(X_test, y_test), 0.9)
        self.assertAlmostEqual(
            ours.score(X_test, y_test), reference.score(X_test, y_test), delta=0.05
        )
        np.testing.assert_allclose(ours.predict_proba(X_test).sum(axis=1), 1.0)
        self.assertLess(ours.train_score_[-1], ours.train_score_[0])

    def test_string_labels_and_staged_probabilities(self):
        X = np.array([[-2], [-1], [1], [2]], dtype=float)
        y = np.array(["no", "no", "yes", "yes"])
        model = GradientBoostingClassifier(n_estimators=10, max_depth=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        staged = list(model.staged_predict_proba(X))
        np.testing.assert_allclose(staged[-1], model.predict_proba(X))

    def test_rejects_multiclass_target(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            GradientBoostingClassifier().fit([[0], [1], [2]], [0, 1, 2])


class BoostingValidationTests(unittest.TestCase):
    def test_predict_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            AdaBoostClassifier().predict([[0]])
        with self.assertRaisesRegex(ValueError, "fit"):
            GradientBoostingRegressor().predict([[0]])

    def test_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            AdaBoostClassifier(n_estimators=0)
        with self.assertRaisesRegex(ValueError, "subsample"):
            GradientBoostingRegressor(subsample=0)
        with self.assertRaisesRegex(ValueError, "n_iter_no_change"):
            GradientBoostingClassifier(n_iter_no_change=0)


if __name__ == "__main__":
    unittest.main()
