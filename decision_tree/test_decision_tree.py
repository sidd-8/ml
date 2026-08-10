import unittest

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnTreeRegressor

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class DecisionTreeClassifierTests(unittest.TestCase):
    def test_simple_tree_matches_sklearn(self):
        X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array(["low", "low", "low", "high", "high", "high"])
        ours = DecisionTreeClassifier(max_depth=2).fit(X, y)
        reference = SklearnTreeClassifier(max_depth=2, random_state=0).fit(X, y)

        query = np.array([[0.5], [2.5], [4.5]])
        np.testing.assert_array_equal(ours.predict(query), reference.predict(query))
        np.testing.assert_allclose(ours.predict_proba(query), reference.predict_proba(query))
        self.assertEqual(ours.get_depth(), reference.get_depth())
        self.assertEqual(ours.get_n_leaves(), reference.get_n_leaves())

    def test_gini_and_entropy_perform_like_sklearn_on_iris(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=4, stratify=y
        )
        for criterion in ("gini", "entropy"):
            ours = DecisionTreeClassifier(
                criterion=criterion, max_depth=4, min_samples_leaf=2
            ).fit(X_train, y_train)
            reference = SklearnTreeClassifier(
                criterion=criterion,
                max_depth=4,
                min_samples_leaf=2,
                random_state=0,
            ).fit(X_train, y_train)
            self.assertGreaterEqual(ours.score(X_test, y_test), 0.9)
            self.assertAlmostEqual(
                ours.score(X_test, y_test), reference.score(X_test, y_test), delta=0.05
            )

    def test_feature_importances_and_tree_inspection(self):
        X = np.array([[0, 10], [1, 10], [2, 10], [3, 10]])
        y = np.array([0, 0, 1, 1])
        model = DecisionTreeClassifier().fit(X, y)

        np.testing.assert_allclose(model.feature_importances_, [1, 0])
        self.assertEqual(len(model.apply(X)), len(X))
        text = model.export_text(feature_names=["signal", "constant"])
        self.assertIn("if signal <=", text)
        self.assertIn("predict:", text)

    def test_random_feature_subsampling_is_reproducible(self):
        X, y = load_iris(return_X_y=True)
        settings = dict(max_depth=4, max_features="sqrt", random_state=42)
        first = DecisionTreeClassifier(**settings).fit(X, y)
        second = DecisionTreeClassifier(**settings).fit(X, y)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        np.testing.assert_allclose(first.feature_importances_, second.feature_importances_)


class DecisionTreeRegressorTests(unittest.TestCase):
    def test_simple_regression_matches_sklearn(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 4, 4, 9, 9, 16, 16], dtype=float)
        ours = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2).fit(X, y)
        reference = SklearnTreeRegressor(
            max_depth=3, min_samples_leaf=2, random_state=0
        ).fit(X, y)
        query = np.arange(0.5, 9, 1).reshape(-1, 1)
        np.testing.assert_allclose(ours.predict(query), reference.predict(query))
        np.testing.assert_allclose(
            ours.feature_importances_, reference.feature_importances_
        )

    def test_diabetes_performance_is_comparable(self):
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        ours = DecisionTreeRegressor(max_depth=4, min_samples_leaf=8).fit(
            X_train, y_train
        )
        reference = SklearnTreeRegressor(
            max_depth=4, min_samples_leaf=8, random_state=0
        ).fit(X_train, y_train)
        self.assertAlmostEqual(
            ours.score(X_test, y_test), reference.score(X_test, y_test), delta=0.02
        )

    def test_depth_and_leaf_constraints(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.sin(X[:, 0])
        model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3).fit(X, y)
        self.assertLessEqual(model.get_depth(), 2)


class DecisionTreeValidationTests(unittest.TestCase):
    def test_predict_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            DecisionTreeClassifier().predict([[0]])

    def test_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "criterion"):
            DecisionTreeClassifier(criterion="invalid")
        with self.assertRaisesRegex(ValueError, "max_depth"):
            DecisionTreeRegressor(max_depth=0)
        with self.assertRaisesRegex(ValueError, "max_features"):
            DecisionTreeClassifier(max_features="bad").fit([[0], [1]], [0, 1])

    def test_rejects_non_finite_data(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            DecisionTreeRegressor().fit([[0], [np.nan]], [0, 1])


if __name__ == "__main__":
    unittest.main()
