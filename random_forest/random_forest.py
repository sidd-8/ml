"""Random forest classification and regression from scratch."""

import numpy as np

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from metrics import accuracy_score, r2_score

from ._base import BaseRandomForest


class RandomForestClassifier(BaseRandomForest):
    """Ensemble of randomized classification trees trained on bootstrap samples."""

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        max_samples=None,
        random_state=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            max_samples=max_samples,
            random_state=random_state,
        )
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.classes_ = None
        self.n_classes_ = None
        self.oob_decision_function_ = None

    def fit(self, X, y):
        """Fit randomized trees and optionally calculate out-of-bag accuracy."""
        X, y = self._validate_fit_data(X, y, numeric_target=False)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        def make_tree(seed):
            return DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=seed,
            )

        self._fit_estimators(X, y, make_tree)
        if self.oob_score:
            self._calculate_oob_score(X, y)
        return self

    def _aligned_tree_probabilities(self, tree, X):
        tree_probabilities = tree.predict_proba(X)
        aligned = np.zeros((len(X), self.n_classes_), dtype=float)
        class_positions = np.searchsorted(self.classes_, tree.classes_)
        aligned[:, class_positions] = tree_probabilities
        return aligned

    def _calculate_oob_score(self, X, y):
        totals = np.zeros((len(X), self.n_classes_), dtype=float)
        counts = np.zeros(len(X), dtype=int)
        for tree, sample_indices in zip(self.estimators_, self.estimators_samples_):
            oob_indices = self._oob_indices(sample_indices)
            if len(oob_indices) == 0:
                continue
            totals[oob_indices] += self._aligned_tree_probabilities(tree, X[oob_indices])
            counts[oob_indices] += 1

        valid = counts > 0
        self.oob_counts_ = counts
        self.oob_decision_function_ = np.full_like(totals, np.nan)
        self.oob_decision_function_[valid] = totals[valid] / counts[valid, np.newaxis]
        if np.any(valid):
            predictions = self.classes_[
                np.argmax(self.oob_decision_function_[valid], axis=1)
            ]
            self.oob_score_ = accuracy_score(y[valid], predictions)
        else:
            self.oob_score_ = float("nan")

    def predict_proba(self, X):
        """Average class probabilities across all fitted trees."""
        X = self._validate_query(X)
        probabilities = np.zeros((len(X), self.n_classes_), dtype=float)
        for tree in self.estimators_:
            probabilities += self._aligned_tree_probabilities(tree, X)
        return probabilities / self.n_estimators

    def predict(self, X):
        """Predict the class with the greatest mean forest probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class RandomForestRegressor(BaseRandomForest):
    """Ensemble of randomized regression trees trained on bootstrap samples."""

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        max_samples=None,
        random_state=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            max_samples=max_samples,
            random_state=random_state,
        )
        self.oob_prediction_ = None

    def fit(self, X, y):
        """Fit randomized trees and optionally calculate out-of-bag R-squared."""
        X, y = self._validate_fit_data(X, y, numeric_target=True)

        def make_tree(seed):
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=seed,
            )

        self._fit_estimators(X, y, make_tree)
        if self.oob_score:
            self._calculate_oob_score(X, y)
        return self

    def _calculate_oob_score(self, X, y):
        totals = np.zeros(len(X), dtype=float)
        counts = np.zeros(len(X), dtype=int)
        for tree, sample_indices in zip(self.estimators_, self.estimators_samples_):
            oob_indices = self._oob_indices(sample_indices)
            if len(oob_indices) == 0:
                continue
            totals[oob_indices] += tree.predict(X[oob_indices])
            counts[oob_indices] += 1

        valid = counts > 0
        self.oob_counts_ = counts
        self.oob_prediction_ = np.full(len(X), np.nan)
        self.oob_prediction_[valid] = totals[valid] / counts[valid]
        self.oob_score_ = (
            r2_score(y[valid], self.oob_prediction_[valid])
            if np.any(valid)
            else float("nan")
        )

    def predict(self, X):
        """Average continuous predictions across all fitted trees."""
        X = self._validate_query(X)
        predictions = np.vstack([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        return r2_score(y, self.predict(X))
