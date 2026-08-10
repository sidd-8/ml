"""Multiclass AdaBoost using weighted decision stumps and SAMME."""

from dataclasses import dataclass

import numpy as np

from metrics import accuracy_score

from ._validation import validate_X, validate_fit_data


@dataclass
class DecisionStump:
    """One-level classification tree fitted with weighted class counts."""

    feature_index: object = None
    threshold: object = None
    left_class: int = 0
    right_class: int = 0
    n_features_in_: object = None

    def fit(self, X, y, sample_weight, n_classes):
        self.n_features_in_ = X.shape[1]
        total_class_weight = np.bincount(
            y, weights=sample_weight, minlength=n_classes
        )
        majority = int(np.argmax(total_class_weight))
        self.left_class = majority
        self.right_class = majority
        best_correct_weight = float(np.max(total_class_weight))

        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            labels = y[order]
            weights = sample_weight[order]
            left_counts = np.zeros(n_classes, dtype=float)
            for position in range(len(X) - 1):
                left_counts[labels[position]] += weights[position]
                if values[position] == values[position + 1]:
                    continue
                right_counts = total_class_weight - left_counts
                left_class = int(np.argmax(left_counts))
                right_class = int(np.argmax(right_counts))
                correct_weight = left_counts[left_class] + right_counts[right_class]
                if correct_weight > best_correct_weight + 1e-15:
                    best_correct_weight = float(correct_weight)
                    self.feature_index = feature
                    self.threshold = float(
                        (values[position] + values[position + 1]) / 2.0
                    )
                    self.left_class = left_class
                    self.right_class = right_class
        return self

    def predict(self, X):
        if self.feature_index is None:
            return np.full(len(X), self.left_class, dtype=int)
        return np.where(
            X[:, self.feature_index] <= self.threshold,
            self.left_class,
            self.right_class,
        )

    @property
    def feature_importances_(self):
        importances = np.zeros(self.n_features_in_, dtype=float)
        if self.feature_index is not None:
            importances[self.feature_index] = 1.0
        return importances


class AdaBoostClassifier:
    """SAMME AdaBoost classifier using weighted decision stumps."""

    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if not np.isscalar(learning_rate) or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.random_state = random_state

        self.estimators_ = []
        self.estimator_weights_ = np.array([])
        self.estimator_errors_ = np.array([])
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.n_estimators_ = 0

    def fit(self, X, y):
        """Sequentially fit stumps to increasingly emphasized mistakes."""
        X, y = validate_fit_data(X, y, numeric_target=False)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("y must contain at least two classes")
        self.n_features_in_ = X.shape[1]
        sample_weight = np.full(len(X), 1.0 / len(X))
        estimators = []
        estimator_weights = []
        estimator_errors = []
        random_limit = 1.0 - 1.0 / self.n_classes_

        for _ in range(self.n_estimators):
            stump = DecisionStump().fit(
                X, encoded, sample_weight, self.n_classes_
            )
            prediction = stump.predict(X)
            incorrect = prediction != encoded
            error = float(np.sum(sample_weight[incorrect]))
            if error >= random_limit - 1e-15:
                if not estimators:
                    raise ValueError("A decision stump is no better than random guessing")
                break

            clipped_error = np.clip(error, np.finfo(float).eps, 1.0)
            estimator_weight = self.learning_rate * (
                np.log((1.0 - clipped_error) / clipped_error)
                + np.log(self.n_classes_ - 1.0)
            )
            estimators.append(stump)
            estimator_weights.append(estimator_weight)
            estimator_errors.append(error)
            if error <= np.finfo(float).eps:
                break

            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight /= np.sum(sample_weight)

        self.estimators_ = estimators
        self.estimator_weights_ = np.asarray(estimator_weights)
        self.estimator_errors_ = np.asarray(estimator_errors)
        self.n_estimators_ = len(estimators)
        weighted_importance = np.sum(
            [
                weight * estimator.feature_importances_
                for estimator, weight in zip(estimators, estimator_weights)
            ],
            axis=0,
        )
        total = np.sum(weighted_importance)
        self.feature_importances_ = (
            weighted_importance / total if total > 0 else weighted_importance
        )
        return self

    def _check_is_fitted(self):
        if not self.estimators_:
            raise ValueError("No model yet. Call fit() first.")

    def _validate_query(self, X):
        self._check_is_fitted()
        X = validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X

    def _scores(self, X, n_estimators=None):
        count = self.n_estimators_ if n_estimators is None else n_estimators
        scores = np.zeros((len(X), self.n_classes_), dtype=float)
        rows = np.arange(len(X))
        for stump, weight in zip(
            self.estimators_[:count], self.estimator_weights_[:count]
        ):
            scores[rows, stump.predict(X)] += weight
        return scores

    def decision_function(self, X):
        """Return accumulated class votes before softmax normalization."""
        X = self._validate_query(X)
        return self._scores(X)

    def predict_proba(self, X):
        """Transform accumulated SAMME votes into class probabilities."""
        scores = self.decision_function(X) / max(1, self.n_classes_ - 1)
        scores -= np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def predict(self, X):
        X = self._validate_query(X)
        return self.classes_[np.argmax(self._scores(X), axis=1)]

    def staged_predict(self, X):
        """Yield predictions after each additional weak learner."""
        X = self._validate_query(X)
        for count in range(1, self.n_estimators_ + 1):
            yield self.classes_[np.argmax(self._scores(X, count), axis=1)]

    def staged_score(self, X, y):
        for prediction in self.staged_predict(X):
            yield accuracy_score(y, prediction)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
