"""Decision tree classification and regression from scratch."""

import numpy as np

from metrics import accuracy_score, r2_score

from ._tree import BaseDecisionTree


class DecisionTreeClassifier(BaseDecisionTree):
    """Binary-split classification tree using Gini impurity or entropy."""

    def __init__(
        self,
        criterion="gini",
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        min_impurity_decrease=0.0,
        random_state=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
        )
        self.criterion = criterion
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        """Grow a classification tree from labeled examples."""
        X, y = self._validate_fit_data(X, y, numeric_target=False)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return self._fit_tree(X, encoded)

    def _impurity_from_counts(self, counts):
        total = np.sum(counts)
        if total == 0:
            return 0.0
        probabilities = counts[counts > 0] / total
        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities**2))
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _impurity(self, y):
        return self._impurity_from_counts(
            np.bincount(y, minlength=self.n_classes_).astype(float)
        )

    def _best_feature_split(self, values, y, parent_impurity):
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_y = y[order]
        total_counts = np.bincount(sorted_y, minlength=self.n_classes_).astype(float)
        left_counts = np.zeros(self.n_classes_, dtype=float)
        best_threshold = None
        best_gain = 0.0
        n_samples = len(y)

        for position in range(n_samples - 1):
            left_counts[sorted_y[position]] += 1
            left_size = position + 1
            right_size = n_samples - left_size
            if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                continue
            if sorted_values[position] == sorted_values[position + 1]:
                continue
            right_counts = total_counts - left_counts
            child_impurity = (
                left_size * self._impurity_from_counts(left_counts)
                + right_size * self._impurity_from_counts(right_counts)
            ) / n_samples
            gain = parent_impurity - child_impurity
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_threshold = (sorted_values[position] + sorted_values[position + 1]) / 2
        return best_threshold, best_gain

    def _leaf_distribution(self, y):
        return np.bincount(y, minlength=self.n_classes_).astype(float)

    def _leaf_value(self, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        return int(np.argmax(counts))

    def _format_value(self, value):
        return repr(self.classes_[value])

    def predict_proba(self, X):
        """Return leaf class proportions in ``classes_`` order."""
        X = self._validate_query(X)
        distributions = np.vstack([self._leaf_for_sample(row).distribution for row in X])
        return distributions / np.sum(distributions, axis=1, keepdims=True)

    def predict(self, X):
        """Predict the majority class in each reached leaf."""
        X = self._validate_query(X)
        encoded = [self._leaf_for_sample(row).value for row in X]
        return self.classes_[encoded]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class DecisionTreeRegressor(BaseDecisionTree):
    """Binary-split regression tree minimizing squared error."""

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        min_impurity_decrease=0.0,
        random_state=None,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
        )

    def fit(self, X, y):
        """Grow a regression tree from continuous targets."""
        X, y = self._validate_fit_data(X, y, numeric_target=True)
        return self._fit_tree(X, y)

    def _impurity(self, y):
        return float(np.var(y))

    def _best_feature_split(self, values, y, parent_impurity):
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_y = y[order]
        cumulative_sum = np.cumsum(sorted_y)
        cumulative_square_sum = np.cumsum(sorted_y**2)
        total_sum = cumulative_sum[-1]
        total_square_sum = cumulative_square_sum[-1]
        best_threshold = None
        best_gain = 0.0
        n_samples = len(y)

        for position in range(n_samples - 1):
            left_size = position + 1
            right_size = n_samples - left_size
            if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                continue
            if sorted_values[position] == sorted_values[position + 1]:
                continue
            left_sum = cumulative_sum[position]
            left_squares = cumulative_square_sum[position]
            right_sum = total_sum - left_sum
            right_squares = total_square_sum - left_squares
            left_sse = left_squares - left_sum**2 / left_size
            right_sse = right_squares - right_sum**2 / right_size
            child_impurity = (left_sse + right_sse) / n_samples
            gain = parent_impurity - child_impurity
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_threshold = (sorted_values[position] + sorted_values[position + 1]) / 2
        return best_threshold, best_gain

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _format_value(self, value):
        return f"{value:.6g}"

    def predict(self, X):
        """Predict the mean training target in each reached leaf."""
        X = self._validate_query(X)
        return np.asarray([self._leaf_for_sample(row).value for row in X])

    def score(self, X, y):
        return r2_score(y, self.predict(X))
