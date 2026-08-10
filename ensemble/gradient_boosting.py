"""Gradient boosting for squared-error regression and binary classification."""

import numpy as np

from decision_tree import DecisionTreeRegressor
from metrics import accuracy_score, mean_squared_error, r2_score

from ._validation import validate_X, validate_fit_data


class BaseGradientBoosting:
    """Shared configuration and tree aggregation for gradient boosting."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        *,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        subsample=1.0,
        min_impurity_decrease=0.0,
        n_iter_no_change=None,
        tol=1e-4,
        random_state=None,
    ):
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if not np.isscalar(learning_rate) or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        if not np.isscalar(subsample) or not np.isfinite(subsample):
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if n_iter_no_change is not None and (
            not isinstance(n_iter_no_change, int) or n_iter_no_change <= 0
        ):
            raise ValueError("n_iter_no_change must be None or a positive integer")
        if tol < 0:
            raise ValueError("tol must be non-negative")

        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = float(subsample)
        self.min_impurity_decrease = min_impurity_decrease
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

        self.estimators_ = []
        self.train_score_ = np.array([])
        self.n_estimators_ = 0
        self.n_features_in_ = None
        self.feature_importances_ = None

    @property
    def loss_history_(self):
        return self.train_score_

    def _make_tree(self, seed):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=seed,
        )

    def _sample_indices(self, rng, n_samples):
        size = max(1, int(self.subsample * n_samples))
        if size == n_samples:
            return np.arange(n_samples)
        return rng.choice(n_samples, size=size, replace=False)

    def _finalize_fit(self, estimators, losses):
        self.estimators_ = estimators
        self.train_score_ = np.asarray(losses)
        self.n_estimators_ = len(estimators)
        importances = np.mean(
            [tree.feature_importances_ for tree in estimators], axis=0
        )
        total = np.sum(importances)
        self.feature_importances_ = importances / total if total > 0 else importances

    def _should_stop(self, loss, best_loss, rounds_without_improvement):
        if self.n_iter_no_change is None:
            return False, min(best_loss, loss), rounds_without_improvement
        if best_loss - loss > self.tol:
            return False, loss, 0
        rounds_without_improvement += 1
        return (
            rounds_without_improvement >= self.n_iter_no_change,
            best_loss,
            rounds_without_improvement,
        )

    def _check_is_fitted(self):
        if not self.estimators_:
            raise ValueError("No model yet. Call fit() first.")

    def _validate_query(self, X):
        self._check_is_fitted()
        X = validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X


class GradientBoostingRegressor(BaseGradientBoosting):
    """Gradient boosting regressor minimizing mean squared error."""

    def __init__(self, n_estimators=100, learning_rate=0.1, **kwargs):
        super().__init__(
            n_estimators=n_estimators, learning_rate=learning_rate, **kwargs
        )
        self.init_ = None

    def fit(self, X, y):
        """Fit regression trees to successive prediction residuals."""
        X, y = validate_fit_data(X, y, numeric_target=True)
        self.n_features_in_ = X.shape[1]
        self.init_ = float(np.mean(y))
        prediction = np.full(len(y), self.init_)
        rng = np.random.default_rng(self.random_state)
        estimators = []
        losses = []
        best_loss = float("inf")
        rounds_without_improvement = 0

        for _ in range(self.n_estimators):
            residual = y - prediction
            indices = self._sample_indices(rng, len(y))
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = self._make_tree(seed).fit(X[indices], residual[indices])
            prediction += self.learning_rate * tree.predict(X)
            loss = mean_squared_error(y, prediction)
            estimators.append(tree)
            losses.append(loss)
            stop, best_loss, rounds_without_improvement = self._should_stop(
                loss, best_loss, rounds_without_improvement
            )
            if stop:
                break

        self._finalize_fit(estimators, losses)
        return self

    def predict(self, X):
        X = self._validate_query(X)
        prediction = np.full(len(X), self.init_)
        for tree in self.estimators_:
            prediction += self.learning_rate * tree.predict(X)
        return prediction

    def staged_predict(self, X):
        X = self._validate_query(X)
        prediction = np.full(len(X), self.init_)
        for tree in self.estimators_:
            prediction += self.learning_rate * tree.predict(X)
            yield prediction.copy()

    def staged_score(self, X, y):
        for prediction in self.staged_predict(X):
            yield r2_score(y, prediction)

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class GradientBoostingClassifier(BaseGradientBoosting):
    """Binary gradient boosting classifier minimizing logistic loss."""

    def __init__(self, n_estimators=100, learning_rate=0.1, **kwargs):
        super().__init__(
            n_estimators=n_estimators, learning_rate=learning_rate, **kwargs
        )
        self.classes_ = None
        self.init_ = None

    @staticmethod
    def _sigmoid(scores):
        result = np.empty_like(scores, dtype=float)
        positive = scores >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
        exponent = np.exp(scores[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        return result

    @staticmethod
    def _nodes_by_id(root):
        nodes = {}
        stack = [root]
        while stack:
            node = stack.pop()
            nodes[node.node_id] = node
            if not node.is_leaf:
                stack.extend((node.left, node.right))
        return nodes

    def fit(self, X, y):
        """Fit Newton-adjusted regression trees to logistic residuals."""
        X, y = validate_fit_data(X, y, numeric_target=False)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier requires exactly two classes")
        target = encoded.astype(float)
        self.n_features_in_ = X.shape[1]
        positive_rate = np.clip(np.mean(target), 1e-15, 1.0 - 1e-15)
        self.init_ = float(np.log(positive_rate / (1.0 - positive_rate)))
        scores = np.full(len(y), self.init_)
        rng = np.random.default_rng(self.random_state)
        estimators = []
        losses = []
        best_loss = float("inf")
        rounds_without_improvement = 0

        for _ in range(self.n_estimators):
            probability = self._sigmoid(scores)
            residual = target - probability
            indices = self._sample_indices(rng, len(y))
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = self._make_tree(seed).fit(X[indices], residual[indices])

            leaf_ids = tree.apply(X[indices])
            nodes = self._nodes_by_id(tree.tree_)
            for leaf_id in np.unique(leaf_ids):
                mask = leaf_ids == leaf_id
                numerator = np.sum(residual[indices][mask])
                denominator = np.sum(
                    probability[indices][mask] * (1.0 - probability[indices][mask])
                )
                nodes[int(leaf_id)].value = float(
                    numerator / max(denominator, np.finfo(float).eps)
                )

            scores += self.learning_rate * tree.predict(X)
            loss = float(np.mean(np.logaddexp(0.0, scores) - target * scores))
            estimators.append(tree)
            losses.append(loss)
            stop, best_loss, rounds_without_improvement = self._should_stop(
                loss, best_loss, rounds_without_improvement
            )
            if stop:
                break

        self._finalize_fit(estimators, losses)
        return self

    def decision_function(self, X):
        X = self._validate_query(X)
        scores = np.full(len(X), self.init_)
        for tree in self.estimators_:
            scores += self.learning_rate * tree.predict(X)
        return scores

    def predict_proba(self, X):
        positive = self._sigmoid(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X):
        positive = self.predict_proba(X)[:, 1] >= 0.5
        return self.classes_[positive.astype(int)]

    def staged_predict_proba(self, X):
        X = self._validate_query(X)
        scores = np.full(len(X), self.init_)
        for tree in self.estimators_:
            scores += self.learning_rate * tree.predict(X)
            positive = self._sigmoid(scores)
            yield np.column_stack((1.0 - positive, positive))

    def staged_predict(self, X):
        for probabilities in self.staged_predict_proba(X):
            yield self.classes_[(probabilities[:, 1] >= 0.5).astype(int)]

    def staged_score(self, X, y):
        for prediction in self.staged_predict(X):
            yield accuracy_score(y, prediction)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
