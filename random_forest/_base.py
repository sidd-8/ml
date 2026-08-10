"""Shared bootstrap, validation, and aggregation for random forests."""

import numpy as np


class BaseRandomForest:
    """Common configuration and fitting utilities for random forests."""

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        max_samples=None,
        random_state=None,
    ):
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(bootstrap, bool) or not isinstance(oob_score, bool):
            raise ValueError("bootstrap and oob_score must be booleans")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        if not bootstrap and max_samples is not None:
            raise ValueError("max_samples requires bootstrap=True")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_samples = max_samples
        self.random_state = random_state

        self.estimators_ = []
        self.estimators_samples_ = []
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.n_samples_fit_ = None
        self.oob_score_ = None
        self.oob_counts_ = None

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    @classmethod
    def _validate_fit_data(cls, X, y, *, numeric_target):
        X = cls._validate_X(X)
        y = np.asarray(y, dtype=float if numeric_target else None)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if numeric_target and not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        if not numeric_target and np.issubdtype(y.dtype, np.number):
            if not np.all(np.isfinite(y)):
                raise ValueError("y must contain only finite labels")
        return X, y

    def _resolve_sample_size(self, n_samples):
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, int) and not isinstance(self.max_samples, bool):
            if not 1 <= self.max_samples <= n_samples:
                raise ValueError("integer max_samples must be between 1 and n_samples")
            return self.max_samples
        if isinstance(self.max_samples, float):
            if not 0 < self.max_samples <= 1:
                raise ValueError("float max_samples must be in (0, 1]")
            return max(1, int(round(self.max_samples * n_samples)))
        raise ValueError("max_samples must be None, an integer, or a float")

    def _fit_estimators(self, X, y, tree_factory):
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        sample_size = self._resolve_sample_size(len(X))
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.estimators_samples_ = []

        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            if self.bootstrap:
                sample_indices = rng.integers(0, len(X), size=sample_size)
            else:
                sample_indices = np.arange(len(X))
            tree = tree_factory(seed)
            tree.fit(X[sample_indices], y[sample_indices])
            self.estimators_.append(tree)
            self.estimators_samples_.append(sample_indices)

        importances = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        total = np.sum(importances)
        self.feature_importances_ = importances / total if total > 0 else importances

    def _check_is_fitted(self):
        if not self.estimators_:
            raise ValueError("No model yet. Call fit() first.")

    def _validate_query(self, X):
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X

    def _oob_indices(self, sample_indices):
        selected = np.zeros(self.n_samples_fit_, dtype=bool)
        selected[np.unique(sample_indices)] = True
        return np.flatnonzero(~selected)
