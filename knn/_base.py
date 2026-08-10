"""Shared nearest-neighbor search and validation."""

import numpy as np


class BaseKNN:
    """Common implementation for KNN classification and regression."""

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", p=2):
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(
                "metric must be 'euclidean', 'manhattan', or 'minkowski'"
            )
        if not np.isscalar(p) or not np.isfinite(p) or p < 1:
            raise ValueError("p must be a finite number greater than or equal to 1")

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = float(p)
        self.X_train_ = None
        self.y_train_ = None
        self.n_features_in_ = None
        self.n_samples_fit_ = None

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
        if len(y) != len(X):
            raise ValueError("X and y must contain the same number of samples")
        if numeric_target and not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        if not numeric_target and np.issubdtype(y.dtype, np.number):
            if not np.all(np.isfinite(y)):
                raise ValueError("y must contain only finite labels")
        return X, y

    def _fit(self, X, y, *, numeric_target):
        X, y = self._validate_fit_data(X, y, numeric_target=numeric_target)
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the number of training samples")
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        return self

    def _check_is_fitted(self):
        if self.X_train_ is None:
            raise ValueError("No model yet. Call fit() first.")

    def _validate_query(self, X):
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X

    def _pairwise_distances(self, X):
        differences = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        if self.metric == "euclidean":
            return np.sqrt(np.sum(differences**2, axis=2))
        if self.metric == "manhattan":
            return np.sum(np.abs(differences), axis=2)
        return np.sum(np.abs(differences) ** self.p, axis=2) ** (1.0 / self.p)

    def kneighbors(self, X, *, return_distance=True):
        """Return the closest training-sample indices for each query sample."""
        X = self._validate_query(X)
        pairwise = self._pairwise_distances(X)
        # Stable sorting makes equal-distance behavior deterministic.
        indices = np.argsort(pairwise, axis=1, kind="stable")[:, : self.n_neighbors]
        distances = np.take_along_axis(pairwise, indices, axis=1)
        if return_distance:
            return distances, indices
        return indices

    def _get_neighbor_weights(self, distances):
        if self.weights == "uniform":
            return np.ones_like(distances)

        result = np.zeros_like(distances)
        zero_distance = distances == 0
        rows_with_exact_match = np.any(zero_distance, axis=1)
        result[rows_with_exact_match] = zero_distance[rows_with_exact_match]
        result[~rows_with_exact_match] = 1.0 / distances[~rows_with_exact_match]
        return result
