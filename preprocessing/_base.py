"""Shared behavior for feature transformers."""

import numpy as np


class BaseScaler:
    """Validation and convenience methods shared by scalers."""

    def __init__(self):
        self.n_features_in_ = None

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

    def _check_is_fitted(self):
        if self.n_features_in_ is None:
            raise ValueError("No scaler yet. Call fit() first.")

    def _validate_transform_data(self, X):
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return X

    def fit_transform(self, X):
        """Fit to X and return the transformed values."""
        return self.fit(X).transform(X)
