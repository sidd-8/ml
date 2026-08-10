"""Validation shared by boosting estimators."""

import numpy as np


def validate_X(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


def validate_fit_data(X, y, *, numeric_target):
    X = validate_X(X)
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
