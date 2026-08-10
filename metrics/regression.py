"""Regression metrics implemented with NumPy."""

import numpy as np

from ._validation import validate_pair


def mean_squared_error(y_true, y_pred):
    """Return the average squared prediction error."""
    y_true, y_pred = validate_pair(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """Return the square root of mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    """Return the average absolute prediction error."""
    y_true, y_pred = validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    """Return the coefficient of determination.

    For a constant target this follows scikit-learn's finite convention: a
    perfect prediction scores 1.0 and any imperfect prediction scores 0.0.
    """
    y_true, y_pred = validate_pair(y_true, y_pred)
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0
    return float(1.0 - residual_sum / total_sum)


def adjusted_r2_score(y_true, y_pred, n_features):
    """Return R-squared adjusted for the number of input features."""
    y_true, y_pred = validate_pair(y_true, y_pred)
    if not isinstance(n_features, int) or n_features < 0:
        raise ValueError("n_features must be a non-negative integer")
    denominator = len(y_true) - n_features - 1
    if denominator <= 0:
        raise ValueError("adjusted R-squared requires n_samples > n_features + 1")
    score = r2_score(y_true, y_pred)
    return float(1.0 - (1.0 - score) * (len(y_true) - 1) / denominator)


def mean_absolute_percentage_error(y_true, y_pred, *, zero_policy="raise"):
    """Return mean absolute relative error (1.0 means 100 percent).

    ``zero_policy='raise'`` prevents silently undefined percentages.
    ``zero_policy='ignore'`` excludes samples whose true target is zero.
    """
    y_true, y_pred = validate_pair(y_true, y_pred)
    if zero_policy not in ("raise", "ignore"):
        raise ValueError("zero_policy must be 'raise' or 'ignore'")

    nonzero = y_true != 0
    if not np.all(nonzero) and zero_policy == "raise":
        raise ZeroDivisionError("MAPE is undefined when y_true contains zero")
    if not np.any(nonzero):
        raise ZeroDivisionError("MAPE has no non-zero targets to evaluate")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
