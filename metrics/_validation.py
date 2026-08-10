"""Internal validation helpers shared by metric implementations."""

import numpy as np


def validate_pair(y_true, y_pred, *, numeric=True):
    dtype = float if numeric else None
    y_true = np.asarray(y_true, dtype=dtype)
    y_pred = np.asarray(y_pred, dtype=dtype)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if numeric and (
        not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred))
    ):
        raise ValueError("y_true and y_pred must contain only finite values")
    return y_true, y_pred


def validate_binary_targets(y_true, *, positive_label=1):
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a non-empty 1D array")
    labels = np.unique(y_true)
    if labels.size > 2:
        raise ValueError("binary metrics support at most two classes")
    if positive_label not in labels:
        raise ValueError("positive_label is not present in y_true")
    return y_true


def validate_zero_division(zero_division):
    if zero_division not in (0, 1, "raise"):
        raise ValueError("zero_division must be 0, 1, or 'raise'")


def safe_divide(numerator, denominator, *, zero_division):
    validate_zero_division(zero_division)
    if denominator != 0:
        return float(numerator / denominator)
    if zero_division == "raise":
        raise ZeroDivisionError("metric is undefined because its denominator is zero")
    return float(zero_division)
