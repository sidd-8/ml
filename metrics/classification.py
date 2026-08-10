"""Binary classification metrics implemented with NumPy."""

import numpy as np

from ._validation import (
    safe_divide,
    validate_binary_targets,
    validate_pair,
    validate_zero_division,
)


def accuracy_score(y_true, y_pred):
    """Return the fraction of exactly matching labels."""
    y_true, y_pred = validate_pair(y_true, y_pred, numeric=False)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, *, labels=None):
    """Return a matrix whose rows are true and columns are predicted labels."""
    y_true, y_pred = validate_pair(y_true, y_pred, numeric=False)
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels must be a non-empty 1D sequence")
        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates")
        if not np.all(np.isin(y_true, labels)) or not np.all(np.isin(y_pred, labels)):
            raise ValueError("y_true and y_pred contain labels not listed in labels")

    result = np.zeros((len(labels), len(labels)), dtype=int)
    positions = {label: index for index, label in enumerate(labels.tolist())}
    for truth, prediction in zip(y_true, y_pred):
        result[positions[truth], positions[prediction]] += 1
    return result


def _binary_counts(y_true, y_pred, positive_label):
    y_true, y_pred = validate_pair(y_true, y_pred, numeric=False)
    validate_binary_targets(y_true, positive_label=positive_label)
    known_labels = np.unique(y_true)
    if not np.all(np.isin(y_pred, known_labels)):
        raise ValueError("y_pred contains labels not present in y_true")
    true_positive = np.sum((y_true == positive_label) & (y_pred == positive_label))
    false_positive = np.sum((y_true != positive_label) & (y_pred == positive_label))
    false_negative = np.sum((y_true == positive_label) & (y_pred != positive_label))
    return int(true_positive), int(false_positive), int(false_negative)


def precision_score(y_true, y_pred, *, positive_label=1, zero_division=0):
    """Return TP / (TP + FP) for the selected positive class."""
    tp, fp, _ = _binary_counts(y_true, y_pred, positive_label)
    return safe_divide(tp, tp + fp, zero_division=zero_division)


def recall_score(y_true, y_pred, *, positive_label=1, zero_division=0):
    """Return TP / (TP + FN) for the selected positive class."""
    tp, _, fn = _binary_counts(y_true, y_pred, positive_label)
    return safe_divide(tp, tp + fn, zero_division=zero_division)


def f1_score(y_true, y_pred, *, positive_label=1, zero_division=0):
    """Return the harmonic mean of precision and recall."""
    validate_zero_division(zero_division)
    tp, fp, fn = _binary_counts(y_true, y_pred, positive_label)
    return safe_divide(2 * tp, 2 * tp + fp + fn, zero_division=zero_division)


def log_loss(y_true, y_probability, *, positive_label=1, epsilon=1e-15):
    """Return binary cross-entropy from positive-class probabilities."""
    y_true = validate_binary_targets(y_true, positive_label=positive_label)
    probabilities = np.asarray(y_probability, dtype=float)
    if probabilities.ndim != 1 or probabilities.shape != y_true.shape:
        raise ValueError("y_probability must be 1D and match y_true")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_probability must contain only finite values")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be between 0 and 1")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")

    targets = (y_true == positive_label).astype(float)
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(
            targets * np.log(probabilities)
            + (1.0 - targets) * np.log(1.0 - probabilities)
        )
    )


def roc_curve(y_true, y_score, *, positive_label=1):
    """Return false-positive rates, true-positive rates, and thresholds."""
    y_true = validate_binary_targets(y_true, positive_label=positive_label)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim != 1 or y_score.shape != y_true.shape:
        raise ValueError("y_score must be 1D and match y_true")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score must contain only finite values")

    positive = y_true == positive_label
    n_positive = np.sum(positive)
    n_negative = len(y_true) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC requires both positive and negative samples")

    order = np.argsort(y_score, kind="stable")[::-1]
    sorted_scores = y_score[order]
    sorted_positive = positive[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(y_score) - 1]
    true_positives = np.cumsum(sorted_positive)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives

    tpr = np.r_[0.0, true_positives / n_positive]
    fpr = np.r_[0.0, false_positives / n_negative]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]
    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score, *, positive_label=1):
    """Return area under the receiver operating characteristic curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score, positive_label=positive_label)
    widths = np.diff(fpr)
    return float(np.sum(widths * (tpr[:-1] + tpr[1:]) / 2.0))


def precision_recall_curve(y_true, y_score, *, positive_label=1):
    """Return precision, recall, and increasing decision thresholds."""
    y_true = validate_binary_targets(y_true, positive_label=positive_label)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim != 1 or y_score.shape != y_true.shape:
        raise ValueError("y_score must be 1D and match y_true")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score must contain only finite values")

    positive = y_true == positive_label
    order = np.argsort(y_score, kind="stable")[::-1]
    sorted_scores = y_score[order]
    sorted_positive = positive[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(y_score) - 1]
    true_positives = np.cumsum(sorted_positive)[threshold_indices]
    predicted_positives = threshold_indices + 1
    precision = true_positives / predicted_positives
    recall = true_positives / np.sum(positive)
    # Match the conventional sklearn ordering: increasing thresholds and
    # decreasing recall, followed by the no-positive-predictions endpoint.
    return (
        np.r_[precision[::-1], 1.0],
        np.r_[recall[::-1], 0.0],
        sorted_scores[threshold_indices][::-1],
    )
