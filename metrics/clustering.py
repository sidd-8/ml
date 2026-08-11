"""Metrics for evaluating clustering results."""

import numpy as np


def silhouette_score(X, labels):
    """Return the mean silhouette coefficient using Euclidean distance."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError("labels must be 1D and match the number of samples")
    classes, encoded = np.unique(labels, return_inverse=True)
    if not 2 <= len(classes) < len(X):
        raise ValueError("silhouette_score requires between 2 and n_samples - 1 clusters")

    differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    coefficients = np.zeros(len(X), dtype=float)
    for index in range(len(X)):
        own_cluster = encoded == encoded[index]
        own_size = np.sum(own_cluster)
        if own_size == 1:
            continue
        a = np.sum(distances[index, own_cluster]) / (own_size - 1)
        b = min(
            np.mean(distances[index, encoded == other])
            for other in range(len(classes))
            if other != encoded[index]
        )
        denominator = max(a, b)
        coefficients[index] = (b - a) / denominator if denominator > 0 else 0.0
    return float(np.mean(coefficients))
