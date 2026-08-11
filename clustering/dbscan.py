"""Density-Based Spatial Clustering of Applications with Noise."""

from collections import deque

import numpy as np


class DBSCAN:
    """Cluster samples by expanding connected dense neighborhoods.

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples in the same neighborhood.
    min_samples : int, default=5
        Samples required in an ``eps`` neighborhood, including the point itself,
        for that point to be considered a core sample.
    metric : {"euclidean", "manhattan", "minkowski"}, default="euclidean"
        Distance function used to construct neighborhoods.
    p : float, default=2
        Power used by the Minkowski metric.
    """

    def __init__(self, eps=0.5, min_samples=5, *, metric="euclidean", p=2):
        if not np.isscalar(eps) or not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite number")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(
                "metric must be 'euclidean', 'manhattan', or 'minkowski'"
            )
        if not np.isscalar(p) or not np.isfinite(p) or p < 1:
            raise ValueError("p must be a finite number greater than or equal to 1")

        self.eps = float(eps)
        self.min_samples = min_samples
        self.metric = metric
        self.p = float(p)

        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.n_features_in_ = None
        self.n_clusters_ = None

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

    def _pairwise_distances(self, X):
        differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        if self.metric == "euclidean":
            return np.sqrt(np.sum(differences**2, axis=2))
        if self.metric == "manhattan":
            return np.sum(np.abs(differences), axis=2)
        return np.sum(np.abs(differences) ** self.p, axis=2) ** (1.0 / self.p)

    def fit(self, X):
        """Find core samples and expand all density-connected clusters."""
        X = self._validate_X(X)
        self.n_features_in_ = X.shape[1]
        distances = self._pairwise_distances(X)
        neighborhoods = [
            np.flatnonzero(distances[index] <= self.eps)
            for index in range(len(X))
        ]
        core = np.asarray(
            [len(neighbors) >= self.min_samples for neighbors in neighborhoods]
        )
        labels = np.full(len(X), -1, dtype=int)
        visited = np.zeros(len(X), dtype=bool)
        cluster = 0

        for sample in range(len(X)):
            if visited[sample]:
                continue
            visited[sample] = True
            if not core[sample]:
                continue

            labels[sample] = cluster
            queue = deque()
            queued = np.zeros(len(X), dtype=bool)
            for neighbor in neighborhoods[sample]:
                if neighbor != sample:
                    queue.append(int(neighbor))
                    queued[neighbor] = True

            while queue:
                neighbor = queue.popleft()
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if core[neighbor]:
                        for candidate in neighborhoods[neighbor]:
                            if not queued[candidate] and candidate != neighbor:
                                queue.append(int(candidate))
                                queued[candidate] = True
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster
            cluster += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core)
        self.components_ = X[self.core_sample_indices_].copy()
        self.n_clusters_ = cluster
        return self

    def fit_predict(self, X):
        """Fit clusters and return a copy of labels, with noise marked as -1."""
        return self.fit(X).labels_.copy()
