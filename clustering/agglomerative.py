"""Agglomerative hierarchical clustering implemented with NumPy."""

import numpy as np


class AgglomerativeClustering:
    """Bottom-up clustering with single, complete, average, or Ward linkage."""

    def __init__(self, n_clusters=2, *, linkage="ward"):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if linkage not in ("single", "complete", "average", "ward"):
            raise ValueError("linkage must be 'single', 'complete', 'average', or 'ward'")
        self.n_clusters, self.linkage = n_clusters, linkage
        self.labels_ = self.children_ = self.distances_ = None
        self.n_leaves_ = None
        self.n_features_in_ = None

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _distance(self, X, left, right):
        a, b = X[np.asarray(left)], X[np.asarray(right)]
        if self.linkage == "ward":
            delta = np.mean(a, axis=0) - np.mean(b, axis=0)
            return float(np.sqrt(2 * len(a) * len(b) / (len(a) + len(b)) * np.dot(delta, delta)))
        pairwise = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
        if self.linkage == "single":
            return float(np.min(pairwise))
        if self.linkage == "complete":
            return float(np.max(pairwise))
        return float(np.mean(pairwise))

    def fit(self, X):
        X = self._validate_X(X)
        n = len(X)
        if self.n_clusters > n:
            raise ValueError("n_clusters cannot exceed the number of samples")
        self.n_features_in_, self.n_leaves_ = X.shape[1], n
        active = {i: [i] for i in range(n)}
        children, distances = [], []
        next_id = n
        labels_at_target = None
        while len(active) > 1:
            keys = sorted(active)
            best = None
            for position, left_id in enumerate(keys[:-1]):
                for right_id in keys[position + 1:]:
                    distance = self._distance(X, active[left_id], active[right_id])
                    candidate = (distance, left_id, right_id)
                    if best is None or candidate < best:
                        best = candidate
            distance, left_id, right_id = best
            children.append([left_id, right_id])
            distances.append(distance)
            active[next_id] = active.pop(left_id) + active.pop(right_id)
            next_id += 1
            if len(active) == self.n_clusters:
                labels_at_target = [members.copy() for _, members in sorted(active.items())]
        if labels_at_target is None:  # one requested cluster
            labels_at_target = [list(range(n))]
        labels = np.empty(n, dtype=int)
        for label, members in enumerate(labels_at_target):
            labels[members] = label
        self.labels_ = labels
        self.children_ = np.asarray(children, dtype=int).reshape(-1, 2)
        self.distances_ = np.asarray(distances)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_.copy()
