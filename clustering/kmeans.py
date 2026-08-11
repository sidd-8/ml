"""K-Means clustering implemented from scratch with NumPy."""

import numpy as np


class KMeans:
    """Partition samples into clusters by minimizing within-cluster variance.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    init : {"k-means++", "random"} or array-like, default="k-means++"
        Center initialization strategy or explicit initial centers.
    n_init : int, default=10
        Independent initializations; the lowest-inertia solution is retained.
        Explicit initial centers always run once.
    max_iter : int, default=300
        Maximum center-update iterations per initialization.
    tol : float, default=1e-4
        Stop when the Euclidean center shift is at most this value.
    random_state : int or None, default=None
        Seed controlling initialization.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
    ):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if isinstance(init, str) and init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++', 'random', or initial centers")
        if not isinstance(n_init, int) or n_init <= 0:
            raise ValueError("n_init must be a positive integer")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not np.isscalar(tol) or not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a non-negative finite number")

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
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

    @staticmethod
    def _squared_distances(X, centers):
        differences = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.sum(differences**2, axis=2)

    def _validate_explicit_centers(self, X):
        centers = np.asarray(self.init, dtype=float)
        expected = (self.n_clusters, X.shape[1])
        if centers.shape != expected:
            raise ValueError(f"initial centers must have shape {expected}")
        if not np.all(np.isfinite(centers)):
            raise ValueError("initial centers must contain only finite values")
        return centers.copy()

    def _initialize_centers(self, X, rng):
        if not isinstance(self.init, str):
            return self._validate_explicit_centers(X)
        if self.init == "random":
            indices = rng.choice(len(X), size=self.n_clusters, replace=False)
            return X[indices].copy()

        first_index = int(rng.integers(len(X)))
        selected = [first_index]
        centers = [X[first_index].copy()]
        closest_squared = self._squared_distances(X, np.asarray(centers))[:, 0]
        for _ in range(1, self.n_clusters):
            total = np.sum(closest_squared)
            if total <= np.finfo(float).eps:
                remaining = np.setdiff1d(np.arange(len(X)), np.asarray(selected))
                next_index = int(rng.choice(remaining))
            else:
                next_index = int(rng.choice(len(X), p=closest_squared / total))
            selected.append(next_index)
            centers.append(X[next_index].copy())
            distance_to_new = self._squared_distances(
                X, np.asarray(centers[-1:])
            )[:, 0]
            closest_squared = np.minimum(closest_squared, distance_to_new)
        return np.asarray(centers)

    def _updated_centers(self, X, labels, squared_distances, previous_centers):
        centers = np.empty_like(previous_centers)
        empty_clusters = []
        for cluster in range(self.n_clusters):
            members = X[labels == cluster]
            if len(members):
                centers[cluster] = np.mean(members, axis=0)
            else:
                empty_clusters.append(cluster)

        if empty_clusters:
            assigned_distance = squared_distances[np.arange(len(X)), labels]
            farthest_samples = np.argsort(assigned_distance, kind="stable")[::-1]
            used = set()
            for cluster in empty_clusters:
                sample_index = next(
                    index for index in farthest_samples if int(index) not in used
                )
                used.add(int(sample_index))
                centers[cluster] = X[sample_index]
        return centers

    def _run_single(self, X, rng):
        centers = self._initialize_centers(X, rng)
        previous_labels = None
        n_iter = 0
        for iteration in range(1, self.max_iter + 1):
            squared_distances = self._squared_distances(X, centers)
            labels = np.argmin(squared_distances, axis=1)
            new_centers = self._updated_centers(
                X, labels, squared_distances, centers
            )
            center_shift = float(np.linalg.norm(new_centers - centers))
            labels_unchanged = previous_labels is not None and np.array_equal(
                labels, previous_labels
            )
            centers = new_centers
            previous_labels = labels
            n_iter = iteration
            if labels_unchanged or center_shift <= self.tol:
                break

        final_squared = self._squared_distances(X, centers)
        final_labels = np.argmin(final_squared, axis=1)
        inertia = float(np.sum(final_squared[np.arange(len(X)), final_labels]))
        return centers, final_labels, inertia, n_iter

    def fit(self, X):
        """Compute cluster centers and retain the best initialization."""
        X = self._validate_X(X)
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of samples")
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        runs = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for _ in range(runs):
            result = self._run_single(X, rng)
            if best is None or result[2] < best[2]:
                best = result

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def _check_is_fitted(self):
        if self.cluster_centers_ is None:
            raise ValueError("No model yet. Call fit() first.")

    def _validate_query(self, X):
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return X

    def predict(self, X):
        """Return the nearest fitted center for each sample."""
        X = self._validate_query(X)
        return np.argmin(self._squared_distances(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        """Fit centers and return training cluster labels."""
        return self.fit(X).labels_.copy()

    def transform(self, X):
        """Return Euclidean distances from samples to every cluster center."""
        X = self._validate_query(X)
        return np.sqrt(self._squared_distances(X, self.cluster_centers_))

    def score(self, X):
        """Return negative inertia, following scikit-learn's convention."""
        distances = self.transform(X)
        return -float(np.sum(np.min(distances, axis=1) ** 2))
