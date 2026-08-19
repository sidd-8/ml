"""Principal component analysis implemented from scratch with NumPy."""

import numpy as np


class PCA:
    """Project centered data onto directions of maximum variance.

    ``n_components`` may be an integer, a fraction of explained variance in
    ``(0, 1)``, or ``None`` to retain every available component.
    """

    def __init__(self, n_components=None, *, whiten=False):
        if n_components is not None:
            if isinstance(n_components, bool) or not np.isscalar(n_components):
                raise ValueError("n_components must be None, an integer, or a fraction")
            if isinstance(n_components, (int, np.integer)):
                if n_components <= 0:
                    raise ValueError("n_components must be positive")
            elif not 0 < float(n_components) < 1:
                raise ValueError("fractional n_components must be in (0, 1)")
        if not isinstance(whiten, bool):
            raise ValueError("whiten must be a boolean")
        self.n_components = n_components
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.noise_variance_ = None
        self.n_components_ = None
        self.n_features_in_ = None
        self.n_samples_ = None

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X):
        X = self._validate_X(X)
        self.n_samples_, self.n_features_in_ = X.shape
        maximum = min(X.shape)
        if isinstance(self.n_components, (int, np.integer)) and self.n_components > maximum:
            raise ValueError("n_components cannot exceed min(n_samples, n_features)")
        self.mean_ = np.mean(X, axis=0)
        _, singular_values, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # SVD signs are arbitrary; make the largest loading positive for reproducibility.
        largest = np.argmax(np.abs(vt), axis=1)
        signs = np.where(vt[np.arange(len(vt)), largest] < 0, -1.0, 1.0)
        vt *= signs[:, None]
        all_variance = singular_values**2 / max(self.n_samples_ - 1, 1)
        total_variance = np.sum(np.var(X, axis=0, ddof=1)) if self.n_samples_ > 1 else 0.0
        ratios = all_variance / total_variance if total_variance > 0 else np.zeros_like(all_variance)
        if self.n_components is None:
            count = maximum
        elif isinstance(self.n_components, (int, np.integer)):
            count = int(self.n_components)
        else:
            count = int(np.searchsorted(np.cumsum(ratios), float(self.n_components)) + 1)
        self.n_components_ = count
        self.components_ = vt[:count]
        self.singular_values_ = singular_values[:count]
        self.explained_variance_ = all_variance[:count]
        self.explained_variance_ratio_ = ratios[:count]
        discarded = all_variance[count:]
        self.noise_variance_ = float(np.mean(discarded)) if len(discarded) else 0.0
        return self

    def _validate_query(self, X):
        if self.components_ is None:
            raise ValueError("No model yet. Call fit() first.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return X

    def transform(self, X):
        projected = (self._validate_query(X) - self.mean_) @ self.components_.T
        if self.whiten:
            scale = np.sqrt(np.maximum(self.explained_variance_, np.finfo(float).eps))
            projected = projected / scale
        return projected

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if self.components_ is None:
            raise ValueError("No model yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_components_ or not np.all(np.isfinite(X)):
            raise ValueError("X must be finite with one column per retained component")
        if self.whiten:
            X = X * np.sqrt(np.maximum(self.explained_variance_, np.finfo(float).eps))
        return X @ self.components_ + self.mean_
