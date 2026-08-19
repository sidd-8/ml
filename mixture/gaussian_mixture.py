"""Gaussian mixture model fitted by expectation-maximization."""

import numpy as np


class GaussianMixture:
    """Mixture of full- or diagonal-covariance Gaussian distributions."""

    def __init__(self, n_components=1, *, covariance_type="full", tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, random_state=None):
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        if covariance_type not in ("full", "diag"):
            raise ValueError("covariance_type must be 'full' or 'diag'")
        if tol < 0 or not np.isfinite(tol):
            raise ValueError("tol must be non-negative and finite")
        if reg_covar <= 0 or not np.isfinite(reg_covar):
            raise ValueError("reg_covar must be positive and finite")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(n_init, int) or n_init <= 0:
            raise ValueError("n_init must be a positive integer")
        self.n_components, self.covariance_type = n_components, covariance_type
        self.tol, self.reg_covar, self.max_iter = float(tol), float(reg_covar), max_iter
        self.n_init, self.random_state = n_init, random_state
        self.weights_ = self.means_ = self.covariances_ = None
        self.lower_bound_ = None
        self.n_iter_ = 0
        self.converged_ = False
        self.n_features_in_ = None

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    @staticmethod
    def _logsumexp(values, axis=1):
        maximum = np.max(values, axis=axis, keepdims=True)
        return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)), axis=axis)

    def _initialize(self, X, rng):
        indices = rng.choice(len(X), self.n_components, replace=False)
        means = X[indices].copy()
        weights = np.full(self.n_components, 1 / self.n_components)
        base = np.cov(X, rowvar=False, bias=True)
        if np.ndim(base) == 0:
            base = np.array([[float(base)]])
        base = base + self.reg_covar * np.eye(X.shape[1])
        covariances = (np.tile(base, (self.n_components, 1, 1)) if self.covariance_type == "full"
                       else np.tile(np.diag(base), (self.n_components, 1)))
        return weights, means, covariances

    def _estimate_log_gaussian(self, X, means, covariances):
        result = np.empty((len(X), self.n_components))
        d = X.shape[1]
        for k in range(self.n_components):
            diff = X - means[k]
            if self.covariance_type == "full":
                sign, logdet = np.linalg.slogdet(covariances[k])
                precision = np.linalg.inv(covariances[k])
                quadratic = np.einsum("ij,jk,ik->i", diff, precision, diff)
            else:
                logdet = np.sum(np.log(covariances[k]))
                quadratic = np.sum(diff**2 / covariances[k], axis=1)
            result[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + quadratic)
        return result

    def _e_step(self, X, weights, means, covariances):
        weighted = self._estimate_log_gaussian(X, means, covariances) + np.log(np.maximum(weights, np.finfo(float).tiny))
        normalizer = self._logsumexp(weighted)
        return float(np.mean(normalizer)), np.exp(weighted - normalizer[:, None])

    def _m_step(self, X, responsibilities):
        counts = responsibilities.sum(axis=0) + 10 * np.finfo(float).eps
        weights = counts / len(X)
        means = responsibilities.T @ X / counts[:, None]
        if self.covariance_type == "full":
            covariance = []
            for k in range(self.n_components):
                diff = X - means[k]
                value = (responsibilities[:, k, None] * diff).T @ diff / counts[k]
                covariance.append(value + self.reg_covar * np.eye(X.shape[1]))
        else:
            covariance = []
            for k in range(self.n_components):
                diff = X - means[k]
                covariance.append(np.sum(responsibilities[:, k, None] * diff**2, axis=0) / counts[k] + self.reg_covar)
        return weights, means, np.asarray(covariance)

    def fit(self, X):
        X = self._validate_X(X)
        if self.n_components > len(X):
            raise ValueError("n_components cannot exceed the number of samples")
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            weights, means, covariances = self._initialize(X, rng)
            previous = -np.inf
            converged = False
            for iteration in range(1, self.max_iter + 1):
                lower, responsibilities = self._e_step(X, weights, means, covariances)
                weights, means, covariances = self._m_step(X, responsibilities)
                if abs(lower - previous) <= self.tol:
                    converged = True
                    break
                previous = lower
            lower, _ = self._e_step(X, weights, means, covariances)
            candidate = (lower, weights, means, covariances, iteration, converged)
            if best is None or lower > best[0]:
                best = candidate
        self.lower_bound_, self.weights_, self.means_, self.covariances_, self.n_iter_, self.converged_ = best
        return self

    def _validate_query(self, X):
        if self.means_ is None:
            raise ValueError("No model yet. Call fit() first.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return X

    def score_samples(self, X):
        X = self._validate_query(X)
        weighted = self._estimate_log_gaussian(X, self.means_, self.covariances_) + np.log(self.weights_)
        return self._logsumexp(weighted)

    def score(self, X):
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X):
        X = self._validate_query(X)
        _, responsibilities = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return responsibilities

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def sample(self, n_samples=1, random_state=None):
        if self.means_ is None:
            raise ValueError("No model yet. Call fit() first.")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        rng = np.random.default_rng(random_state)
        labels = rng.choice(self.n_components, n_samples, p=self.weights_)
        samples = np.empty((n_samples, self.n_features_in_))
        for k in range(self.n_components):
            mask = labels == k
            if np.any(mask):
                covariance = self.covariances_[k] if self.covariance_type == "full" else np.diag(self.covariances_[k])
                samples[mask] = rng.multivariate_normal(self.means_[k], covariance, mask.sum())
        return samples, labels
