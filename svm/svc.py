"""Kernel support-vector classification using simplified SMO."""

import numpy as np

from metrics import accuracy_score


class SVC:
    """C-support vector classifier with linear or radial-basis kernels.

    Multiclass targets are handled with one-vs-rest binary classifiers.
    """

    def __init__(self, C=1.0, *, kernel="rbf", gamma="scale", tol=1e-3,
                 max_iter=1000, max_passes=10, random_state=None):
        if not np.isscalar(C) or not np.isfinite(C) or C <= 0:
            raise ValueError("C must be a positive finite number")
        if kernel not in ("linear", "rbf"):
            raise ValueError("kernel must be 'linear' or 'rbf'")
        if gamma not in ("scale", "auto"):
            if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 0:
                raise ValueError("gamma must be 'scale', 'auto', or a positive number")
        if tol <= 0 or not np.isfinite(tol):
            raise ValueError("tol must be positive and finite")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(max_passes, int) or max_passes <= 0:
            raise ValueError("max_passes must be a positive integer")
        self.C, self.kernel, self.gamma = float(C), kernel, gamma
        self.tol, self.max_iter, self.max_passes = float(tol), max_iter, max_passes
        self.random_state = random_state
        self.classes_ = self.support_ = self.support_vectors_ = None
        self.dual_coef_ = self.intercept_ = self.coef_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.gamma_ = None
        self._models = []
        self._X_fit = None

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _kernel_matrix(self, X, Y):
        if self.kernel == "linear":
            return X @ Y.T
        squared = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
        return np.exp(-self.gamma_ * np.maximum(squared, 0.0))

    def _fit_binary(self, K, target, rng):
        n = len(target)
        alpha = np.zeros(n)
        bias = 0.0
        passes = iterations = 0
        while passes < self.max_passes and iterations < self.max_iter:
            changed = 0
            decision = (alpha * target) @ K + bias
            errors = decision - target
            for i in range(n):
                violates = (target[i] * errors[i] < -self.tol and alpha[i] < self.C) or (target[i] * errors[i] > self.tol and alpha[i] > 0)
                if not violates:
                    continue
                candidates = np.arange(n)
                candidates = candidates[candidates != i]
                # Prefer the largest error difference; random tie-breaking keeps seeded reproducibility.
                differences = np.abs(errors[i] - errors[candidates])
                tied = candidates[np.flatnonzero(differences == np.max(differences))]
                j = int(rng.choice(tied))
                old_i, old_j = alpha[i], alpha[j]
                if target[i] != target[j]:
                    low, high = max(0.0, old_j - old_i), min(self.C, self.C + old_j - old_i)
                else:
                    low, high = max(0.0, old_i + old_j - self.C), min(self.C, old_i + old_j)
                if high - low <= 1e-15:
                    continue
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                alpha[j] = np.clip(old_j - target[j] * (errors[i] - errors[j]) / eta, low, high)
                if abs(alpha[j] - old_j) < 1e-8:
                    alpha[j] = old_j
                    continue
                alpha[i] = old_i + target[i] * target[j] * (old_j - alpha[j])
                b1 = bias - errors[i] - target[i] * (alpha[i] - old_i) * K[i, i] - target[j] * (alpha[j] - old_j) * K[i, j]
                b2 = bias - errors[j] - target[i] * (alpha[i] - old_i) * K[i, j] - target[j] * (alpha[j] - old_j) * K[j, j]
                if 0 < alpha[i] < self.C:
                    bias = b1
                elif 0 < alpha[j] < self.C:
                    bias = b2
                else:
                    bias = (b1 + b2) / 2
                decision = (alpha * target) @ K + bias
                errors = decision - target
                changed += 1
            passes = passes + 1 if changed == 0 else 0
            iterations += 1
        return {"alpha": alpha, "target": target, "bias": float(bias), "n_iter": iterations}

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per sample")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least two classes")
        self.n_features_in_ = X.shape[1]
        variance = float(np.var(X))
        self.gamma_ = (1 / (X.shape[1] * variance) if self.gamma == "scale" and variance > 0
                       else 1 / X.shape[1] if self.gamma in ("scale", "auto") else float(self.gamma))
        self._X_fit = X.copy()
        K = self._kernel_matrix(X, X)
        rng = np.random.default_rng(self.random_state)
        positive_classes = self.classes_[1:] if len(self.classes_) == 2 else self.classes_
        self._models = [self._fit_binary(K, np.where(y == label, 1.0, -1.0), rng) for label in positive_classes]
        self.n_iter_ = np.asarray([model["n_iter"] for model in self._models])
        union = np.unique(np.concatenate([np.flatnonzero(model["alpha"] > 1e-8) for model in self._models]))
        self.support_ = union
        self.support_vectors_ = X[union].copy()
        self.intercept_ = np.asarray([model["bias"] for model in self._models])
        self.dual_coef_ = np.vstack([(model["alpha"] * model["target"])[union] for model in self._models])
        if self.kernel == "linear":
            self.coef_ = np.vstack([(model["alpha"] * model["target"]) @ X for model in self._models])
            if len(self.classes_) == 2:
                self.coef_ = self.coef_[0]
        else:
            self.coef_ = None
        return self

    def decision_function(self, X):
        if not self._models:
            raise ValueError("No model yet. Call fit() first.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        K = self._kernel_matrix(X, self._X_fit)
        scores = np.column_stack([K @ (model["alpha"] * model["target"]) + model["bias"] for model in self._models])
        return scores[:, 0] if len(self.classes_) == 2 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return self.classes_[(scores >= 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
