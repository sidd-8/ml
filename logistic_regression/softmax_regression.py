"""Multiclass softmax regression implemented with NumPy."""

import numpy as np

from metrics import accuracy_score


class SoftmaxRegression:
    """Multinomial logistic regression trained by mini-batch gradient descent."""

    def __init__(self, learning_rate=0.1, n_iters=1000, *, l2=0.0,
                 tolerance=0.0, batch_size=None, shuffle=True, random_state=None):
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(n_iters, int) or n_iters <= 0:
            raise ValueError("n_iters must be a positive integer")
        if l2 < 0 or tolerance < 0:
            raise ValueError("l2 and tolerance must be non-negative")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("batch_size must be None or a positive integer")
        self.learning_rate, self.n_iters = float(learning_rate), n_iters
        self.l2, self.tolerance, self.batch_size = float(l2), float(tolerance), batch_size
        self.shuffle, self.random_state = shuffle, random_state
        self.classes_ = self.coef_ = self.intercept_ = None
        self.loss_history_ = []
        self.n_iter_ = 0
        self.converged_ = False
        self.n_features_in_ = None

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

    @staticmethod
    def _softmax(scores):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.sum(exponent, axis=1, keepdims=True)

    def _loss(self, X, targets):
        probabilities = self._softmax(X @ self.coef_.T + self.intercept_)
        data_loss = -np.mean(np.sum(targets * np.log(np.maximum(probabilities, np.finfo(float).tiny)), axis=1))
        return float(data_loss + 0.5 * self.l2 * np.sum(self.coef_**2))

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per sample")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least two classes")
        self.n_features_in_ = X.shape[1]
        targets = np.eye(len(self.classes_))[encoded]
        self.coef_ = np.zeros((len(self.classes_), X.shape[1]))
        self.intercept_ = np.zeros(len(self.classes_))
        self.loss_history_ = []
        self.converged_ = False
        rng = np.random.default_rng(self.random_state)
        batch_size = min(self.batch_size or len(X), len(X))
        previous = self._loss(X, targets)
        for epoch in range(self.n_iters):
            indices = rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                probabilities = self._softmax(X[batch] @ self.coef_.T + self.intercept_)
                error = probabilities - targets[batch]
                self.coef_ -= self.learning_rate * (error.T @ X[batch] / len(batch) + self.l2 * self.coef_)
                self.intercept_ -= self.learning_rate * np.mean(error, axis=0)
            loss = self._loss(X, targets)
            self.loss_history_.append(loss)
            self.n_iter_ = epoch + 1
            if self.tolerance and abs(previous - loss) <= self.tolerance:
                self.converged_ = True
                break
            previous = loss
        return self

    def decision_function(self, X):
        if self.coef_ is None:
            raise ValueError("No model yet. Call fit() first.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        return self._softmax(self.decision_function(X))

    def predict_log_proba(self, X):
        return np.log(np.maximum(self.predict_proba(X), np.finfo(float).tiny))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
