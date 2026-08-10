"""Shared probability and validation logic for Naive Bayes models."""

import numpy as np

from metrics import accuracy_score


class BaseNaiveBayes:
    """Base class for generative classifiers using Bayes' rule."""

    def __init__(self, priors=None):
        self.priors = priors
        self.classes_ = None
        self.class_count_ = None
        self.class_prior_ = None
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

    @classmethod
    def _validate_fit_data(cls, X, y):
        X = cls._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(y) != len(X):
            raise ValueError("X and y must contain the same number of samples")
        if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite labels")
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("y must contain at least two classes")
        return X, y, classes, encoded

    def _set_class_statistics(self, classes, encoded):
        self.classes_ = classes
        self.class_count_ = np.bincount(encoded, minlength=len(classes)).astype(float)
        if self.priors is None:
            self.class_prior_ = self.class_count_ / np.sum(self.class_count_)
            return

        priors = np.asarray(self.priors, dtype=float)
        if priors.ndim != 1 or len(priors) != len(classes):
            raise ValueError("priors must provide one value for each class")
        if not np.all(np.isfinite(priors)) or np.any(priors <= 0):
            raise ValueError("priors must contain positive finite values")
        if not np.isclose(np.sum(priors), 1.0):
            raise ValueError("priors must sum to 1")
        self.class_prior_ = priors.copy()

    def _check_is_fitted(self):
        if self.classes_ is None:
            raise ValueError("No model yet. Call fit() first.")

    def _validate_query(self, X):
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X

    @staticmethod
    def _logsumexp(values, axis=1):
        maximum = np.max(values, axis=axis, keepdims=True)
        return maximum + np.log(
            np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
        )

    def _joint_log_likelihood(self, X):
        raise NotImplementedError

    def predict_log_proba(self, X):
        """Return normalized log-probabilities in ``classes_`` order."""
        X = self._validate_query(X)
        joint = self._joint_log_likelihood(X)
        return joint - self._logsumexp(joint)

    def predict_proba(self, X):
        """Return class probabilities in ``classes_`` order."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """Return the class with maximum posterior probability."""
        X = self._validate_query(X)
        joint = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint, axis=1)]

    def score(self, X, y):
        """Return mean classification accuracy."""
        return accuracy_score(y, self.predict(X))
