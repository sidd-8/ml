"""Gaussian, Multinomial, and Bernoulli Naive Bayes classifiers."""

import numpy as np

from ._base import BaseNaiveBayes


class GaussianNB(BaseNaiveBayes):
    """Naive Bayes with a Gaussian distribution for every feature and class."""

    def __init__(self, priors=None, var_smoothing=1e-9):
        super().__init__(priors=priors)
        if not np.isscalar(var_smoothing) or not np.isfinite(var_smoothing):
            raise ValueError("var_smoothing must be a positive finite number")
        if var_smoothing <= 0:
            raise ValueError("var_smoothing must be a positive finite number")
        self.var_smoothing = float(var_smoothing)
        self.theta_ = None
        self.var_ = None
        self.epsilon_ = None

    def fit(self, X, y):
        """Estimate class priors and per-class Gaussian parameters."""
        X, _, classes, encoded = self._validate_fit_data(X, y)
        self.n_features_in_ = X.shape[1]
        self._set_class_statistics(classes, encoded)
        maximum_variance = max(np.max(np.var(X, axis=0)), np.finfo(float).eps)
        self.epsilon_ = self.var_smoothing * maximum_variance

        self.theta_ = np.empty((len(classes), self.n_features_in_))
        self.var_ = np.empty_like(self.theta_)
        for class_index in range(len(classes)):
            class_samples = X[encoded == class_index]
            self.theta_[class_index] = np.mean(class_samples, axis=0)
            self.var_[class_index] = np.var(class_samples, axis=0) + self.epsilon_
        return self

    def _joint_log_likelihood(self, X):
        log_prior = np.log(self.class_prior_)
        log_normalizer = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_), axis=1)
        squared_distance = -0.5 * np.sum(
            (X[:, np.newaxis, :] - self.theta_[np.newaxis, :, :]) ** 2
            / self.var_[np.newaxis, :, :],
            axis=2,
        )
        return log_prior + log_normalizer + squared_distance


class MultinomialNB(BaseNaiveBayes):
    """Naive Bayes for non-negative feature counts or frequencies."""

    def __init__(self, alpha=1.0, priors=None):
        super().__init__(priors=priors)
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive finite number")
        self.alpha = float(alpha)
        self.feature_count_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """Estimate smoothed feature probabilities for each class."""
        X, _, classes, encoded = self._validate_fit_data(X, y)
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")
        self.n_features_in_ = X.shape[1]
        self._set_class_statistics(classes, encoded)

        self.feature_count_ = np.vstack(
            [np.sum(X[encoded == index], axis=0) for index in range(len(classes))]
        )
        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(
            np.sum(smoothed, axis=1, keepdims=True)
        )
        return self

    def _joint_log_likelihood(self, X):
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")
        return X @ self.feature_log_prob_.T + np.log(self.class_prior_)


class BernoulliNB(BaseNaiveBayes):
    """Naive Bayes for binary feature occurrence indicators."""

    def __init__(self, alpha=1.0, binarize=0.0, priors=None):
        super().__init__(priors=priors)
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive finite number")
        if binarize is not None and (
            not np.isscalar(binarize) or not np.isfinite(binarize)
        ):
            raise ValueError("binarize must be None or a finite number")
        self.alpha = float(alpha)
        self.binarize = None if binarize is None else float(binarize)
        self.feature_count_ = None
        self.feature_log_prob_ = None

    def _binarize(self, X):
        if self.binarize is not None:
            return (X > self.binarize).astype(float)
        if not np.all(np.isin(X, (0, 1))):
            raise ValueError("BernoulliNB with binarize=None requires binary features")
        return X

    def fit(self, X, y):
        """Estimate smoothed feature-presence probabilities for each class."""
        X, _, classes, encoded = self._validate_fit_data(X, y)
        X = self._binarize(X)
        self.n_features_in_ = X.shape[1]
        self._set_class_statistics(classes, encoded)

        self.feature_count_ = np.vstack(
            [np.sum(X[encoded == index], axis=0) for index in range(len(classes))]
        )
        smoothed_count = self.feature_count_ + self.alpha
        smoothed_class_count = self.class_count_[:, np.newaxis] + 2.0 * self.alpha
        self.feature_log_prob_ = np.log(smoothed_count) - np.log(smoothed_class_count)
        return self

    def _joint_log_likelihood(self, X):
        X = self._binarize(X)
        negative_log_probability = np.log1p(-np.exp(self.feature_log_prob_))
        return (
            X @ (self.feature_log_prob_ - negative_log_probability).T
            + np.sum(negative_log_probability, axis=1)
            + np.log(self.class_prior_)
        )
