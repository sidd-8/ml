"""Feature-scaling algorithms implemented from scratch with NumPy."""

import numpy as np

from ._base import BaseScaler


class StandardScaler(BaseScaler):
    """Standardize features by removing the mean and scaling to unit variance.

    Population variance (``ddof=0``) is used, matching the convention used by
    scikit-learn. Constant features receive a scale of one and transform to zero
    when centering is enabled.
    """

    def __init__(self, with_mean=True, with_std=True):
        super().__init__()
        if not isinstance(with_mean, bool) or not isinstance(with_std, bool):
            raise ValueError("with_mean and with_std must be booleans")
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None
        self.scale_ = None

    def fit(self, X):
        """Calculate per-feature mean and variance."""
        X = self._validate_X(X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0) if self.with_mean else None
        self.var_ = np.var(X, axis=0) if self.with_std else None
        if self.with_std:
            standard_deviation = np.sqrt(self.var_)
            self.scale_ = np.where(standard_deviation == 0, 1.0, standard_deviation)
        else:
            self.scale_ = None
        return self

    def transform(self, X):
        """Standardize X using the fitted training statistics."""
        X = self._validate_transform_data(X).copy()
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Undo standardization and return values in the original scale."""
        X = self._validate_transform_data(X).copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        return X


class MinMaxScaler(BaseScaler):
    """Scale each feature into a configurable range.

    Parameters
    ----------
    feature_range : tuple of float, default=(0, 1)
        Desired lower and upper bounds.
    clip : bool, default=False
        Clip transformed values from unseen data to ``feature_range``.
    """

    def __init__(self, feature_range=(0, 1), clip=False):
        super().__init__()
        if not isinstance(feature_range, (tuple, list)) or len(feature_range) != 2:
            raise ValueError("feature_range must contain exactly two values")
        lower, upper = feature_range
        if not np.isscalar(lower) or not np.isscalar(upper):
            raise ValueError("feature_range bounds must be numeric scalars")
        lower, upper = float(lower), float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("feature_range requires finite lower < upper")
        if not isinstance(clip, bool):
            raise ValueError("clip must be a boolean")

        self.feature_range = (lower, upper)
        self.clip = clip
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None

    def fit(self, X):
        """Calculate per-feature minima, maxima, and scaling factors."""
        X = self._validate_X(X)
        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        lower, upper = self.feature_range
        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        self.scale_ = (upper - lower) / safe_range
        self.min_ = lower - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        """Scale X using the fitted feature bounds."""
        X = self._validate_transform_data(X)
        transformed = X * self.scale_ + self.min_
        if self.clip:
            transformed = np.clip(transformed, *self.feature_range)
        return transformed

    def inverse_transform(self, X):
        """Undo min-max scaling and return values in the original scale."""
        X = self._validate_transform_data(X)
        return (X - self.min_) / self.scale_
