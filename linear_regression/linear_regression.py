"""Linear regression algorithms implemented from scratch with NumPy."""

import numpy as np

from metrics import r2_score


class BaseLinearRegression:
    """Shared validation, prediction, and metrics for linear regressors."""

    def __init__(self, fit_intercept=True):
        if not isinstance(fit_intercept, bool):
            raise ValueError("fit_intercept must be a boolean")

        self.fit_intercept = fit_intercept
        self.theta = None
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.loss_history_ = []

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
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        return X, y

    def _design_matrix(self, X):
        if not self.fit_intercept:
            return X
        return np.column_stack((np.ones(X.shape[0]), X))

    def _set_fitted_parameters(self, theta, n_features):
        self.theta = np.asarray(theta, dtype=float)
        self.n_features_in_ = n_features
        if self.fit_intercept:
            self.intercept_ = float(self.theta[0])
            self.coef_ = self.theta[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta.copy()

    def _check_is_fitted(self):
        if self.theta is None:
            raise ValueError("No model yet. Call fit() first.")

    def predict(self, X):
        """Predict continuous target values for X."""
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return self._design_matrix(X) @ self.theta

    def score(self, X, y):
        """Return the coefficient of determination (R-squared)."""
        y = np.asarray(y, dtype=float)
        predictions = self.predict(X)
        if y.ndim != 1 or y.shape[0] != predictions.shape[0]:
            raise ValueError("y must be 1D and match the number of samples in X")

        return r2_score(y, predictions)

    def get_theta(self):
        """Return a copy of [intercept, coefficients] for backwards compatibility."""
        self._check_is_fitted()
        return self.theta.copy()


class NormalEquationLR(BaseLinearRegression):
    """Ordinary least squares solved using a stable least-squares routine."""

    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        self.rank_ = None
        self.singular_values_ = None

    def fit(self, X, y):
        """Fit the least-squares solution, including rank-deficient datasets."""
        X, y = self._validate_fit_data(X, y)
        design = self._design_matrix(X)

        # lstsq avoids explicitly forming X.T @ X, which squares the condition
        # number and is less stable for correlated features.
        theta, _, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
        self._set_fitted_parameters(theta, X.shape[1])
        self.rank_ = int(rank)
        self.singular_values_ = singular_values

        errors = design @ self.theta - y
        self.loss_history_ = [float(np.mean(errors**2))]
        return self


class GradientDescentLR(BaseLinearRegression):
    """Linear regression trained with batch, mini-batch, or stochastic GD.

    ``batch_size=None`` uses full-batch gradient descent. A batch size of one
    gives stochastic gradient descent; intermediate values give mini-batch GD.
    Loss is mean squared error plus ``l2 * sum(coefficients ** 2)``.
    """

    def __init__(
        self,
        lr=0.01,
        n_iters=1000,
        tolerance=0.0,
        l2=0.0,
        batch_size=None,
        shuffle=True,
        random_state=None,
        fit_intercept=True,
    ):
        super().__init__(fit_intercept=fit_intercept)
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(n_iters, int) or n_iters <= 0:
            raise ValueError("n_iters must be a positive integer")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if batch_size is not None and (
            not isinstance(batch_size, int) or batch_size <= 0
        ):
            raise ValueError("batch_size must be None or a positive integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")

        self.lr = lr
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.l2 = l2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_iters_ = 0
        self.converged_ = False

    def _loss(self, design, y):
        errors = design @ self.theta - y
        coefficients = self.theta[1:] if self.fit_intercept else self.theta
        return float(np.mean(errors**2) + self.l2 * np.sum(coefficients**2))

    def fit(self, X, y):
        """Fit model parameters and record one objective value per epoch."""
        X, y = self._validate_fit_data(X, y)
        design = self._design_matrix(X)
        n_samples, n_parameters = design.shape
        batch_size = min(self.batch_size or n_samples, n_samples)
        rng = np.random.default_rng(self.random_state)

        self._set_fitted_parameters(np.zeros(n_parameters), X.shape[1])
        self.loss_history_ = []
        self.n_iters_ = 0
        self.converged_ = False
        previous_loss = self._loss(design, y)

        for epoch in range(self.n_iters):
            indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)

            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = design[batch_indices]
                y_batch = y[batch_indices]
                errors = X_batch @ self.theta - y_batch
                gradient = (2.0 / len(batch_indices)) * (X_batch.T @ errors)

                regularized = self.theta.copy()
                if self.fit_intercept:
                    regularized[0] = 0.0
                gradient += 2.0 * self.l2 * regularized
                self.theta -= self.lr * gradient

            loss = self._loss(design, y)
            if not np.isfinite(loss):
                raise ValueError(
                    "Training diverged; reduce lr or scale the input features"
                )
            self.loss_history_.append(loss)
            self.n_iters_ = epoch + 1

            if self.tolerance > 0 and abs(previous_loss - loss) <= self.tolerance:
                self.converged_ = True
                break
            previous_loss = loss

        # Synchronize the sklearn-style attributes after in-place optimization.
        self._set_fitted_parameters(self.theta, X.shape[1])
        return self

    @property
    def loss_history(self):
        """Backward-compatible alias matching the logistic regression model."""
        return self.loss_history_
