"""Binary logistic regression implemented from scratch with NumPy."""

import numpy as np

from metrics import accuracy_score


class LogisticRegression:
    """Binary logistic regression trained with gradient descent.

    ``batch_size=None`` uses full-batch gradient descent. A batch size of one
    gives stochastic gradient descent; intermediate values give mini-batch GD.
    The two sorted target labels are exposed through ``classes_`` and probability
    columns follow that same order.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size used by gradient descent.
    n_iters : int, default=1000
        Maximum number of training epochs.
    threshold : float, default=0.5
        Positive-class probability cutoff used by ``predict``.
    l2 : float, default=0.0
        L2 regularization strength. The intercept is not regularized.
    tolerance : float, default=0.0
        Stop when the absolute loss improvement is at most this value. Zero
        disables early stopping.
    batch_size : int or None, default=None
        Samples per gradient update. None uses the entire dataset.
    shuffle : bool, default=True
        Shuffle samples before each epoch.
    random_state : int or None, default=None
        Seed controlling reproducible shuffling.
    class_weight : dict, "balanced", or None, default=None
        Per-class loss weights. ``"balanced"`` uses inverse class frequency.
    """

    def __init__(
        self,
        learning_rate=0.1,
        n_iters=1000,
        threshold=0.5,
        l2=0.0,
        tolerance=0.0,
        batch_size=None,
        shuffle=True,
        random_state=None,
        class_weight=None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(n_iters, int) or n_iters <= 0:
            raise ValueError("n_iters must be a positive integer")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if batch_size is not None and (
            not isinstance(batch_size, int) or batch_size <= 0
        ):
            raise ValueError("batch_size must be None or a positive integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if class_weight is not None and class_weight != "balanced" and not isinstance(
            class_weight, dict
        ):
            raise ValueError("class_weight must be None, 'balanced', or a dictionary")

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.l2 = l2
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.class_weight = class_weight

        self.weights = None
        self.bias = None
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.class_weight_ = None
        self.loss_history_ = []
        self.n_iters_ = 0
        self.converged_ = False

    @property
    def loss_history(self):
        """Backward-compatible alias for the per-epoch objective values."""
        return self.loss_history_

    @staticmethod
    def _sigmoid(z):
        """Compute sigmoid probabilities without overflowing for large values."""
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z)
        positive = z >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1.0 + exp_z)
        return result

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
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite labels")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("y must contain exactly two binary classes")
        return X, y, classes

    def _make_sample_weights(self, y):
        if self.class_weight is None:
            weights = {label: 1.0 for label in self.classes_}
        elif self.class_weight == "balanced":
            weights = {
                label: len(y) / (len(self.classes_) * np.sum(y == label))
                for label in self.classes_
            }
        else:
            missing = [label for label in self.classes_ if label not in self.class_weight]
            if missing:
                raise ValueError("class_weight must provide a weight for every class")
            weights = {label: float(self.class_weight[label]) for label in self.classes_}
            if any(not np.isfinite(value) or value <= 0 for value in weights.values()):
                raise ValueError("class weights must be positive finite numbers")

        self.class_weight_ = weights
        return np.asarray([weights[label] for label in y], dtype=float)

    def _loss_and_gradients(self, X, y_binary, sample_weight):
        scores = X @ self.weights + self.bias
        probabilities = self._sigmoid(scores)
        normalizer = np.sum(sample_weight)
        errors = sample_weight * (probabilities - y_binary)

        weight_gradient = (X.T @ errors) / normalizer + self.l2 * self.weights
        bias_gradient = float(np.sum(errors) / normalizer)
        data_loss = np.sum(
            sample_weight * (np.logaddexp(0.0, scores) - y_binary * scores)
        ) / normalizer
        loss = float(data_loss + 0.5 * self.l2 * np.sum(self.weights**2))
        return loss, weight_gradient, bias_gradient

    def fit(self, X, y):
        """Learn model parameters from features X and any two target labels."""
        X, y, classes = self._validate_fit_data(X, y)
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        y_binary = (y == self.classes_[1]).astype(float)
        sample_weight = self._make_sample_weights(y)
        n_samples = len(y)
        batch_size = min(self.batch_size or n_samples, n_samples)
        rng = np.random.default_rng(self.random_state)

        self.weights = np.zeros(self.n_features_in_, dtype=float)
        self.bias = 0.0
        self.loss_history_ = []
        self.n_iters_ = 0
        self.converged_ = False
        previous_loss, _, _ = self._loss_and_gradients(X, y_binary, sample_weight)

        for epoch in range(self.n_iters):
            indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                _, weight_gradient, bias_gradient = self._loss_and_gradients(
                    X[batch_indices],
                    y_binary[batch_indices],
                    sample_weight[batch_indices],
                )
                self.weights -= self.learning_rate * weight_gradient
                self.bias -= self.learning_rate * bias_gradient

            loss, _, _ = self._loss_and_gradients(X, y_binary, sample_weight)
            if not np.isfinite(loss):
                raise ValueError(
                    "Training diverged; reduce learning_rate or scale the features"
                )
            self.loss_history_.append(loss)
            self.n_iters_ = epoch + 1
            if self.tolerance > 0 and abs(previous_loss - loss) <= self.tolerance:
                self.converged_ = True
                break
            previous_loss = loss

        self.coef_ = self.weights.copy()
        self.intercept_ = float(self.bias)
        return self

    def _check_is_fitted(self):
        if self.weights is None or self.bias is None:
            raise ValueError("No model yet. Call fit() first.")

    def decision_function(self, X):
        """Return unbounded log-odds scores for the positive class."""
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X @ self.weights + self.bias

    def predict_proba(self, X):
        """Return class probabilities in the order given by ``classes_``."""
        positive_probability = self._sigmoid(self.decision_function(X))
        return np.column_stack((1.0 - positive_probability, positive_probability))

    def predict(self, X):
        """Predict labels using the configured positive-class threshold."""
        is_positive = self.predict_proba(X)[:, 1] >= self.threshold
        return self.classes_[is_positive.astype(int)]

    def score(self, X, y):
        """Return mean classification accuracy."""
        y = np.asarray(y)
        predictions = self.predict(X)
        if y.ndim != 1 or y.shape[0] != predictions.shape[0]:
            raise ValueError("y must be 1D and match the number of samples in X")
        return accuracy_score(y, predictions)

    def get_theta(self):
        """Return a copy of [intercept, coefficients]."""
        self._check_is_fitted()
        return np.r_[self.bias, self.weights].copy()
