"""Multilayer perceptrons trained with backpropagation and Adam."""

import numpy as np

from metrics import accuracy_score, r2_score


class BaseMLP:
    """Shared dense-network initialization, forward pass, and optimization."""

    def __init__(self, hidden_layer_sizes=(100,), *, activation="relu",
                 learning_rate=0.001, n_iters=200, batch_size=32, l2=0.0001,
                 tol=1e-4, n_iter_no_change=10, shuffle=True, random_state=None):
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)
        if not isinstance(hidden_layer_sizes, tuple) or any(not isinstance(size, int) or size <= 0 for size in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive integers")
        if activation not in ("relu", "tanh", "sigmoid"):
            raise ValueError("activation must be 'relu', 'tanh', or 'sigmoid'")
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(n_iters, int) or n_iters <= 0:
            raise ValueError("n_iters must be a positive integer")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("batch_size must be None or a positive integer")
        if l2 < 0 or tol < 0:
            raise ValueError("l2 and tol must be non-negative")
        if n_iter_no_change is not None and (not isinstance(n_iter_no_change, int) or n_iter_no_change <= 0):
            raise ValueError("n_iter_no_change must be None or a positive integer")
        self.hidden_layer_sizes, self.activation = hidden_layer_sizes, activation
        self.learning_rate, self.n_iters, self.batch_size = float(learning_rate), n_iters, batch_size
        self.l2, self.tol, self.n_iter_no_change = float(l2), float(tol), n_iter_no_change
        self.shuffle, self.random_state = shuffle, random_state
        self.coefs_, self.intercepts_, self.loss_curve_ = [], [], []
        self.n_iter_, self.n_features_in_, self.n_outputs_ = 0, None, None
        self.converged_ = False

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

    def _activation(self, values):
        if self.activation == "relu":
            return np.maximum(values, 0.0)
        if self.activation == "tanh":
            return np.tanh(values)
        positive = values >= 0
        result = np.empty_like(values)
        result[positive] = 1 / (1 + np.exp(-values[positive]))
        exponent = np.exp(values[~positive])
        result[~positive] = exponent / (1 + exponent)
        return result

    def _activation_derivative(self, values):
        if self.activation == "relu":
            return (values > 0).astype(float)
        activated = self._activation(values)
        return 1 - activated**2 if self.activation == "tanh" else activated * (1 - activated)

    def _initialize(self, input_size, output_size, rng):
        sizes = (input_size,) + self.hidden_layer_sizes + (output_size,)
        self.coefs_, self.intercepts_ = [], []
        for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2 / fan_in) if self.activation == "relu" and index < len(sizes) - 2 else np.sqrt(1 / fan_in)
            self.coefs_.append(rng.normal(0, scale, (fan_in, fan_out)))
            self.intercepts_.append(np.zeros(fan_out))

    def _output_activation(self, values):
        raise NotImplementedError

    def _forward(self, X):
        activations, preactivations = [X], []
        current = X
        for weights, bias in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            values = current @ weights + bias
            preactivations.append(values)
            current = self._activation(values)
            activations.append(current)
        values = current @ self.coefs_[-1] + self.intercepts_[-1]
        preactivations.append(values)
        activations.append(self._output_activation(values))
        return activations, preactivations

    def _fit_network(self, X, target):
        rng = np.random.default_rng(self.random_state)
        self._initialize(X.shape[1], target.shape[1], rng)
        moment_w = [np.zeros_like(value) for value in self.coefs_]
        velocity_w = [np.zeros_like(value) for value in self.coefs_]
        moment_b = [np.zeros_like(value) for value in self.intercepts_]
        velocity_b = [np.zeros_like(value) for value in self.intercepts_]
        beta1, beta2, epsilon, step = 0.9, 0.999, 1e-8, 0
        batch_size = min(self.batch_size or len(X), len(X))
        self.loss_curve_, self.converged_ = [], False
        best_loss, stale = np.inf, 0
        for epoch in range(self.n_iters):
            indices = rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                activations, preactivations = self._forward(X[batch])
                delta = self._output_delta(activations[-1], target[batch]) / len(batch)
                gradients_w, gradients_b = [], []
                for layer in range(len(self.coefs_) - 1, -1, -1):
                    gradients_w.append(activations[layer].T @ delta + self.l2 * self.coefs_[layer])
                    gradients_b.append(np.sum(delta, axis=0))
                    if layer:
                        delta = (delta @ self.coefs_[layer].T) * self._activation_derivative(preactivations[layer - 1])
                gradients_w.reverse()
                gradients_b.reverse()
                step += 1
                for layer in range(len(self.coefs_)):
                    moment_w[layer] = beta1 * moment_w[layer] + (1 - beta1) * gradients_w[layer]
                    velocity_w[layer] = beta2 * velocity_w[layer] + (1 - beta2) * gradients_w[layer]**2
                    moment_b[layer] = beta1 * moment_b[layer] + (1 - beta1) * gradients_b[layer]
                    velocity_b[layer] = beta2 * velocity_b[layer] + (1 - beta2) * gradients_b[layer]**2
                    mw = moment_w[layer] / (1 - beta1**step)
                    vw = velocity_w[layer] / (1 - beta2**step)
                    mb = moment_b[layer] / (1 - beta1**step)
                    vb = velocity_b[layer] / (1 - beta2**step)
                    self.coefs_[layer] -= self.learning_rate * mw / (np.sqrt(vw) + epsilon)
                    self.intercepts_[layer] -= self.learning_rate * mb / (np.sqrt(vb) + epsilon)
            prediction = self._forward(X)[0][-1]
            loss = self._loss(target, prediction) + 0.5 * self.l2 * sum(np.sum(weights**2) for weights in self.coefs_)
            self.loss_curve_.append(float(loss))
            self.n_iter_ = epoch + 1
            if best_loss - loss > self.tol:
                best_loss, stale = loss, 0
            else:
                stale += 1
                if self.n_iter_no_change is not None and stale >= self.n_iter_no_change:
                    self.converged_ = True
                    break
        return self

    def _validate_query(self, X):
        if not self.coefs_:
            raise ValueError("No model yet. Call fit() first.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X


class MLPClassifier(BaseMLP):
    """Dense neural-network classifier with a softmax output layer."""

    def __init__(self, hidden_layer_sizes=(100,), **kwargs):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, **kwargs)
        self.classes_ = None

    @staticmethod
    def _output_activation(values):
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.sum(exponent, axis=1, keepdims=True)

    @staticmethod
    def _output_delta(prediction, target):
        return prediction - target

    @staticmethod
    def _loss(target, prediction):
        return -np.mean(np.sum(target * np.log(np.maximum(prediction, np.finfo(float).tiny)), axis=1))

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per sample")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least two classes")
        self.n_features_in_, self.n_outputs_ = X.shape[1], len(self.classes_)
        return self._fit_network(X, np.eye(self.n_outputs_)[encoded])

    def predict_proba(self, X):
        return self._forward(self._validate_query(X))[0][-1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class MLPRegressor(BaseMLP):
    """Dense neural-network regressor with a linear output layer."""

    @staticmethod
    def _output_activation(values):
        return values

    @staticmethod
    def _output_delta(prediction, target):
        return 2 * (prediction - target) / target.shape[1]

    @staticmethod
    def _loss(target, prediction):
        return np.mean((target - prediction)**2)

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float)
        self._single_output = y.ndim == 1
        if self._single_output:
            y = y[:, None]
        if y.ndim != 2 or len(y) != len(X):
            raise ValueError("y must have one row per sample")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        self.n_features_in_, self.n_outputs_ = X.shape[1], y.shape[1]
        return self._fit_network(X, y)

    def predict(self, X):
        prediction = self._forward(self._validate_query(X))[0][-1]
        return prediction[:, 0] if self._single_output else prediction

    def score(self, X, y):
        return r2_score(y, self.predict(X))
