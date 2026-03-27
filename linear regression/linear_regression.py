import numpy as np

class BaseLinearRegression:
    def predict(self, X):
        ## Add 1s to account for the extra parameter.
        X = self._add_bias(X)

        ## Once theta is learned, we return our prediction h(x)
        return X @ self.theta

    def _add_bias(self, X):
        # Add column of 1s for the extra parameter (intercept)
        return np.c_[np.ones((X.shape[0], 1)), X]


class NormalEquationLR(BaseLinearRegression):
    ## For the closed form solution: Not very scalable as matrix inversion is an expensive operation for large datasets.
    def fit(self, X, y):
        X = self._add_bias(X)

        # θ = (XᵀX)^(-1) Xᵀy
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self


class GradientDescentLR(BaseLinearRegression):
    ## Gradient Descent: Scalable
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        X = self._add_bias(X)

        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.n_iters):
            # Predictions
            y_pred = X @ self.theta

            # Gradient: (2/m) * Xᵀ (y_pred - y)
            gradient = (2 / m) * X.T @ (y_pred - y)

            # Update
            self.theta -= self.lr * gradient

        return self
    

X = np.array([
    [8, 8, 9],
    [2, 0, 1],
    [5, 5, 5],
    [9, 6, 10]
])

y = np.array([95, 15, 54, 96])

model = NormalEquationLR()
model.fit(X, y)
preds = model.predict(np.array([[10, 5, 0]]))     # make predictions
print(preds)