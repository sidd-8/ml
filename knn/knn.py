"""K-nearest-neighbors classification and regression."""

import numpy as np

from metrics import accuracy_score, r2_score

from ._base import BaseKNN


class KNeighborsClassifier(BaseKNN):
    """Classify samples by a vote among their nearest training examples."""

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", p=2):
        super().__init__(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p)
        self.classes_ = None

    def fit(self, X, y):
        """Store training examples and discover the sorted class labels."""
        self._fit(X, y, numeric_target=False)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict_proba(self, X):
        """Return neighbor-vote proportions in ``classes_`` order."""
        distances, indices = self.kneighbors(X)
        neighbor_labels = self.y_train_[indices]
        neighbor_weights = self._get_neighbor_weights(distances)
        probabilities = np.zeros((len(indices), len(self.classes_)), dtype=float)

        for class_index, label in enumerate(self.classes_):
            probabilities[:, class_index] = np.sum(
                neighbor_weights * (neighbor_labels == label), axis=1
            )
        probabilities /= np.sum(neighbor_weights, axis=1, keepdims=True)
        return probabilities

    def predict(self, X):
        """Predict the class with the greatest weighted neighbor vote."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        """Return mean classification accuracy."""
        return accuracy_score(y, self.predict(X))


class KNeighborsRegressor(BaseKNN):
    """Predict continuous targets from nearby training examples."""

    def fit(self, X, y):
        """Store training examples and continuous targets."""
        return self._fit(X, y, numeric_target=True)

    def predict(self, X):
        """Return the weighted mean target among nearest neighbors."""
        distances, indices = self.kneighbors(X)
        neighbor_targets = self.y_train_[indices]
        neighbor_weights = self._get_neighbor_weights(distances)
        return np.sum(neighbor_targets * neighbor_weights, axis=1) / np.sum(
            neighbor_weights, axis=1
        )

    def score(self, X, y):
        """Return the coefficient of determination (R-squared)."""
        return r2_score(y, self.predict(X))
