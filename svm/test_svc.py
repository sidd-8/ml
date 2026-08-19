import unittest

import numpy as np
from sklearn.datasets import load_iris, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from svm import SVC


class SVCTests(unittest.TestCase):
    def test_linear_binary_classification_and_attributes(self):
        X, y = load_iris(return_X_y=True)
        mask = y < 2
        X = StandardScaler().fit_transform(X[mask])
        model = SVC(kernel="linear", C=2, random_state=2).fit(X, y[mask])
        self.assertGreater(model.score(X, y[mask]), 0.98)
        self.assertGreater(len(model.support_), 0)
        self.assertEqual(model.coef_.shape, (X.shape[1],))

    def test_rbf_learns_nonlinear_boundary(self):
        X, y = make_circles(n_samples=120, factor=0.35, noise=0.05, random_state=4)
        model = SVC(C=10, gamma=2.0, random_state=3).fit(X, y)
        self.assertGreater(model.score(X, y), 0.97)

    def test_multiclass_and_arbitrary_labels(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.25, stratify=y, random_state=3)
        labels = np.asarray(["a", "b", "c"])[y_train]
        expected = np.asarray(["a", "b", "c"])[y_test]
        model = SVC(C=3, gamma="scale", random_state=5).fit(X_train, labels)
        self.assertGreater(model.score(X_test, expected), 0.88)
        self.assertEqual(model.decision_function(X_test).shape, (len(X_test), 3))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "kernel"):
            SVC(kernel="bad")
        with self.assertRaisesRegex(ValueError, "fit"):
            SVC().predict([[0]])


if __name__ == "__main__":
    unittest.main()
