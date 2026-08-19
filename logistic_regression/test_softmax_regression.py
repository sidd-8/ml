import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression import SoftmaxRegression


class SoftmaxRegressionTests(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=9)

    def test_multiclass_accuracy_and_probabilities(self):
        model = SoftmaxRegression(learning_rate=0.1, n_iters=1500, l2=0.001, random_state=1).fit(self.X_train, self.y_train)
        self.assertGreater(model.score(self.X_test, self.y_test), 0.9)
        probabilities = model.predict_proba(self.X_test)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(model.coef_.shape, (3, 4))

    def test_minibatches_are_reproducible_and_labels_are_arbitrary(self):
        labels = np.asarray(["setosa", "versicolor", "virginica"])[self.y_train]
        settings = dict(n_iters=100, batch_size=16, random_state=4)
        first = SoftmaxRegression(**settings).fit(self.X_train, labels)
        second = SoftmaxRegression(**settings).fit(self.X_train, labels)
        np.testing.assert_allclose(first.coef_, second.coef_)
        np.testing.assert_array_equal(first.predict(self.X_test), second.predict(self.X_test))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            SoftmaxRegression().predict([[0]])
        with self.assertRaisesRegex(ValueError, "at least two"):
            SoftmaxRegression().fit([[0], [1]], ["a", "a"])


if __name__ == "__main__":
    unittest.main()
