import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

from decomposition import PCA


class PCATests(unittest.TestCase):
    def setUp(self):
        self.X = load_iris().data

    def test_variance_and_reconstruction_match_sklearn(self):
        ours = PCA(n_components=2).fit(self.X)
        reference = SklearnPCA(n_components=2).fit(self.X)
        np.testing.assert_allclose(ours.explained_variance_, reference.explained_variance_)
        np.testing.assert_allclose(ours.explained_variance_ratio_, reference.explained_variance_ratio_)
        np.testing.assert_allclose(ours.inverse_transform(ours.transform(self.X)), reference.inverse_transform(reference.transform(self.X)))

    def test_fractional_components_and_whitening(self):
        model = PCA(n_components=0.95, whiten=True).fit(self.X)
        transformed = model.transform(self.X)
        self.assertGreaterEqual(np.sum(model.explained_variance_ratio_), 0.95)
        np.testing.assert_allclose(np.var(transformed, axis=0, ddof=1), 1.0)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "n_components"):
            PCA(n_components=0)
        with self.assertRaisesRegex(ValueError, "fit"):
            PCA(2).transform(self.X)
        with self.assertRaisesRegex(ValueError, "finite"):
            PCA(1).fit([[np.nan, 1]])


if __name__ == "__main__":
    unittest.main()
