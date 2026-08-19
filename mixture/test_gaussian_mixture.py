import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

from mixture import GaussianMixture


class GaussianMixtureTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=4)

    def test_em_finds_clusters_and_normalized_responsibilities(self):
        model = GaussianMixture(3, n_init=4, random_state=7).fit(self.X)
        self.assertGreater(adjusted_rand_score(self.y, model.predict(self.X)), 0.98)
        np.testing.assert_allclose(model.predict_proba(self.X).sum(axis=1), 1.0)
        self.assertAlmostEqual(model.weights_.sum(), 1.0)

    def test_score_is_comparable_to_sklearn(self):
        ours = GaussianMixture(3, n_init=5, random_state=2).fit(self.X)
        reference = SklearnGaussianMixture(3, n_init=5, random_state=2).fit(self.X)
        self.assertAlmostEqual(ours.score(self.X), reference.score(self.X), places=2)

    def test_diagonal_covariance_and_sampling(self):
        model = GaussianMixture(3, covariance_type="diag", random_state=1).fit(self.X)
        samples, labels = model.sample(25, random_state=3)
        self.assertEqual(samples.shape, (25, 2))
        self.assertEqual(labels.shape, (25,))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "covariance_type"):
            GaussianMixture(covariance_type="bad")
        with self.assertRaisesRegex(ValueError, "fit"):
            GaussianMixture().predict([[0]])


if __name__ == "__main__":
    unittest.main()
