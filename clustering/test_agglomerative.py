import unittest

import numpy as np
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering import AgglomerativeClustering


class AgglomerativeTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=8)

    def test_linkages_match_sklearn_partition(self):
        for linkage in ("single", "complete", "average", "ward"):
            with self.subTest(linkage=linkage):
                ours = AgglomerativeClustering(3, linkage=linkage).fit(self.X)
                reference = SklearnAgglomerative(n_clusters=3, linkage=linkage).fit(self.X)
                self.assertGreater(adjusted_rand_score(ours.labels_, reference.labels_), 0.99)
                self.assertEqual(ours.children_.shape, (len(self.X) - 1, 2))
                self.assertEqual(ours.distances_.shape, (len(self.X) - 1,))

    def test_validation_and_independent_labels(self):
        with self.assertRaisesRegex(ValueError, "linkage"):
            AgglomerativeClustering(linkage="bad")
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            AgglomerativeClustering(3).fit([[0], [1]])
        model = AgglomerativeClustering(3)
        labels = model.fit_predict(self.X)
        labels[0] = 99
        self.assertNotEqual(model.labels_[0], 99)


if __name__ == "__main__":
    unittest.main()
