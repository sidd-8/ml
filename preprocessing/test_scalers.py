import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from preprocessing import MinMaxScaler, StandardScaler


class StandardScalerTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [[1.0, 10.0, 5.0], [2.0, 20.0, 5.0], [4.0, 30.0, 5.0]]
        )

    def test_matches_sklearn_including_constant_feature(self):
        ours = StandardScaler().fit(self.X)
        reference = SklearnStandardScaler().fit(self.X)

        np.testing.assert_allclose(ours.mean_, reference.mean_)
        np.testing.assert_allclose(ours.var_, reference.var_)
        np.testing.assert_allclose(ours.scale_, reference.scale_)
        np.testing.assert_allclose(ours.transform(self.X), reference.transform(self.X))

    def test_inverse_transform_recovers_input(self):
        scaler = StandardScaler()
        transformed = scaler.fit_transform(self.X)
        np.testing.assert_allclose(scaler.inverse_transform(transformed), self.X)

    def test_optional_centering_and_scaling_match_sklearn(self):
        for with_mean, with_std in ((False, True), (True, False), (False, False)):
            ours = StandardScaler(with_mean=with_mean, with_std=with_std)
            reference = SklearnStandardScaler(
                with_mean=with_mean, with_std=with_std
            )
            np.testing.assert_allclose(
                ours.fit_transform(self.X), reference.fit_transform(self.X)
            )


class MinMaxScalerTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [[1.0, 10.0, 5.0], [2.0, 20.0, 5.0], [4.0, 30.0, 5.0]]
        )

    def test_matches_sklearn_with_custom_range(self):
        ours = MinMaxScaler(feature_range=(-1, 2)).fit(self.X)
        reference = SklearnMinMaxScaler(feature_range=(-1, 2)).fit(self.X)

        np.testing.assert_allclose(ours.data_min_, reference.data_min_)
        np.testing.assert_allclose(ours.data_max_, reference.data_max_)
        np.testing.assert_allclose(ours.data_range_, reference.data_range_)
        np.testing.assert_allclose(ours.scale_, reference.scale_)
        np.testing.assert_allclose(ours.min_, reference.min_)
        np.testing.assert_allclose(ours.transform(self.X), reference.transform(self.X))

    def test_inverse_transform_recovers_input(self):
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(self.X)
        np.testing.assert_allclose(scaler.inverse_transform(transformed), self.X)

    def test_clip_limits_unseen_values(self):
        scaler = MinMaxScaler(clip=True).fit([[0], [10]])
        np.testing.assert_allclose(scaler.transform([[-5], [15]]), [[0], [1]])


class ScalerValidationTests(unittest.TestCase):
    def test_transform_before_fit_fails(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            StandardScaler().transform([[1]])

    def test_feature_count_must_match(self):
        scaler = MinMaxScaler().fit([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "different number"):
            scaler.transform([[1, 2, 3]])

    def test_rejects_non_finite_data(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            StandardScaler().fit([[1], [np.nan]])

    def test_rejects_invalid_feature_range(self):
        with self.assertRaisesRegex(ValueError, "lower < upper"):
            MinMaxScaler(feature_range=(1, 1))


if __name__ == "__main__":
    unittest.main()
