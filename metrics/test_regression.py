import unittest

import numpy as np
from sklearn.metrics import (
    mean_absolute_error as sklearn_mae,
    mean_absolute_percentage_error as sklearn_mape,
    mean_squared_error as sklearn_mse,
    r2_score as sklearn_r2,
    root_mean_squared_error as sklearn_rmse,
)

from metrics import (
    adjusted_r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


class RegressionMetricTests(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([3.0, -0.5, 2.0, 7.0])
        self.y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    def test_metrics_match_sklearn(self):
        self.assertAlmostEqual(
            mean_squared_error(self.y_true, self.y_pred),
            sklearn_mse(self.y_true, self.y_pred),
        )
        self.assertAlmostEqual(
            root_mean_squared_error(self.y_true, self.y_pred),
            sklearn_rmse(self.y_true, self.y_pred),
        )
        self.assertAlmostEqual(
            mean_absolute_error(self.y_true, self.y_pred),
            sklearn_mae(self.y_true, self.y_pred),
        )
        self.assertAlmostEqual(
            r2_score(self.y_true, self.y_pred),
            sklearn_r2(self.y_true, self.y_pred),
        )
        self.assertAlmostEqual(
            mean_absolute_percentage_error(self.y_true, self.y_pred),
            sklearn_mape(self.y_true, self.y_pred),
        )

    def test_adjusted_r2_uses_feature_count(self):
        expected = 1 - (1 - r2_score(self.y_true, self.y_pred)) * 3 / 2
        self.assertAlmostEqual(
            adjusted_r2_score(self.y_true, self.y_pred, n_features=1), expected
        )

    def test_constant_target_has_finite_r2(self):
        self.assertEqual(r2_score([2, 2], [2, 2]), 1.0)
        self.assertEqual(r2_score([2, 2], [1, 2]), 0.0)

    def test_mape_zero_policy_is_explicit(self):
        with self.assertRaises(ZeroDivisionError):
            mean_absolute_percentage_error([0, 2], [1, 1])
        self.assertAlmostEqual(
            mean_absolute_percentage_error([0, 2], [1, 1], zero_policy="ignore"),
            0.5,
        )

    def test_rejects_invalid_shapes_and_adjusted_r2_sample_count(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            mean_squared_error([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "n_samples"):
            adjusted_r2_score([1, 2], [1, 2], n_features=1)


if __name__ == "__main__":
    unittest.main()
