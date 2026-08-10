import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score as sklearn_f1,
    log_loss as sklearn_log_loss,
    precision_recall_curve as sklearn_precision_recall_curve,
    precision_score as sklearn_precision,
    recall_score as sklearn_recall,
    roc_auc_score as sklearn_roc_auc,
    roc_curve as sklearn_roc_curve,
)

from metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ClassificationMetricTests(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 0, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0])
        self.y_score = np.array([0.05, 0.65, 0.8, 0.9, 0.2, 0.4])

    def test_label_metrics_match_sklearn(self):
        self.assertEqual(
            accuracy_score(self.y_true, self.y_pred),
            sklearn_accuracy(self.y_true, self.y_pred),
        )
        np.testing.assert_array_equal(
            confusion_matrix(self.y_true, self.y_pred),
            sklearn_confusion_matrix(self.y_true, self.y_pred),
        )
        self.assertEqual(
            precision_score(self.y_true, self.y_pred),
            sklearn_precision(self.y_true, self.y_pred),
        )
        self.assertEqual(
            recall_score(self.y_true, self.y_pred),
            sklearn_recall(self.y_true, self.y_pred),
        )
        self.assertEqual(
            f1_score(self.y_true, self.y_pred),
            sklearn_f1(self.y_true, self.y_pred),
        )

    def test_probability_metrics_match_sklearn(self):
        self.assertAlmostEqual(
            log_loss(self.y_true, self.y_score),
            sklearn_log_loss(self.y_true, self.y_score),
        )
        self.assertAlmostEqual(
            roc_auc_score(self.y_true, self.y_score),
            sklearn_roc_auc(self.y_true, self.y_score),
        )

    def test_curves_match_sklearn(self):
        actual_fpr, actual_tpr, actual_thresholds = roc_curve(
            self.y_true, self.y_score
        )
        expected_fpr, expected_tpr, expected_thresholds = sklearn_roc_curve(
            self.y_true, self.y_score, drop_intermediate=False
        )
        np.testing.assert_allclose(actual_fpr, expected_fpr)
        np.testing.assert_allclose(actual_tpr, expected_tpr)
        np.testing.assert_allclose(actual_thresholds, expected_thresholds)

        actual_precision, actual_recall, actual_thresholds = precision_recall_curve(
            self.y_true, self.y_score
        )
        expected_precision, expected_recall, expected_thresholds = (
            sklearn_precision_recall_curve(self.y_true, self.y_score)
        )
        np.testing.assert_allclose(actual_precision, expected_precision)
        np.testing.assert_allclose(actual_recall, expected_recall)
        np.testing.assert_allclose(actual_thresholds, expected_thresholds)

    def test_custom_string_labels(self):
        y_true = np.array(["no", "yes", "yes", "no"])
        y_pred = np.array(["no", "yes", "no", "no"])
        self.assertEqual(
            precision_score(y_true, y_pred, positive_label="yes"), 1.0
        )
        self.assertEqual(recall_score(y_true, y_pred, positive_label="yes"), 0.5)

    def test_zero_division_policy(self):
        self.assertEqual(precision_score([0, 1], [0, 0]), 0.0)
        self.assertEqual(
            precision_score([0, 1], [0, 0], zero_division=1), 1.0
        )
        with self.assertRaises(ZeroDivisionError):
            precision_score([0, 1], [0, 0], zero_division="raise")

    def test_rejects_invalid_probabilities(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            log_loss([0, 1], [-0.1, 1.0])


if __name__ == "__main__":
    unittest.main()
