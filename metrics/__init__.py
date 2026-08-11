"""Evaluation metrics implemented from scratch."""

from .classification import (
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
from .clustering import silhouette_score
from .regression import (
    adjusted_r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

__all__ = [
    "accuracy_score",
    "adjusted_r2_score",
    "confusion_matrix",
    "f1_score",
    "log_loss",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "precision_recall_curve",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
    "roc_curve",
    "root_mean_squared_error",
    "silhouette_score",
]
