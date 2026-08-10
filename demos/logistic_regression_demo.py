"""Compare our logistic regression with scikit-learn on breast cancer data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from logistic_regression import LogisticRegression
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
from preprocessing import StandardScaler


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "logistic_regression.png"


def evaluate(y_true, y_pred, probability):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "Log loss": log_loss(y_true, probability),
    }


def print_comparison(ours, reference):
    print("\nBreast cancer classification — held-out test set")
    print(f"{'Metric':<12} {'From scratch':>14} {'scikit-learn':>16}")
    print("-" * 44)
    for metric in ours:
        print(f"{metric:<12} {ours[metric]:>14.4f} {reference[metric]:>16.4f}")


def main():
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(
        learning_rate=0.1,
        n_iters=10_000,
        tolerance=1e-9,
        l2=0.01,
        class_weight="balanced",
        random_state=42,
    ).fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    reference_scaler = SklearnStandardScaler()
    reference_X_train = reference_scaler.fit_transform(X_train)
    reference_X_test = reference_scaler.transform(X_test)
    # Our objective averages data loss; sklearn's L2 objective uses summed loss.
    equivalent_c = 1 / (model.l2 * len(y_train))
    reference_model = SklearnLogisticRegression(
        C=equivalent_c,
        class_weight="balanced",
        max_iter=10_000,
        random_state=42,
    ).fit(reference_X_train, y_train)
    reference_predictions = reference_model.predict(reference_X_test)
    reference_probabilities = reference_model.predict_proba(reference_X_test)[:, 1]

    our_metrics = evaluate(y_test, predictions, probabilities)
    reference_metrics = evaluate(
        y_test, reference_predictions, reference_probabilities
    )
    print_comparison(our_metrics, reference_metrics)
    print(f"\nEpochs: {model.n_iters_} | converged: {model.converged_}")

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    figure.suptitle("Logistic Regression from Scratch — Breast Cancer Dataset", fontsize=15)

    image = axes[0, 0].imshow(matrix, cmap="Blues")
    for row, column in np.ndindex(matrix.shape):
        axes[0, 0].text(column, row, matrix[row, column], ha="center", va="center")
    axes[0, 0].set(
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
        xticks=[0, 1],
        yticks=[0, 1],
    )
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046)

    axes[0, 1].plot(fpr, tpr, color="#2563eb", label=f"AUC = {our_metrics['ROC-AUC']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="#64748b")
    axes[0, 1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(recall, precision, color="#059669")
    axes[1, 0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")

    axes[1, 1].plot(model.loss_history_, color="#7c3aed")
    axes[1, 1].set(xlabel="Epoch", ylabel="Cross-entropy + L2", title="Optimization history", yscale="log")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
