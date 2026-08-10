"""Compare our linear regression with scikit-learn on the diabetes dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from linear_regression import GradientDescentLR
from metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from preprocessing import StandardScaler


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "linear_regression.png"


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_comparison(ours, reference):
    print("\nDiabetes regression — held-out test set")
    print(f"{'Metric':<10} {'From scratch':>14} {'scikit-learn':>16}")
    print("-" * 42)
    for metric in ours:
        print(f"{metric:<10} {ours[metric]:>14.4f} {reference[metric]:>16.4f}")


def main():
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientDescentLR(
        lr=0.05,
        n_iters=10_000,
        tolerance=1e-8,
        random_state=42,
    ).fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    # Use an independent scaler so the comparison exercises the full pipeline.
    reference_scaler = SklearnStandardScaler()
    reference_X_train = reference_scaler.fit_transform(X_train)
    reference_X_test = reference_scaler.transform(X_test)
    reference_model = LinearRegression().fit(reference_X_train, y_train)
    reference_predictions = reference_model.predict(reference_X_test)

    our_metrics = evaluate(y_test, predictions)
    reference_metrics = evaluate(y_test, reference_predictions)
    print_comparison(our_metrics, reference_metrics)
    print(f"\nEpochs: {model.n_iters_} | converged: {model.converged_}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    figure.suptitle("Linear Regression from Scratch — Diabetes Dataset", fontsize=15)

    bounds = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    axes[0].scatter(y_test, predictions, alpha=0.7, color="#2563eb")
    axes[0].plot(bounds, bounds, "--", color="#dc2626", label="Perfect prediction")
    axes[0].set(xlabel="Actual target", ylabel="Predicted target", title="Predicted vs actual")
    axes[0].legend(frameon=False)

    residuals = y_test - predictions
    axes[1].scatter(predictions, residuals, alpha=0.7, color="#059669")
    axes[1].axhline(0, linestyle="--", color="#dc2626")
    axes[1].set(xlabel="Predicted target", ylabel="Residual", title="Residual diagnostics")

    excess_loss = np.asarray(model.loss_history_) - model.loss_history_[-1]
    axes[2].plot(np.maximum(excess_loss, 1e-12), color="#7c3aed")
    axes[2].set(
        xlabel="Epoch",
        ylabel="MSE above final value",
        title="Optimization history",
        yscale="log",
    )

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
