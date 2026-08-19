"""Demonstrate neural-network classification and regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network import MLPClassifier, MLPRegressor


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "mlp.png"


def main():
    X, y = make_moons(n_samples=300, noise=0.14, random_state=4)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=5)
    classifier = MLPClassifier((24, 12), learning_rate=0.01, n_iters=300, batch_size=32, n_iter_no_change=35, random_state=2).fit(X_train, y_train)
    Xr, yr = make_regression(n_samples=240, n_features=4, noise=3, random_state=7)
    Xr = StandardScaler().fit_transform(Xr)
    yr = (yr - yr.mean()) / yr.std()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, random_state=6)
    regressor = MLPRegressor((20,), activation="tanh", learning_rate=0.01, n_iters=300, batch_size=32, n_iter_no_change=35, random_state=3).fit(Xr_train, yr_train)
    print("\nMultilayer perceptron — backpropagation with Adam")
    print(f"Moon classification accuracy: {classifier.score(X_test, y_test):.4f}")
    print(f"Synthetic regression R-squared: {regressor.score(Xr_test, yr_test):.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    low, high = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 180), np.linspace(low[1], high[1], 180))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    axes[0].contourf(xx, yy, classifier.predict_proba(grid)[:, 1].reshape(xx.shape), levels=20, cmap="coolwarm", alpha=0.35)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="white")
    axes[0].set(title="Nonlinear classification boundary")
    axes[1].plot(classifier.loss_curve_, label="Classifier")
    axes[1].plot(regressor.loss_curve_, label="Regressor")
    axes[1].set(title="Training losses", xlabel="Epoch", ylabel="Objective")
    axes[1].legend(frameon=False)
    prediction = regressor.predict(Xr_test)
    axes[2].scatter(yr_test, prediction, alpha=0.75, color="#059669")
    bounds = [min(yr_test.min(), prediction.min()), max(yr_test.max(), prediction.max())]
    axes[2].plot(bounds, bounds, "--", color="#111827")
    axes[2].set(title="Regression predictions", xlabel="True target", ylabel="Prediction")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
