"""Visualize decision tree classification, regression, and feature importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnTreeRegressor

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "decision_tree.png"


def classification_panel(axis):
    X, y = make_moons(n_samples=300, noise=0.25, random_state=8)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5).fit(
        X_train, y_train
    )
    reference = SklearnTreeClassifier(
        max_depth=5, min_samples_leaf=5, random_state=0
    ).fit(X_train, y_train)

    x_values = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 220)
    y_values = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 220)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    probabilities = model.predict_proba(grid)[:, 1].reshape(grid_x.shape)
    axis.contourf(
        grid_x,
        grid_y,
        probabilities,
        levels=np.linspace(0, 1, 11),
        cmap="RdBu",
        alpha=0.55,
    )
    axis.contour(grid_x, grid_y, probabilities, levels=[0.5], colors="black")
    axis.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="RdBu",
        edgecolors="white",
        linewidths=0.6,
        s=32,
    )
    score = model.score(X_test, y_test)
    reference_score = reference.score(X_test, y_test)
    axis.set(
        xlabel="Feature 1",
        ylabel="Feature 2",
        title=f"Classification boundary\nAccuracy {score:.3f} (sklearn {reference_score:.3f})",
    )
    return score, reference_score


def regression_panel(axis):
    rng = np.random.default_rng(12)
    X = np.sort(rng.uniform(0, 2 * np.pi, 220)).reshape(-1, 1)
    y = np.sin(X[:, 0]) + rng.normal(scale=0.16, size=len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=6).fit(
        X_train, y_train
    )
    reference = SklearnTreeRegressor(
        max_depth=5, min_samples_leaf=6, random_state=0
    ).fit(X_train, y_train)
    curve = np.linspace(0, 2 * np.pi, 500).reshape(-1, 1)

    axis.scatter(X_test[:, 0], y_test, alpha=0.6, color="#64748b", label="Test data")
    axis.plot(curve[:, 0], np.sin(curve[:, 0]), "--", color="#dc2626", label="True signal")
    axis.plot(curve[:, 0], model.predict(curve), color="#2563eb", label="Tree prediction")
    score = model.score(X_test, y_test)
    reference_score = reference.score(X_test, y_test)
    axis.set(
        xlabel="x",
        ylabel="Target",
        title=f"Piecewise regression\nR² {score:.3f} (sklearn {reference_score:.3f})",
    )
    axis.legend(frameon=False)
    return score, reference_score


def importance_panel(axis):
    dataset = load_wine()
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3).fit(
        dataset.data, dataset.target
    )
    top_indices = np.argsort(model.feature_importances_)[-7:]
    axis.barh(
        np.asarray(dataset.feature_names)[top_indices],
        model.feature_importances_[top_indices],
        color="#059669",
    )
    axis.set(
        xlabel="Normalized impurity reduction",
        title="Wine feature importances",
    )
    return model


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("Decision Trees from Scratch", fontsize=15)
    classification_scores = classification_panel(axes[0])
    regression_scores = regression_panel(axes[1])
    wine_model = importance_panel(axes[2])

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print("\nDecision Trees — held-out comparisons")
    print(
        f"Moon classification accuracy: {classification_scores[0]:.4f} "
        f"(sklearn={classification_scores[1]:.4f})"
    )
    print(
        f"Sine regression R2: {regression_scores[0]:.4f} "
        f"(sklearn={regression_scores[1]:.4f})"
    )
    print(f"Wine tree: depth={wine_model.get_depth()}, leaves={wine_model.get_n_leaves()}")
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
