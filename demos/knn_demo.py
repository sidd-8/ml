"""Demonstrate KNN classification, regression, and decision boundaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes, load_wine, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNNClassifier
from sklearn.neighbors import KNeighborsRegressor as SklearnKNNRegressor

from knn import KNeighborsClassifier, KNeighborsRegressor
from metrics import accuracy_score, r2_score
from preprocessing import StandardScaler


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "knn.png"


def classification_results():
    dataset = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=42,
        stratify=dataset.target,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    neighbors = np.arange(1, 21)
    our_scores = []
    reference_scores = []
    for k in neighbors:
        ours = KNeighborsClassifier(n_neighbors=int(k)).fit(X_train, y_train)
        reference = SklearnKNNClassifier(n_neighbors=int(k)).fit(X_train, y_train)
        our_scores.append(accuracy_score(y_test, ours.predict(X_test)))
        reference_scores.append(accuracy_score(y_test, reference.predict(X_test)))
    return neighbors, np.asarray(our_scores), np.asarray(reference_scores)


def regression_results():
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    neighbors = np.arange(1, 26)
    our_scores = []
    reference_scores = []
    for k in neighbors:
        ours = KNeighborsRegressor(n_neighbors=int(k), weights="distance").fit(
            X_train, y_train
        )
        reference = SklearnKNNRegressor(
            n_neighbors=int(k), weights="distance"
        ).fit(X_train, y_train)
        our_scores.append(r2_score(y_test, ours.predict(X_test)))
        reference_scores.append(r2_score(y_test, reference.predict(X_test)))
    return neighbors, np.asarray(our_scores), np.asarray(reference_scores)


def decision_boundary(axis):
    X, y = make_moons(n_samples=180, noise=0.22, random_state=7)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KNeighborsClassifier(n_neighbors=9, weights="distance").fit(X_scaled, y)

    x_values = np.linspace(X_scaled[:, 0].min() - 0.6, X_scaled[:, 0].max() + 0.6, 180)
    y_values = np.linspace(X_scaled[:, 1].min() - 0.6, X_scaled[:, 1].max() + 0.6, 180)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    positive_probability = model.predict_proba(grid)[:, 1].reshape(grid_x.shape)

    axis.contourf(
        grid_x,
        grid_y,
        positive_probability,
        levels=np.linspace(0, 1, 11),
        cmap="RdBu",
        alpha=0.55,
    )
    axis.contour(grid_x, grid_y, positive_probability, levels=[0.5], colors="black")
    axis.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=y,
        cmap="RdBu",
        edgecolors="white",
        linewidths=0.5,
        s=28,
    )
    axis.set(
        xlabel="Scaled feature 1",
        ylabel="Scaled feature 2",
        title="Nonlinear decision boundary (k=9)",
    )


def main():
    class_k, class_ours, class_reference = classification_results()
    reg_k, reg_ours, reg_reference = regression_results()

    best_class_index = int(np.argmax(class_ours))
    best_reg_index = int(np.argmax(reg_ours))
    print("\nKNN — held-out model selection")
    print(
        f"Wine classification: best k={class_k[best_class_index]}, "
        f"accuracy={class_ours[best_class_index]:.4f} "
        f"(sklearn={class_reference[best_class_index]:.4f})"
    )
    print(
        f"Diabetes regression: best k={reg_k[best_reg_index]}, "
        f"R2={reg_ours[best_reg_index]:.4f} "
        f"(sklearn={reg_reference[best_reg_index]:.4f})"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("K-Nearest Neighbors from Scratch", fontsize=15)

    axes[0].plot(class_k, class_ours, marker="o", color="#2563eb", label="From scratch")
    axes[0].plot(class_k, class_reference, "--", color="#dc2626", label="scikit-learn")
    axes[0].set(
        xlabel="Number of neighbors (k)",
        ylabel="Test accuracy",
        title="Wine classification",
        xticks=[1, 5, 10, 15, 20],
    )
    axes[0].legend(frameon=False)

    axes[1].plot(reg_k, reg_ours, marker="o", color="#059669", label="From scratch")
    axes[1].plot(reg_k, reg_reference, "--", color="#dc2626", label="scikit-learn")
    axes[1].set(
        xlabel="Number of neighbors (k)",
        ylabel="Test R²",
        title="Diabetes regression",
        xticks=[1, 5, 10, 15, 20, 25],
    )
    axes[1].legend(frameon=False)

    decision_boundary(axes[2])
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
