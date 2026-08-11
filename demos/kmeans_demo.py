"""Demonstrate K-Means clusters, elbow analysis, and silhouette selection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering import KMeans
from metrics import silhouette_score


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "kmeans.png"


def main():
    X, true_labels = make_blobs(
        n_samples=500,
        centers=[[-5, -2], [-1, 4], [3, 3], [5, -2]],
        cluster_std=[0.8, 0.65, 0.9, 0.75],
        random_state=12,
    )
    cluster_counts = np.arange(2, 9)
    inertias = []
    reference_inertias = []
    silhouettes = []
    models = {}
    for count in cluster_counts:
        model = KMeans(n_clusters=int(count), n_init=10, random_state=42).fit(X)
        reference = SklearnKMeans(
            n_clusters=int(count), n_init=10, random_state=42
        ).fit(X)
        models[count] = model
        inertias.append(model.inertia_)
        reference_inertias.append(reference.inertia_)
        silhouettes.append(silhouette_score(X, model.labels_))

    best_count = int(cluster_counts[np.argmax(silhouettes)])
    model = models[best_count]
    reference = SklearnKMeans(
        n_clusters=best_count, n_init=10, random_state=42
    ).fit(X)
    print("\nK-Means — unsupervised model selection")
    print(f"Best silhouette k: {best_count}")
    print(f"Silhouette score: {max(silhouettes):.4f}")
    print(f"Inertia: {model.inertia_:.4f} (sklearn={reference.inertia_:.4f})")
    print(f"Adjusted Rand index against known blob groups: {adjusted_rand_score(true_labels, model.labels_):.4f}")
    print(f"Iterations: {model.n_iter_}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("K-Means Clustering from Scratch", fontsize=15)

    x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 220)
    y_values = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 220)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    regions = model.predict(grid).reshape(grid_x.shape)
    axes[0].contourf(grid_x, grid_y, regions, alpha=0.18, cmap="viridis")
    axes[0].scatter(
        X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=18, alpha=0.75
    )
    axes[0].scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=180,
        color="#dc2626",
        edgecolors="white",
        linewidths=1.2,
        label="Learned centers",
    )
    axes[0].set(
        xlabel="Feature 1",
        ylabel="Feature 2",
        title=f"Assignments and Voronoi regions (k={best_count})",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(cluster_counts, inertias, marker="o", color="#2563eb", label="From scratch")
    axes[1].plot(
        cluster_counts,
        reference_inertias,
        "--",
        color="#dc2626",
        label="scikit-learn",
    )
    axes[1].set(
        xlabel="Number of clusters (k)",
        ylabel="Within-cluster sum of squares",
        title="Elbow analysis",
        xticks=cluster_counts,
    )
    axes[1].legend(frameon=False)

    axes[2].plot(cluster_counts, silhouettes, marker="o", color="#059669")
    axes[2].axvline(best_count, linestyle="--", color="#7c3aed", label=f"Best k = {best_count}")
    axes[2].set(
        xlabel="Number of clusters (k)",
        ylabel="Mean silhouette coefficient",
        title="Silhouette analysis",
        xticks=cluster_counts,
    )
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
