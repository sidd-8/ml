"""Contrast DBSCAN and K-Means on non-convex moon-shaped clusters."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

from clustering import DBSCAN, KMeans
from preprocessing import StandardScaler


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "dbscan.png"
EPS = 0.22
MIN_SAMPLES = 5


def main():
    X, true_labels = make_moons(n_samples=450, noise=0.08, random_state=11)
    X = StandardScaler().fit_transform(X)
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)

    dbscan_ari = adjusted_rand_score(true_labels, dbscan.labels_)
    kmeans_ari = adjusted_rand_score(true_labels, kmeans.labels_)
    noise_count = int(np.sum(dbscan.labels_ == -1))
    border_mask = dbscan.labels_ != -1
    border_mask[dbscan.core_sample_indices_] = False

    differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    pairwise = np.sqrt(np.sum(differences**2, axis=2))
    neighbor_distances = np.sort(pairwise, axis=1)[:, MIN_SAMPLES - 1]
    sorted_neighbor_distances = np.sort(neighbor_distances)

    print("\nDBSCAN — non-convex clustering")
    print(f"Clusters found: {dbscan.n_clusters_}")
    print(f"Core samples: {len(dbscan.core_sample_indices_)}")
    print(f"Border samples: {np.sum(border_mask)}")
    print(f"Noise samples: {noise_count}")
    print(f"Adjusted Rand index: DBSCAN={dbscan_ari:.4f}, K-Means={kmeans_ari:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("DBSCAN from Scratch — Density vs Centroids", fontsize=15)

    axes[0].scatter(
        X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", s=22, alpha=0.8
    )
    axes[0].scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=180,
        color="#dc2626",
        edgecolors="white",
        label="Centers",
    )
    axes[0].set(
        xlabel="Scaled feature 1",
        ylabel="Scaled feature 2",
        title=f"K-Means\nAdjusted Rand = {kmeans_ari:.3f}",
    )
    axes[0].legend(frameon=False)

    core_mask = np.zeros(len(X), dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    noise_mask = dbscan.labels_ == -1
    axes[1].scatter(
        X[core_mask, 0],
        X[core_mask, 1],
        c=dbscan.labels_[core_mask],
        cmap="viridis",
        s=24,
        alpha=0.85,
        label="Core",
    )
    axes[1].scatter(
        X[border_mask, 0],
        X[border_mask, 1],
        c=dbscan.labels_[border_mask],
        cmap="viridis",
        marker="s",
        s=38,
        edgecolors="black",
        linewidths=0.5,
        label="Border",
    )
    axes[1].scatter(
        X[noise_mask, 0],
        X[noise_mask, 1],
        color="black",
        marker="x",
        s=55,
        label="Noise",
    )
    axes[1].set(
        xlabel="Scaled feature 1",
        ylabel="Scaled feature 2",
        title=f"DBSCAN\nAdjusted Rand = {dbscan_ari:.3f}",
    )
    axes[1].legend(frameon=False)

    axes[2].plot(sorted_neighbor_distances, color="#2563eb")
    axes[2].axhline(EPS, linestyle="--", color="#dc2626", label=f"eps = {EPS}")
    axes[2].set(
        xlabel="Samples sorted by neighborhood distance",
        ylabel=f"Distance to neighbor {MIN_SAMPLES}",
        title="k-distance diagnostic",
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
