"""Demonstrate hierarchical clustering and its merge distances."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering import AgglomerativeClustering


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "agglomerative.png"


def main():
    X, truth = make_blobs(n_samples=120, centers=4, cluster_std=0.65, random_state=12)
    model = AgglomerativeClustering(4, linkage="ward").fit(X)
    reference = SklearnAgglomerative(n_clusters=4, linkage="ward").fit(X)
    print("\nAgglomerative clustering — Ward linkage")
    print(f"Agreement with sklearn: {adjusted_rand_score(reference.labels_, model.labels_):.4f}")
    print(f"Adjusted Rand index against known groups: {adjusted_rand_score(truth, model.labels_):.4f}")
    print(f"Recorded merges: {len(model.children_)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=24, alpha=0.8)
    axes[0].set(title="Ward-linkage assignments", xlabel="Feature 1", ylabel="Feature 2")
    merge_number = np.arange(1, len(model.distances_) + 1)
    axes[1].plot(merge_number, model.distances_, color="#7c3aed")
    axes[1].set(title="Hierarchy merge distances", xlabel="Merge", ylabel="Ward distance")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
