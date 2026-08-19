"""Demonstrate PCA projection, explained variance, and reconstruction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

from decomposition import PCA


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "pca.png"


def main():
    data = load_iris()
    X, y = data.data, data.target
    model = PCA(n_components=2).fit(X)
    reference = SklearnPCA(n_components=2).fit(X)
    projected = model.transform(X)
    reconstruction_error = np.mean((X - model.inverse_transform(projected)) ** 2)
    print("\nPCA — Iris dimensionality reduction")
    print(f"Explained variance: {model.explained_variance_ratio_.sum():.4f} (sklearn={reference.explained_variance_ratio_.sum():.4f})")
    print(f"Two-component reconstruction MSE: {reconstruction_error:.6f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, name in enumerate(data.target_names):
        mask = y == label
        axes[0].scatter(projected[mask, 0], projected[mask, 1], label=name, alpha=0.75)
    axes[0].set(xlabel="Principal component 1", ylabel="Principal component 2", title="Iris projected to two dimensions")
    axes[0].legend(frameon=False)
    full = PCA().fit(X)
    axes[1].bar(np.arange(1, len(full.explained_variance_ratio_) + 1), full.explained_variance_ratio_, color="#2563eb")
    axes[1].plot(np.arange(1, len(full.explained_variance_ratio_) + 1), np.cumsum(full.explained_variance_ratio_), marker="o", color="#dc2626")
    axes[1].set(xlabel="Component", ylabel="Variance ratio", title="Individual and cumulative variance", ylim=(0, 1.05))
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
