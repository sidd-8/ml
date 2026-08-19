"""Demonstrate Gaussian-mixture clustering and soft assignments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SklearnGMM

from mixture import GaussianMixture


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "gmm.png"


def main():
    X, truth = make_blobs(n_samples=450, centers=[[-4, -1], [0, 4], [4, 0]], cluster_std=[0.7, 1.1, 0.8], random_state=8)
    model = GaussianMixture(3, n_init=5, random_state=3).fit(X)
    reference = SklearnGMM(3, n_init=5, random_state=3).fit(X)
    labels = model.predict(X)
    confidence = np.max(model.predict_proba(X), axis=1)
    print("\nGaussian mixture — probabilistic clustering")
    print(f"Adjusted Rand index: {adjusted_rand_score(truth, labels):.4f}")
    print(f"Mean log likelihood: {model.score(X):.4f} (sklearn={reference.score(X):.4f})")
    print(f"Converged in {model.n_iter_} EM iterations: {model.converged_}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20, alpha=0.75)
    axes[0].scatter(model.means_[:, 0], model.means_[:, 1], marker="X", s=180, color="#dc2626", edgecolors="white")
    axes[0].set(title="Maximum-responsibility clusters", xlabel="Feature 1", ylabel="Feature 2")
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=confidence, cmap="magma", s=20, vmin=0.5, vmax=1)
    axes[1].set(title="Assignment confidence", xlabel="Feature 1", ylabel="Feature 2")
    figure.colorbar(scatter, ax=axes[1], label="Maximum responsibility")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
