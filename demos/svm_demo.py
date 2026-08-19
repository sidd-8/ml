"""Demonstrate an RBF support-vector classifier on nonlinear data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC as SklearnSVC

from svm import SVC


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "svm.png"


def main():
    X, y = make_circles(n_samples=180, factor=0.35, noise=0.07, random_state=7)
    model = SVC(C=10, gamma=2, random_state=2).fit(X, y)
    reference = SklearnSVC(C=10, gamma=2).fit(X, y)
    print("\nSupport vector machine — RBF kernel")
    print(f"Training accuracy: {model.score(X, y):.4f} (sklearn={reference.score(X, y):.4f})")
    print(f"Support vectors: {len(model.support_)} (sklearn={len(reference.support_)})")

    low, high = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 220), np.linspace(low[1], high[1], 220))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    scores = model.decision_function(grid).reshape(xx.shape)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.contourf(xx, yy, scores, levels=30, cmap="coolwarm", alpha=0.25)
    axis.contour(xx, yy, scores, levels=[-1, 0, 1], colors=["#64748b", "#111827", "#64748b"], linestyles=["--", "-", "--"])
    axis.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="white", s=32)
    axis.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], facecolors="none", edgecolors="#111827", s=85, label="Support vectors")
    axis.set(title="RBF SVM decision function", xlabel="Feature 1", ylabel="Feature 2")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
