"""Demonstrate multinomial softmax regression on Iris."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SklearnLogistic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression import SoftmaxRegression


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "softmax_regression.png"


def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, stratify=data.target, random_state=8)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    model = SoftmaxRegression(learning_rate=0.1, n_iters=1200, l2=0.001, random_state=4).fit(X_train, y_train)
    reference = SklearnLogistic(C=1000, max_iter=2000).fit(X_train, y_train)
    print("\nSoftmax regression — multiclass Iris classification")
    print(f"Held-out accuracy: {model.score(X_test, y_test):.4f} (sklearn={reference.score(X_test, y_test):.4f})")
    print(f"Final cross-entropy objective: {model.loss_history_[-1]:.5f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(model.loss_history_, color="#2563eb")
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Cross entropy + L2")
    probabilities = model.predict_proba(X_test)
    axes[1].hist(probabilities.max(axis=1), bins=12, color="#059669", alpha=0.85)
    axes[1].set(title="Held-out prediction confidence", xlabel="Maximum class probability", ylabel="Samples")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
