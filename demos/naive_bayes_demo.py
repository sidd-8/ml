"""Compare three Naive Bayes likelihood models with scikit-learn."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

from metrics import accuracy_score, confusion_matrix
from naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "naive_bayes.png"


def evaluate(model, reference, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    reference.fit(X_train, y_train)
    predictions = model.predict(X_test)
    reference_predictions = reference.predict(X_test)
    return (
        accuracy_score(y_test, predictions),
        accuracy_score(y_test, reference_predictions),
        confusion_matrix(y_test, predictions),
    )


def main():
    wine = load_wine()
    wine_split = train_test_split(
        wine.data,
        wine.target,
        test_size=0.25,
        random_state=42,
        stratify=wine.target,
    )
    gaussian = evaluate(GaussianNB(), SklearnGaussianNB(), *wine_split)

    digits = load_digits()
    digit_split = train_test_split(
        digits.data,
        digits.target,
        test_size=0.25,
        random_state=42,
        stratify=digits.target,
    )
    multinomial = evaluate(
        MultinomialNB(alpha=1.0), SklearnMultinomialNB(alpha=1.0), *digit_split
    )
    bernoulli = evaluate(
        BernoulliNB(alpha=1.0, binarize=8.0),
        SklearnBernoulliNB(alpha=1.0, binarize=8.0),
        *digit_split,
    )

    results = [
        ("Gaussian NB — Wine", gaussian),
        ("Multinomial NB — Digits", multinomial),
        ("Bernoulli NB — Digits", bernoulli),
    ]
    print("\nNaive Bayes — held-out test sets")
    print(f"{'Model':<28} {'From scratch':>14} {'scikit-learn':>16}")
    print("-" * 60)
    for name, (our_accuracy, reference_accuracy, _) in results:
        print(f"{name:<28} {our_accuracy:>14.4f} {reference_accuracy:>16.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("Naive Bayes from Scratch — Likelihood Models", fontsize=15)

    for axis, (name, (accuracy, _, matrix)) in zip(axes, results):
        image = axis.imshow(matrix, cmap="Blues")
        if len(matrix) <= 3:
            for row, column in np.ndindex(matrix.shape):
                axis.text(
                    column,
                    row,
                    matrix[row, column],
                    ha="center",
                    va="center",
                )
        axis.set(
            xlabel="Predicted class",
            ylabel="True class",
            title=f"{name}\nAccuracy = {accuracy:.3f}",
            xticks=np.arange(len(matrix)),
            yticks=np.arange(len(matrix)),
        )
        figure.colorbar(image, ax=axis, fraction=0.046)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
