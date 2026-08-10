"""Visualize random forest convergence, OOB scores, and feature importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier as SklearnForestClassifier
from sklearn.ensemble import RandomForestRegressor as SklearnForestRegressor
from sklearn.model_selection import train_test_split

from metrics import accuracy_score, r2_score
from random_forest import RandomForestClassifier, RandomForestRegressor


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "random_forest.png"
N_TREES = 40


def cumulative_classifier_scores(forest, X, y):
    probability_sum = np.zeros((len(X), len(forest.classes_)))
    scores = []
    for count, tree in enumerate(forest.estimators_, start=1):
        tree_probabilities = tree.predict_proba(X)
        positions = np.searchsorted(forest.classes_, tree.classes_)
        probability_sum[:, positions] += tree_probabilities
        predictions = forest.classes_[np.argmax(probability_sum, axis=1)]
        scores.append(accuracy_score(y, predictions))
    return np.asarray(scores)


def cumulative_sklearn_classifier_scores(forest, X, y):
    probability_sum = np.zeros((len(X), len(forest.classes_)))
    scores = []
    for tree in forest.estimators_:
        tree_probabilities = tree.predict_proba(X)
        positions = np.searchsorted(forest.classes_, tree.classes_)
        probability_sum[:, positions] += tree_probabilities
        predictions = forest.classes_[np.argmax(probability_sum, axis=1)]
        scores.append(accuracy_score(y, predictions))
    return np.asarray(scores)


def cumulative_regressor_scores(forest, X, y):
    prediction_sum = np.zeros(len(X))
    scores = []
    for count, tree in enumerate(forest.estimators_, start=1):
        prediction_sum += tree.predict(X)
        scores.append(r2_score(y, prediction_sum / count))
    return np.asarray(scores)


def main():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data,
        cancer.target,
        test_size=0.2,
        random_state=42,
        stratify=cancer.target,
    )
    classifier = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=7,
        min_samples_leaf=2,
        oob_score=True,
        random_state=42,
    ).fit(X_train, y_train)
    reference_classifier = SklearnForestClassifier(
        n_estimators=N_TREES,
        max_depth=7,
        min_samples_leaf=2,
        oob_score=True,
        random_state=42,
    ).fit(X_train, y_train)
    classifier_scores = cumulative_classifier_scores(classifier, X_test, y_test)
    reference_classifier_scores = cumulative_sklearn_classifier_scores(
        reference_classifier, X_test, y_test
    )

    diabetes = load_diabetes()
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )
    regressor = RandomForestRegressor(
        n_estimators=N_TREES,
        max_depth=7,
        min_samples_leaf=3,
        oob_score=True,
        random_state=42,
    ).fit(X_train_reg, y_train_reg)
    reference_regressor = SklearnForestRegressor(
        n_estimators=N_TREES,
        max_depth=7,
        min_samples_leaf=3,
        oob_score=True,
        random_state=42,
    ).fit(X_train_reg, y_train_reg)
    regressor_scores = cumulative_regressor_scores(
        regressor, X_test_reg, y_test_reg
    )
    reference_regressor_scores = cumulative_regressor_scores(
        reference_regressor, X_test_reg, y_test_reg
    )

    print("\nRandom Forest — held-out and out-of-bag comparisons")
    print(
        f"Cancer accuracy: {classifier_scores[-1]:.4f} "
        f"(sklearn={reference_classifier_scores[-1]:.4f})"
    )
    print(
        f"Cancer OOB accuracy: {classifier.oob_score_:.4f} "
        f"(sklearn={reference_classifier.oob_score_:.4f})"
    )
    print(
        f"Diabetes R2: {regressor_scores[-1]:.4f} "
        f"(sklearn={reference_regressor_scores[-1]:.4f})"
    )
    print(
        f"Diabetes OOB R2: {regressor.oob_score_:.4f} "
        f"(sklearn={reference_regressor.oob_score_:.4f})"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("Random Forest from Scratch", fontsize=15)
    tree_counts = np.arange(1, N_TREES + 1)

    axes[0].plot(tree_counts, classifier_scores, color="#2563eb", label="From scratch")
    axes[0].plot(
        tree_counts,
        reference_classifier_scores,
        "--",
        color="#dc2626",
        label="scikit-learn",
    )
    axes[0].set(
        xlabel="Number of trees",
        ylabel="Test accuracy",
        title="Breast cancer classification",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(tree_counts, regressor_scores, color="#059669", label="From scratch")
    axes[1].plot(
        tree_counts,
        reference_regressor_scores,
        "--",
        color="#dc2626",
        label="scikit-learn",
    )
    axes[1].set(
        xlabel="Number of trees",
        ylabel="Test R²",
        title="Diabetes regression",
    )
    axes[1].legend(frameon=False)

    top_indices = np.argsort(classifier.feature_importances_)[-8:]
    axes[2].barh(
        np.asarray(cancer.feature_names)[top_indices],
        classifier.feature_importances_[top_indices],
        color="#7c3aed",
    )
    axes[2].set(
        xlabel="Mean normalized impurity reduction",
        title="Cancer feature importances",
    )

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
