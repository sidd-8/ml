"""Compare AdaBoost and gradient boosting learning curves with scikit-learn."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBClassifier
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBRegressor
from sklearn.model_selection import train_test_split

from ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from metrics import accuracy_score, log_loss, r2_score


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "boosting.png"


def adaboost_results():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7, stratify=y
    )
    ours = AdaBoostClassifier(n_estimators=60, learning_rate=0.5).fit(
        X_train, y_train
    )
    reference = SklearnAdaBoostClassifier(
        n_estimators=60, learning_rate=0.5, random_state=0
    ).fit(X_train, y_train)
    our_scores = np.asarray(list(ours.staged_score(X_test, y_test)))
    reference_scores = np.asarray(list(reference.staged_score(X_test, y_test)))
    return ours, reference, our_scores, reference_scores


def classifier_results():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    settings = dict(
        n_estimators=60,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        random_state=42,
    )
    ours = GradientBoostingClassifier(**settings).fit(X_train, y_train)
    reference = SklearnGBClassifier(**settings).fit(X_train, y_train)
    our_losses = np.asarray(
        [log_loss(y_test, probability[:, 1]) for probability in ours.staged_predict_proba(X_test)]
    )
    reference_losses = np.asarray(
        [
            log_loss(y_test, probability[:, 1])
            for probability in reference.staged_predict_proba(X_test)
        ]
    )
    return ours, reference, our_losses, reference_losses, X_test, y_test


def regressor_results():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    settings = dict(
        n_estimators=60,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        random_state=42,
    )
    ours = GradientBoostingRegressor(**settings).fit(X_train, y_train)
    reference = SklearnGBRegressor(**settings).fit(X_train, y_train)
    our_scores = np.asarray(
        [r2_score(y_test, prediction) for prediction in ours.staged_predict(X_test)]
    )
    reference_scores = np.asarray(
        [
            r2_score(y_test, prediction)
            for prediction in reference.staged_predict(X_test)
        ]
    )
    return ours, reference, our_scores, reference_scores


def main():
    ada, reference_ada, ada_scores, reference_ada_scores = adaboost_results()
    gbc, reference_gbc, gbc_losses, reference_gbc_losses, X_test, y_test = (
        classifier_results()
    )
    gbr, reference_gbr, gbr_scores, reference_gbr_scores = regressor_results()

    print("\nBoosting — held-out comparisons")
    print(
        f"AdaBoost Wine accuracy: {ada_scores[-1]:.4f} "
        f"(sklearn={reference_ada_scores[-1]:.4f})"
    )
    print(
        f"Gradient Boosting cancer accuracy: {accuracy_score(y_test, gbc.predict(X_test)):.4f} "
        f"(sklearn={accuracy_score(y_test, reference_gbc.predict(X_test)):.4f})"
    )
    print(
        f"Gradient Boosting diabetes R2: {gbr_scores[-1]:.4f} "
        f"(sklearn={reference_gbr_scores[-1]:.4f})"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("Boosting Ensembles from Scratch", fontsize=15)

    axes[0].plot(
        np.arange(1, len(ada_scores) + 1),
        ada_scores,
        color="#2563eb",
        label="From scratch",
    )
    axes[0].plot(
        np.arange(1, len(reference_ada_scores) + 1),
        reference_ada_scores,
        "--",
        color="#dc2626",
        label="scikit-learn",
    )
    axes[0].set(
        xlabel="Weak learners",
        ylabel="Test accuracy",
        title="SAMME AdaBoost — Wine",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(gbc_losses, color="#7c3aed", label="From scratch")
    axes[1].plot(reference_gbc_losses, "--", color="#dc2626", label="scikit-learn")
    axes[1].set(
        xlabel="Boosting stage",
        ylabel="Test log loss",
        title="Gradient Boosting — Cancer",
    )
    axes[1].legend(frameon=False)

    axes[2].plot(gbr_scores, color="#059669", label="From scratch")
    axes[2].plot(reference_gbr_scores, "--", color="#dc2626", label="scikit-learn")
    axes[2].set(
        xlabel="Boosting stage",
        ylabel="Test R²",
        title="Gradient Boosting — Diabetes",
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
