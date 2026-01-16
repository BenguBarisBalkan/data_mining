from __future__ import annotations

from sklearn.svm import LinearSVC
import numpy as np


def build_linear_svm(
    *,
    C: float = 1.0,
    class_weight: str | dict = "balanced",
    max_iter: int = 20000,
    random_state: int = 42,
):
    """
    Linear SVM baseline for time-series classification.

    NOTE:
    - Expects FLATTENED inputs of shape (N, T*F)
    - Scaling must be done beforehand
    """
    return LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )


def train_and_predict(
    svm: LinearSVC,
    X_train_flat: np.ndarray,
    y_train: np.ndarray,
    X_eval_flat: np.ndarray,
) -> np.ndarray:
    """
    Fit SVM and return predictions on evaluation set.
    """
    svm.fit(X_train_flat, y_train)
    y_pred = svm.predict(X_eval_flat)
    return y_pred
