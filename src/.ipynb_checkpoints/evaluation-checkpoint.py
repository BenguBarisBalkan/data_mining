from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

def build_eval_dataset(
    eval_files: list[str],
    load_xy_fn,
):
    X_list, y_list = [], []
    for path in eval_files:
        X, y = load_xy_fn(path)
        if X is None or y is None:
            continue
        X_list.append(X)
        y_list.append(y)

    if len(X_list) == 0:
        raise ValueError("No sequences found in eval files.")

    X_eval = np.concatenate(X_list, axis=0)
    y_eval = np.concatenate(y_list, axis=0)
    return X_eval, y_eval


def evaluate_model_multiclass(
    model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    class_names: list[str],
):
    y_probs = model.predict(X_eval, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    report = classification_report(
        y_eval,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(
        y_eval,
        y_pred,
        labels=np.arange(len(class_names))
    )
    return report, cm


def check_train_eval_overlap(train_files: list[str], eval_files: list[str]) -> set[str]:
    train_set = {Path(p).name for p in train_files}
    eval_set  = {Path(p).name for p in eval_files}
    return train_set.intersection(eval_set)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()
