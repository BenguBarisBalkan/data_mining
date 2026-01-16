from __future__ import annotations
import numpy as np
from typing import Callable, Optional

def train_global_passes(
    model,
    train_files: list[str],
    load_xy_fn: Callable[[str], tuple[np.ndarray | None, np.ndarray | None]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    class_weight: dict[int, float] | None = None,
    global_passes: int = 20,
    epochs_per_file: int = 1,
    batch_size: int = 64,
    patience: int = 3,
    min_delta: float = 1e-4,
    seed: int = 42,
    verbose: int = 0,
):
    """
    Manual early stopping across global passes.
    load_xy_fn(path) must return (X,y) already scaled/shaped.
    """
    rng = np.random.default_rng(seed)

    best_val_loss = np.inf
    patience_counter = 0
    best_weights = None

    for gp in range(global_passes):
        files_shuffled = train_files.copy()
        rng.shuffle(files_shuffled)
        print(f"\n=== Global pass {gp+1}/{global_passes} ===")

        for path in files_shuffled:
            X, y = load_xy_fn(path)
            if X is None or y is None:
                continue

            idx = rng.permutation(len(X))
            X, y = X[idx], y[idx]

            model.fit(
                X, y,
                epochs=epochs_per_file,
                batch_size=batch_size,
                verbose=verbose,
                class_weight=class_weight
            )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation -> loss: {val_loss:.4f} | acc: {val_acc:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.get_weights()
            print("Improved val_loss. Saved best weights.")
        else:
            patience_counter += 1
            print(f"No improvement. patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_weights is not None:
        model.set_weights(best_weights)
        print("Restored best model weights from validation.")

    return model
