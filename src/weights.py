from __future__ import annotations
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def compute_balanced_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y
    )
    return {int(i): float(w) for i, w in enumerate(weights)}
