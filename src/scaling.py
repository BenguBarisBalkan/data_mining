import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Iterable, Tuple

def fit_global_scaler(X_batches: Iterable[np.ndarray]) -> StandardScaler:
    """
    Fit scaler on training data only. Ignores padded rows that are all-zeros.
    X_batches yields arrays shaped (N, T, F).
    """
    scaler = StandardScaler()
    for X in X_batches:
        Xp = X.reshape(-1, X.shape[-1])
        nonpad = ~(np.all(Xp == 0, axis=1))
        if np.any(nonpad):
            scaler.partial_fit(Xp[nonpad])
    return scaler

def scale_X(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, T, F = X.shape
    X2 = X.reshape(-1, F)
    X2 = scaler.transform(X2)
    return X2.reshape(N, T, F).astype(np.float32)
