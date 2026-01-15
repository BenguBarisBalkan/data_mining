from __future__ import annotations
import os
import numpy as np
from typing import Callable, List, Optional, Set, Tuple
from sklearn.preprocessing import StandardScaler
from .preprocess import load_and_clean_csv
from .sequences import df_to_sequences
from .scaling import scale_X

import numpy as np
from typing import Callable, Tuple

def build_dataset_from_files(
    files: List[str],
    feature_cols: List[str],
    seq_len: int,
    allowed_patterns: Set[str],
    label_fn: Callable[[str], Optional[int]],
    scaler: Optional[StandardScaler] = None,
    min_group_len: int = 1,
    max_seqs_per_file: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for path in files:
        df = load_and_clean_csv(path, feature_cols=feature_cols, allowed_patterns=allowed_patterns)
        X, y = df_to_sequences(
            df,
            feature_cols=feature_cols,
            seq_len=seq_len,
            label_fn=label_fn,
            min_group_len=min_group_len,
            max_seqs=max_seqs_per_file,
        )
        if X is None:
            continue

        if scaler is not None:
            X = scale_X(X, scaler)

        X_list.append(X)
        y_list.append(y)

    if len(X_list) == 0:
        raise ValueError("No sequences found in the provided files.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all

def flatten_for_svm(X: np.ndarray) -> np.ndarray:
    N, T, F = X.shape
    return X.reshape(N, T * F)


def make_load_xy_fn(
    *,
    feature_cols: list[str],
    seq_len: int,
    class_to_id: dict[str, int],
    scaler,
    junk_cols: set[str],
    required_cols: set[str],
    allowed_patterns: set[str] | None = None,
    min_group_len: int = 1,
    max_seqs_per_file: int | None = None,
    drop_no_pattern: bool = True,
    pad_value: float = 0.0,
):
    """
    Factory that returns a function:
        load_xy_fn(path) -> (X, y)
    """

    if allowed_patterns is None:
        allowed_patterns = set(class_to_id.keys()) | {"no-pattern"}

    def load_xy_fn(path: str):
        # IMPORTANT: use the correct module names you actually have
        from .preprocess import load_and_clean_csv
        from .sequences import df_to_sequences

        df = load_and_clean_csv(
            path,
            feature_cols=feature_cols,
            allowed_patterns=allowed_patterns,
            junk_cols=junk_cols,
            required_cols=required_cols,
        )

        X, y = df_to_sequences(
            df,
            feature_cols=feature_cols,
            seq_len=seq_len,
            class_to_id=class_to_id,
            min_group_len=min_group_len,
            max_seqs_per_file=max_seqs_per_file,
            drop_no_pattern=drop_no_pattern,
            pad_value=pad_value,
        )

        if X is None:
            return None, None

        # scale
        s, t, f = X.shape
        X2 = X.reshape(-1, f)
        X2 = scaler.transform(X2)
        X = X2.reshape(s, t, f).astype(np.float32)

        return X, y

    return load_xy_fn
