from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple, List

def df_to_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    seq_len: int,
    class_to_id: dict[str, int],
    min_group_len: int = 1,
    max_seqs_per_file: int | None = None,
    drop_no_pattern: bool = True,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Converts a labeled OHLC dataframe into fixed-length sequences by:
    - grouping consecutive identical pattern labels into segments
    - for each segment, pad/truncate to seq_len
    - output X: (n_seq, seq_len, n_features), y: (n_seq,)
    """

    if drop_no_pattern:
        df = df[df["pattern"] != "no-pattern"].copy()

    if len(df) == 0:
        return None, None

    # Group consecutive identical patterns
    df["group_id"] = (df["pattern"] != df["pattern"].shift(1)).cumsum()

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _, g in df.groupby("group_id"):
        label = str(g["pattern"].iloc[0])
        if label not in class_to_id:
            continue

        feat = g[feature_cols].values.astype(np.float32)

        if len(feat) < min_group_len:
            continue

        # fixed-length window via pad/truncate
        if len(feat) >= seq_len:
            seq = feat[-seq_len:]
        else:
            pad_len = seq_len - len(feat)
            pad = np.full((pad_len, feat.shape[1]), pad_value, dtype=np.float32)
            seq = np.concatenate([pad, feat], axis=0)

        X_list.append(seq)
        y_list.append(class_to_id[label])

        if max_seqs_per_file is not None and len(X_list) >= max_seqs_per_file:
            break

    if len(X_list) == 0:
        return None, None

    X = np.stack(X_list)  # (n_seq, seq_len, n_feat)
    y = np.array(y_list, dtype=np.int64)
    return X, y
