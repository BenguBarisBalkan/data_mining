from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_baseline(
    *,
    seq_len: int,
    num_features: int,
    num_classes: int,
    lstm_units: int = 64,
    dense_units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    """
    Simple LSTM baseline for sequence classification.
    (Kept intentionally small/simpler as a baseline.)

    Returns:
        (model, optimizer_kwargs) style not needed; just compiled model.
    """
    model = Sequential(
        [
            Input(shape=(seq_len, num_features)),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dropout(dropout),
            Dense(num_classes, activation="softmax"),
        ],
        name="LSTM_Baseline",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
