from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model


def dilated_conv_block(x, filters: int, kernel_size: int, dilation: int,
                       dropout: float = 0.0, name: str | None = None):
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv_k{kernel_size}_d{dilation}",
    )(x)
    x = layers.BatchNormalization(
        name=None if name is None else f"{name}_bn_k{kernel_size}_d{dilation}"
    )(x)
    x = layers.ReLU(
        name=None if name is None else f"{name}_relu_k{kernel_size}_d{dilation}"
    )(x)
    if dropout and dropout > 0:
        x = layers.Dropout(
            dropout, name=None if name is None else f"{name}_drop_k{kernel_size}_d{dilation}"
        )(x)
    return x


def module_7(x, filters: int = 64, dropout: float = 0.0, name: str = "m7"):
    x = dilated_conv_block(x, filters, 3, 1, dropout, name)
    x = dilated_conv_block(x, filters, 3, 2, dropout, name)
    return x


def module_10(x, filters: int = 64, dropout: float = 0.0, name: str = "m10"):
    x = dilated_conv_block(x, filters, 3, 1, dropout, name)
    x = dilated_conv_block(x, filters, 3, 2, dropout, name)
    x = dilated_conv_block(x, filters, 2, 4, dropout, name)
    return x


def module_15(x, filters: int = 64, dropout: float = 0.0, name: str = "m15"):
    x = dilated_conv_block(x, filters, 3, 1, dropout, name)
    x = dilated_conv_block(x, filters, 3, 2, dropout, name)
    x = dilated_conv_block(x, filters, 3, 4, dropout, name)
    return x


def build_liu_si_cnn_bilstm(
    *,
    seq_len: int,
    num_features: int,
    num_classes: int,
    filters: int = 64,
    module_dropout: float = 0.1,
    head_dropout: float = 0.3,
    lstm_units: int = 64,
    bidirectional: bool = True,
    modules: Sequence[str] = ("m7", "m10", "m15"), 
) -> Model:
    """
    CNN + (Bi)LSTM hybrid.
    modules: choose from {"m7","m10","m15"} to run ablations.
      e.g. ("m7","m15") => without module10
    """

    modules = tuple(modules)
    valid = {"m7", "m10", "m15"}
    if not set(modules).issubset(valid):
        raise ValueError(f"modules must be subset of {valid}, got {modules}")
    if len(modules) < 1:
        raise ValueError("modules must contain at least one module.")

    inp = layers.Input(shape=(seq_len, num_features), name="input")

    branches = []
    if "m7" in modules:
        branches.append(module_7(inp, filters=filters, dropout=module_dropout, name="module7"))
    if "m10" in modules:
        branches.append(module_10(inp, filters=filters, dropout=module_dropout, name="module10"))
    if "m15" in modules:
        branches.append(module_15(inp, filters=filters, dropout=module_dropout, name="module15"))

    # If only one module is used, skip Concatenate
    x = branches[0] if len(branches) == 1 else layers.Concatenate(name="concat")(branches)

    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

    # 1x1 conv to mix channels
    x = layers.Conv1D(filters=filters, kernel_size=1, padding="same",
                      use_bias=False, name="conv1x1")(x)
    x = layers.BatchNormalization(name="conv1x1_bn")(x)
    x = layers.ReLU(name="conv1x1_relu")(x)

    x = layers.Dropout(head_dropout, name="head_dropout")(x)

    # LSTM while time dimension exists
    if bidirectional:
        x = layers.Bidirectional(layers.LSTM(lstm_units), name="bilstm")(x)
    else:
        x = layers.LSTM(lstm_units, name="lstm")(x)

    # Classifier head
    x = layers.Dropout(0.3, name="post_lstm_dropout")(x)
    x = layers.Dense(64, activation="relu", name="fc")(x)
    x = layers.Dropout(0.3, name="fc_dropout")(x)

    out = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model_name = "LiuSi_CNN_BiLSTM_" + "_".join(modules)
    return Model(inp, out, name=model_name)
