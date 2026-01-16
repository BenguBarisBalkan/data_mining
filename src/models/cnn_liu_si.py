from __future__ import annotations

from tensorflow.keras import layers, Model


def dilated_causal_block(
    x,
    filters: int,
    kernel_size: int,
    dilation: int,
    dropout: float = 0.0,
    name: str | None = None,
):
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="causal",  # causal conv
        activation=None,
        name=None if name is None else f"{name}_conv_k{kernel_size}_d{dilation}",
    )(x)
    x = layers.ReLU(name=None if name is None else f"{name}_relu_k{kernel_size}_d{dilation}")(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout, name=None if name is None else f"{name}_drop_k{kernel_size}_d{dilation}")(x)
    return x


def module_7(x, filters: int = 64, dropout: float = 0.0, name: str = "m7"):
    # Fig.7 Module(7): k=3 d=1 -> k=3 d=2
    x = dilated_causal_block(x, filters, kernel_size=3, dilation=1, dropout=dropout, name=name)
    x = dilated_causal_block(x, filters, kernel_size=3, dilation=2, dropout=dropout, name=name)
    return x


def module_10(x, filters: int = 64, dropout: float = 0.0, name: str = "m10"):
    # Fig.7 Module(10): k=3 d=1 -> k=3 d=2 -> k=2 d=4
    x = dilated_causal_block(x, filters, kernel_size=3, dilation=1, dropout=dropout, name=name)
    x = dilated_causal_block(x, filters, kernel_size=3, dilation=2, dropout=dropout, name=name)
    x = dilated_causal_block(x, filters, kernel_size=2, dilation=4, dropout=dropout, name=name)
    return x


def module_15(x, filters: int = 64, dropout: float = 0.0, name: str = "m15"):
    # Fig.7 Module(15): k=3 d=1 -> k=3 d=2 -> k=3 d=4
    x = dilated_causal_block(x, filters, kernel_size=3, dilation=1, dropout=dropout, name=name)
    x = dilated_causal_block(x, filters, kernel_size=3, dilation=2, dropout=dropout, name=name)
    x = dilated_causal_block(x, filters, kernel_size=3, dilation=4, dropout=dropout, name=name)
    return x


def build_liu_si_cnn(
    *,
    seq_len: int,
    num_features: int,
    num_classes: int,
    filters: int = 64,
    module_dropout: float = 0.0,
    head_dropout: float = 0.3,
) -> Model:
    
    inp = layers.Input(shape=(seq_len, num_features), name="input")

    b7 = module_7(inp, filters=filters, dropout=module_dropout, name="module7")
    b10 = module_10(inp, filters=filters, dropout=module_dropout, name="module10")
    b15 = module_15(inp, filters=filters, dropout=module_dropout, name="module15")

    x = layers.Concatenate(name="concat")([b7, b10, b15])

    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

    x = layers.Conv1D(filters=filters, kernel_size=1, padding="same", name="conv1x1")(x)
    x = layers.ReLU(name="conv1x1_relu")(x)

    x = layers.Dropout(head_dropout, name="head_dropout")(x)
    x = layers.Flatten(name="flatten")(x)

    out = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    return Model(inp, out, name="LiuSi_1DCNN")
