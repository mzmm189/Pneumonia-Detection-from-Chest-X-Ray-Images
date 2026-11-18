"""
model.py

Defines the two models used in this project:

1. build_baseline_model()  — a simple CNN trained from scratch, used
   purely as a reference point (see report Section 5).
2. build_densenet_model()  — DenseNet121 pretrained on ImageNet, used
   as the main model via transfer learning (see report Section 6).
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.applications import DenseNet121

INPUT_SHAPE = (150, 150, 3)


def build_baseline_model(input_shape=INPUT_SHAPE):
    """
    Simple CNN baseline:
    Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten
    -> Dense(64) -> Dropout(0.5) -> Dense(1, sigmoid)
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_densenet_model(input_shape=INPUT_SHAPE, fine_tune_last_n=0, learning_rate=None):
    """
    DenseNet121 (ImageNet weights) as a frozen feature extractor, with a
    custom classification head on top:
        GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.5)
        -> Dense(1, sigmoid)

    Parameters
    ----------
    fine_tune_last_n : int
        If > 0, unfreezes the last N layers of the DenseNet121 base for
        fine-tuning (see report Section 7, Experiment 3). Use a small
        learning_rate (e.g. 1e-5) whenever this is > 0.
    learning_rate : float or None
        If provided, overrides the default Adam learning rate (1e-3).
    """
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = fine_tune_last_n > 0

    if fine_tune_last_n > 0:
        for layer in base_model.layers[:-fine_tune_last_n]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=learning_rate) if learning_rate else "adam"
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model
