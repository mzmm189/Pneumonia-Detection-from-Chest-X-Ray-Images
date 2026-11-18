"""
data_preprocessing.py

Builds the training, validation, and test data generators for the
Chest X-Ray Images (Pneumonia) dataset (Kermany / Mooney, Kaggle).

The dataset is expected to already be downloaded and unzipped locally,
with the standard folder structure:

    <data_dir>/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150, 150)
BATCH_SIZE = 32


def count_images(data_dir):
    """
    Prints the number of NORMAL and PNEUMONIA images in each split
    (train / val / test). Useful for confirming class imbalance
    before training (see the EDA section of the project report).
    """
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        normal = len(os.listdir(os.path.join(split_dir, "NORMAL")))
        pneumonia = len(os.listdir(os.path.join(split_dir, "PNEUMONIA")))
        counts[split] = {"NORMAL": normal, "PNEUMONIA": pneumonia}
        print(f"{split}: NORMAL={normal}, PNEUMONIA={pneumonia}")
    return counts


def build_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Returns (train_generator, val_generator, test_generator).

    - Training data is rescaled and augmented (rotation, zoom, horizontal
      flip) to increase diversity and reduce overfitting, since the
      dataset is relatively small and imbalanced.
    - Validation and test data are rescaled only (no augmentation), so
      evaluation reflects real, unmodified images.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    val_generator = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    test_generator = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,  # keep order aligned with true_classes for evaluation
    )

    return train_generator, val_generator, test_generator
