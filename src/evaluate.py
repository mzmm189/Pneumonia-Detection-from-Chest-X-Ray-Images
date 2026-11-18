"""
evaluate.py

Loads a trained model (.h5) and evaluates it on the test set:
  - classification_report (precision, recall, F1 per class)
  - confusion matrix (printed + saved as a heatmap image)
  - error analysis: saves example misclassified images (False Positives
    and False Negatives) with the model's confidence score for each.

Usage
-----
    python evaluate.py --model_path outputs/densenet_model.h5 \
                        --data_dir /path/to/chest_xray --out_dir outputs
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from src.data_preprocessing import build_generators

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def plot_confusion_matrix(cm, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def show_error_images(indices, predictions, filenames, data_dir, split, title, out_dir, max_images=6):
    n = min(len(indices), max_images)
    if n == 0:
        print(f"No errors of type: {title}")
        return

    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices[:n]):
        img_path = os.path.join(data_dir, split, filenames[idx])
        img = plt.imread(img_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Confidence: {predictions[idx][0]:.2f}", fontsize=10)
        plt.axis("off")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{title.replace(' ', '_').replace(':', '')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {n} example images for '{title}' to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained pneumonia classification model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .h5 model.")
    parser.add_argument("--data_dir", type=str, default="./chest_xray_data/chest_xray",
                         help="Path to the dataset folder containing train/val/test.")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path)

    _, _, test_gen = build_generators(args.data_dir, batch_size=args.batch_size)

    print("Running predictions on the test set...")
    predictions = model.predict(test_gen)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_gen.classes
    filenames = test_gen.filenames

    print("\n=== Classification Report ===")
    report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES)
    print(report)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:\n", cm)
    plot_confusion_matrix(cm, os.path.join(args.out_dir, "confusion_matrix.png"))

    # ---- Error analysis ----
    false_positives = np.where((predicted_classes == 1) & (true_classes == 0))[0]
    false_negatives = np.where((predicted_classes == 0) & (true_classes == 1))[0]

    total = len(true_classes)
    print(f"\nFalse Positives: {len(false_positives)} ({len(false_positives) / total * 100:.2f}% of test set)")
    print(f"False Negatives: {len(false_negatives)} ({len(false_negatives) / total * 100:.2f}% of test set)")

    show_error_images(false_positives, predictions, filenames, args.data_dir, "test",
                       "False Positives", args.out_dir)
    show_error_images(false_negatives, predictions, filenames, args.data_dir, "test",
                       "False Negatives", args.out_dir)

    print(f"\nAll evaluation outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
