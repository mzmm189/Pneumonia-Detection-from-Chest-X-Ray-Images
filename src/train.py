"""
train.py

Trains both the baseline CNN and the main DenseNet121 transfer-learning
model on the Chest X-Ray Images (Pneumonia) dataset, and saves both
trained models plus their training-history plots.

Usage
-----
    python train.py --data_dir /path/to/chest_xray --epochs 10 --model both

    --model can be: baseline | densenet | both   (default: both)

Before running, either:
  1) manually download and unzip the dataset from Kaggle
     (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
     so that --data_dir points at the folder containing train/val/test, or
  2) place your kaggle.json in ~/.kaggle/ and let the script download it
     automatically with --download.
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_preprocessing import build_generators, count_images
from src.model import build_baseline_model, build_densenet_model


def maybe_download_dataset(data_dir):
    """Downloads and unzips the Kaggle dataset if data_dir does not exist yet."""
    if os.path.isdir(os.path.join(data_dir, "train")):
        return
    print(f"'{data_dir}/train' not found — attempting to download from Kaggle...")
    os.makedirs(data_dir, exist_ok=True)
    os.system(
        f"kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {data_dir}"
    )
    zip_path = os.path.join(data_dir, "chest-xray-pneumonia.zip")
    os.system(f"unzip -q {zip_path} -d {data_dir}")


def plot_history(history, title, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{title} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{title.replace(' ', '_')}_history.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia classification models.")
    parser.add_argument("--data_dir", type=str, default="./chest_xray_data/chest_xray",
                         help="Path to the dataset folder containing train/val/test.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="both", choices=["baseline", "densenet", "both"])
    parser.add_argument("--fine_tune_last_n", type=int, default=0,
                         help="If > 0, fine-tunes the last N DenseNet121 layers instead of using a frozen base.")
    parser.add_argument("--learning_rate", type=float, default=None,
                         help="Override the default Adam learning rate for the DenseNet model.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--download", action="store_true",
                         help="Download the dataset from Kaggle if not already present.")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.download:
        maybe_download_dataset(args.data_dir)

    print("=== Dataset class counts ===")
    count_images(args.data_dir)

    train_gen, val_gen, test_gen = build_generators(
        args.data_dir, batch_size=args.batch_size
    )

    if args.model in ("baseline", "both"):
        print("\n=== Training baseline CNN ===")
        baseline_model = build_baseline_model()
        history_baseline = baseline_model.fit(
            train_gen, validation_data=val_gen, epochs=args.epochs
        )
        baseline_model.save(os.path.join(args.out_dir, "baseline_model.h5"))
        plot_history(history_baseline, "Baseline_Model", args.out_dir)
        print("Baseline train/val accuracy:",
              history_baseline.history["accuracy"][-1],
              history_baseline.history["val_accuracy"][-1])

    if args.model in ("densenet", "both"):
        print("\n=== Training DenseNet121 (transfer learning) ===")
        densenet_model = build_densenet_model(
            fine_tune_last_n=args.fine_tune_last_n,
            learning_rate=args.learning_rate,
        )
        history_main = densenet_model.fit(
            train_gen, validation_data=val_gen, epochs=args.epochs
        )
        densenet_model.save(os.path.join(args.out_dir, "densenet_model.h5"))
        plot_history(history_main, "Main_Model_DenseNet121", args.out_dir)
        print("Main model train/val accuracy:",
              history_main.history["accuracy"][-1],
              history_main.history["val_accuracy"][-1])

    print(f"\nAll trained models and plots saved to: {args.out_dir}")
    print("Run evaluate.py next to compute test-set metrics.")


if __name__ == "__main__":
    main()
