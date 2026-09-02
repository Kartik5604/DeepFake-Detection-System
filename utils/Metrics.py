"""
metrics.py

Helpers for computing and reporting classification metrics
(accuracy, precision, recall, F1) and plotting training curves.
"""

import matplotlib
matplotlib.use("Agg")  # allow saving plots without a display
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred):
    """
    Compute standard binary classification metrics.

    Args:
        y_true (array-like): Ground truth labels (0/1).
        y_pred (array-like): Predicted labels (0/1).

    Returns:
        dict: accuracy, precision, recall, f1, and confusion matrix.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_classification_report(y_true, y_pred, target_names=("real", "fake")):
    """Print a full sklearn classification report to stdout."""
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_training_history(history, output_path="training_history.png"):
    """
    Plot training and validation accuracy/loss curves and save to disk.

    Args:
        history: Keras History object returned by model.fit().
        output_path (str): File path to save the resulting plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved training history plot to {output_path}")
