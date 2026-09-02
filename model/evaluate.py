"""
evaluate.py

Loads a trained model and reports accuracy, precision, recall, F1
score, and a confusion matrix on a held out test set.

Usage:
    python model/evaluate.py --model saved_models/best_model.h5 --dataset my_dataset
"""

import os
import sys
import argparse
import yaml
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model

from utils.data_loader import load_datasets, cleanup_tmp_dir
from utils.metrics import compute_metrics, print_classification_report


def load_config(config_path="data/dataset_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model_path, dataset_name, config):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print("Loading test data...")
    _, _, test_gen, tmp_dir = load_datasets(config, dataset_name)

    try:
        print("Running predictions on the test set...")
        y_true = test_gen.classes
        y_probs = model.predict(test_gen).ravel()
        y_pred = (y_probs >= 0.5).astype(int)

        metrics = compute_metrics(y_true, y_pred)

        print("\n--- Evaluation Results ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"Confusion Matrix: {metrics['confusion_matrix']}")
        print()
        print_classification_report(y_true, y_pred, target_names=config["dataset"]["classes"])

        return metrics

    finally:
        cleanup_tmp_dir(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained deepfake detection model")
    parser.add_argument("--model", required=True, help="Path to a saved .h5 model")
    parser.add_argument("--dataset", required=True, help="Name of dataset under data/processed/")
    parser.add_argument("--config", default="data/dataset_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(args.model, args.dataset, cfg)
