"""
train.py

Trains the CNN deepfake classifier on a processed dataset, following
an iterative test, train, and evaluate cycle with early stopping and
checkpointing on validation accuracy.

Usage:
    python model/train.py --dataset my_dataset --epochs 30 --batch_size 32
"""

import os
import sys
import argparse
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model.cnn_model import build_cnn_model
from utils.data_loader import load_datasets, cleanup_tmp_dir
from utils.metrics import plot_training_history


def load_config(config_path="data/dataset_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(dataset_name, config, epochs=None, batch_size=None, output_model_path="saved_models/best_model.h5"):
    ds_cfg = config["dataset"]
    train_cfg = config["training"]

    if epochs is not None:
        train_cfg["epochs"] = epochs
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size

    print("Loading datasets...")
    train_gen, val_gen, test_gen, tmp_dir = load_datasets(config, dataset_name)

    try:
        input_shape = tuple(ds_cfg["image_size"]) + (3,)
        model = build_cnn_model(input_shape=input_shape, learning_rate=train_cfg["learning_rate"])
        model.summary()

        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
            ModelCheckpoint(output_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ]

        print(f"Starting training for up to {train_cfg['epochs']} epochs...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=train_cfg["epochs"],
            callbacks=callbacks,
        )

        plot_training_history(history, output_path="training_history.png")

        print("Evaluating on the held out test set...")
        test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)
        print(f"Test accuracy: {test_acc:.4f} | precision: {test_prec:.4f} | recall: {test_rec:.4f}")

        print(f"Best model saved to {output_model_path}")

    finally:
        cleanup_tmp_dir(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the deepfake detection CNN")
    parser.add_argument("--dataset", required=True, help="Name of dataset under data/processed/")
    parser.add_argument("--config", default="data/dataset_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", default="saved_models/best_model.h5")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(args.dataset, cfg, epochs=args.epochs, batch_size=args.batch_size, output_model_path=args.output)
