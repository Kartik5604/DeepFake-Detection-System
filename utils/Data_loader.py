"""
data_loader.py

Utilities for loading a processed dataset (real/fake image folders)
into Keras-ready train, validation, and test generators.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
import shutil
import tempfile

from preprocessing.augmentation import get_augmentation_generator, get_validation_generator


def _split_dataset(processed_dir, classes, test_split, val_split, seed):
    """
    Build file lists split into train/val/test for each class, then
    materialize them into a temporary directory structure that
    Keras' flow_from_directory can consume.
    """
    tmp_root = tempfile.mkdtemp(prefix="dfd_split_")

    for split in ("train", "val", "test"):
        for cls in classes:
            os.makedirs(os.path.join(tmp_root, split, cls), exist_ok=True)

    for cls in classes:
        class_dir = os.path.join(processed_dir, cls)
        files = [
            f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        train_files, temp_files = train_test_split(
            files, test_size=(test_split + val_split), random_state=seed
        )
        relative_val_size = val_split / (test_split + val_split)
        val_files, test_files = train_test_split(
            temp_files, test_size=(1 - relative_val_size), random_state=seed
        )

        for split_name, split_files in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            for f in split_files:
                src = os.path.join(class_dir, f)
                dst = os.path.join(tmp_root, split_name, cls, f)
                shutil.copyfile(src, dst)

    return tmp_root


def load_datasets(config, dataset_name):
    """
    Load a processed dataset and return train, validation, and test
    Keras generators.

    Args:
        config (dict): Parsed dataset_config.yaml.
        dataset_name (str): Name of the dataset subfolder under
            data/processed/.

    Returns:
        tuple: (train_generator, val_generator, test_generator, tmp_dir)
            tmp_dir is returned so the caller can clean it up when done.
    """
    ds_cfg = config["dataset"]
    train_cfg = config["training"]
    aug_cfg = config["augmentation"]

    processed_dir = os.path.join(ds_cfg["processed_dir"], dataset_name)
    image_size = tuple(ds_cfg["image_size"])
    classes = ds_cfg["classes"]

    tmp_root = _split_dataset(
        processed_dir,
        classes,
        ds_cfg["test_split"],
        ds_cfg["val_split"],
        ds_cfg["seed"],
    )

    train_datagen = get_augmentation_generator(aug_cfg)
    eval_datagen = get_validation_generator()

    train_generator = train_datagen.flow_from_directory(
        os.path.join(tmp_root, "train"),
        target_size=image_size,
        batch_size=train_cfg["batch_size"],
        class_mode="binary",
        classes=classes,
    )

    val_generator = eval_datagen.flow_from_directory(
        os.path.join(tmp_root, "val"),
        target_size=image_size,
        batch_size=train_cfg["batch_size"],
        class_mode="binary",
        classes=classes,
        shuffle=False,
    )

    test_generator = eval_datagen.flow_from_directory(
        os.path.join(tmp_root, "test"),
        target_size=image_size,
        batch_size=train_cfg["batch_size"],
        class_mode="binary",
        classes=classes,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator, tmp_root


def cleanup_tmp_dir(tmp_dir):
    """Remove the temporary split directory created by load_datasets."""
    shutil.rmtree(tmp_dir, ignore_errors=True)
