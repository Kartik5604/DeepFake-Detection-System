"""
predict.py

Runs inference on a single image or video file using a trained model
and prints whether the media is classified as real or deepfake, along
with a confidence score.

Usage:
    python model/predict.py --input path/to/media_file.jpg --model saved_models/best_model.h5
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model

from preprocessing.face_detector import detect_and_crop_face
from preprocessing.frame_extractor import extract_frames

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def preprocess_image(image, image_size=(128, 128)):
    """Face crop, resize, and scale a single BGR image for the model."""
    cropped = detect_and_crop_face(image)
    resized = cv2.resize(cropped, image_size)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_image(model, image_path, image_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not read image: {image_path}")

    batch = preprocess_image(image, image_size)
    prob = float(model.predict(batch, verbose=0)[0][0])
    return prob


def predict_video(model, video_path, image_size=(128, 128), sample_rate=15, max_frames=30):
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = extract_frames(video_path, tmp_dir, sample_rate=sample_rate, max_frames=max_frames)

        if not frame_paths:
            raise ValueError(f"No frames could be extracted from {video_path}")

        probs = []
        for frame_path in frame_paths:
            image = cv2.imread(frame_path)
            if image is None:
                continue
            batch = preprocess_image(image, image_size)
            probs.append(float(model.predict(batch, verbose=0)[0][0]))

        return float(np.mean(probs)) if probs else 0.5


def run_prediction(model_path, input_path, image_size=(128, 128)):
    model = load_model(model_path)
    ext = os.path.splitext(input_path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        prob = predict_image(model, input_path, image_size)
    elif ext in VIDEO_EXTENSIONS:
        prob = predict_video(model, input_path, image_size)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else (1 - prob)

    print(f"\nFile:       {input_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake detection on a media file")
    parser.add_argument("--input", required=True, help="Path to an image or video file")
    parser.add_argument("--model", default="saved_models/best_model.h5", help="Path to a trained model")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128])
    args = parser.parse_args()

    run_prediction(args.model, args.input, image_size=tuple(args.image_size))
