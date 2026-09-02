"""
face_detector.py

Detects and crops facial regions from images using OpenCV's built in
Haar Cascade classifier. Deepfake artifacts are usually concentrated
around the face, so cropping to the face improves signal to noise
for the classifier.
"""

import os
import cv2

# Load OpenCV's bundled Haar Cascade for frontal face detection
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_and_crop_face(image, margin=0.2):
    """
    Detect the largest face in an image and return a cropped version.
    Falls back to the full image if no face is found.

    Args:
        image (np.ndarray): Input image in BGR format.
        margin (float): Extra margin around the detected face box,
            expressed as a fraction of the face box size.

    Returns:
        np.ndarray: Cropped face image, or the original image if no
            face was detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) == 0:
        return image

    # Pick the largest detected face box
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(image.shape[1], x + w + mx)
    y2 = min(image.shape[0], y + h + my)

    return image[y1:y2, x1:x2]


def process_directory(input_dir, output_dir, margin=0.2):
    """
    Run face detection and cropping on every image in `input_dir`
    and save the results to `output_dir`.

    Args:
        input_dir (str): Directory of source images.
        output_dir (str): Directory to save cropped face images.
        margin (float): Margin passed to detect_and_crop_face.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        cropped = detect_and_crop_face(image, margin=margin)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped)

    print(f"Processed images from {input_dir} into {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect and crop faces from images")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.margin)
