"""
cnn_model.py

Defines the CNN architecture used to classify media as real or
deepfake. A straightforward stack of convolution + pooling blocks
followed by dense layers, with dropout and batch normalization to
help generalization.
"""

from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(128, 128, 3), learning_rate=0.0001):
    """
    Build and compile the CNN classifier.

    Args:
        input_shape (tuple): Shape of input images (H, W, C).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    model = models.Sequential(name="deepfake_detector_cnn")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers_adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall"],
    )

    return model


def optimizers_adam(learning_rate):
    """Small wrapper so the optimizer import stays local to this file."""
    from tensorflow.keras.optimizers import Adam

    return Adam(learning_rate=learning_rate)


if __name__ == "__main__":
    m = build_cnn_model()
    m.summary()
