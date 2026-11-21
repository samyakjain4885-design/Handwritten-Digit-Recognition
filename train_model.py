"""Train a small CNN on MNIST and save it for the demo app."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "mnist_digit_model.keras"
CUSTOM_SAMPLES_PATH = ROOT / "custom_digit_samples"


def load_custom_samples() -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    if not CUSTOM_SAMPLES_PATH.exists():
        return np.empty((0, 28, 28, 1), dtype="float32"), np.empty((0,), dtype="int64")

    for image_path in sorted(CUSTOM_SAMPLES_PATH.glob("*/*.png")):
        try:
            label = int(image_path.parent.name)
        except ValueError:
            continue
        if not 0 <= label <= 9:
            continue
        with Image.open(image_path) as image:
            grayscale = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
            pixels = np.asarray(grayscale, dtype="float32") / 255.0
        images.append(pixels[..., None])
        labels.append(label)

    if not images:
        return np.empty((0, 28, 28, 1), dtype="float32"), np.empty((0,), dtype="int64")
    return np.stack(images), np.asarray(labels, dtype="int64")


def main() -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    # Keep validation examples from MNIST only so custom feedback samples never leak into validation.
    validation_size = 6000
    x_validation = x_train[-validation_size:]
    y_validation = y_train[-validation_size:]
    x_train = x_train[:-validation_size]
    y_train = y_train[:-validation_size]

    custom_x, custom_y = load_custom_samples()
    if len(custom_x):
        # Reuse feedback examples with fresh random augmentation during training.
        repeat_factor = 20
        x_train = np.concatenate([x_train, np.repeat(custom_x, repeat_factor, axis=0)])
        y_train = np.concatenate([y_train, np.repeat(custom_y, repeat_factor, axis=0)])
    print(f"Custom correction samples: {len(custom_x)}")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=0.04,
                fill_mode="constant",
                fill_value=0.0,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="constant",
                fill_value=0.0,
            ),
        ],
        name="training_augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            augmentation,
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=8,
        batch_size=128,
        verbose=1,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
