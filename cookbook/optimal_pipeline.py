from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

RANDOM_STATE: int = 42
N_SAMPLES: int = 1000
IMAGE_SIZE: Tuple[int, int] = (48, 48)
EPOCHS: int = 15
BATCH_SIZE: int = 64
TEST_RATIO: float = 0.2
VAL_RATIO: float = 0.2


def download_dataset(n: int = N_SAMPLES, repo_id: str = "rixmape/utkface", filename: str = "data/train-00000-of-00001.parquet", repo_type: str = "dataset") -> pd.DataFrame:
    """Download dataset from HuggingFace."""
    filepath: str = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    df: pd.DataFrame = pd.read_parquet(filepath)
    return df.sample(n=n, random_state=RANDOM_STATE) if n > 0 else df


def prepare_images(df: pd.DataFrame, image_size: Tuple[int, int] = IMAGE_SIZE, color_mode: str = "L") -> Tuple[np.ndarray, np.ndarray]:
    """Prepare images and labels from a dataframe."""
    processed = [((np.array(Image.open(BytesIO(row["image"]["bytes"])).convert(color_mode).resize(image_size), dtype=np.float32) / 255.0).reshape(image_size[0], image_size[1], 1), row["gender"]) for _, row in df.iterrows()]
    images, labels = zip(*processed)
    return np.array(images), np.array(labels)


def build_vgg16_model(input_shape: Tuple[int, int, int] = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)) -> tf.keras.Sequential:
    """Build a VGG16-like model."""
    model: tf.keras.Sequential = tf.keras.Sequential()
    blocks: List[Tuple[str, int, int]] = [("block1", 64, 2), ("block2", 128, 2), ("block3", 256, 3), ("block4", 512, 3), ("block5", 512, 3)]
    for block, filters, conv_count in blocks:
        for i in range(1, conv_count + 1):
            kwargs = {"filters": filters, "kernel_size": (3, 3), "activation": "relu", "padding": "same", "name": f"{block}_conv{i}"}
            if block == "block1" and i == 1:
                kwargs["input_shape"] = input_shape
            model.add(tf.keras.layers.Conv2D(**kwargs))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"{block}_pool"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def split_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into training, validation, and test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y)
    val_size_adjusted: float = VAL_RATIO / (1 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> tf.keras.Model:
    """Train the model using provided training and validation sets."""
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return model


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """Evaluate the model."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    return loss, accuracy


def generate_cam(model: tf.keras.Model, image: np.ndarray, label: int, target_layer: str = "block3_conv3") -> np.ndarray:
    """Generate Grad-CAM++ for an image."""
    modifier_fn = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
    score_fn = lambda output: output[0][label]
    visualizer = GradcamPlusPlus(model, model_modifier=modifier_fn, clone=True)
    return visualizer(score_fn, image[np.newaxis, ...], penultimate_layer=target_layer)[0]


def generate_activation_maps(model: tf.keras.Model, images: np.ndarray, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Generate activation maps for a batch of images."""
    return [{"image": np.squeeze(image), "activation": generate_cam(model, image, int(label)), "label": label} for image, label in zip(images[:32], labels[:32])]


def display_activation_grid(results: List[Dict[str, Any]], rows: int = 4, cols: int = 8) -> None:
    """Display a grid of activation maps."""
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    for i, result in enumerate(results[: rows * cols]):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(result["image"], cmap="gray")
        ax.imshow(result["activation"], cmap="jet", alpha=0.5)
        ax.axis("off")
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.tight_layout()
    plt.savefig("activation_maps.png")
    plt.show()


def run_pipeline() -> None:
    """Run the complete pipeline."""
    df = download_dataset()
    X, y = prepare_images(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    model = build_vgg16_model()
    model = train_model(model, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test)
    maps = generate_activation_maps(model, X_train, y_train)
    display_activation_grid(maps)


if __name__ == "__main__":
    run_pipeline()
