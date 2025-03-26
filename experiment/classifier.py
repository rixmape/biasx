"""Model training for gender classification."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from config import ClassifierConfig
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """Handles training of gender classification models."""

    def __init__(self, config: ClassifierConfig):
        """Initialize with model configuration."""
        self.config = config
        self.model = self._build_model()
        self.history = None
        self.metrics = None

    def _build_model(self) -> tf.keras.Model:
        """Create and compile gender classification model."""
        model = tf.keras.Sequential(name="gender_classifier")

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1", input_shape=self.config.input_shape))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool"))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1"))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool"))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1"))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2"))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool"))

        model.add(tf.keras.layers.Flatten(name="flatten"))
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense_1"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="dense_output"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess all images in dataset for model input."""
        processed_df = df.copy()
        processed_df["preprocessed"] = processed_df["image"].apply(lambda img: self._preprocess_image(img))
        return processed_df

    def _preprocess_image(self, image) -> np.ndarray:
        """Preprocess a single image."""
        img_array = np.array(image.convert("L").resize(self.config.input_shape[:2]))
        return (img_array / 255.0).reshape(self.config.input_shape)

    def _split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets."""
        train_val, test = train_test_split(
            df,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
            stratify=df["gender"],
        )
        train, val = train_test_split(
            train_val,
            test_size=self.config.val_split / (1 - self.config.test_split),
            random_state=self.config.random_seed,
            stratify=train_val["gender"],
        )
        return train, val, test

    def _extract_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe."""
        features = np.stack(df["preprocessed"].values)
        labels = df["gender"].values
        return features, labels

    def _train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, List[float]]:
        """Train model and return history."""
        train_data, train_labels = self._extract_data(train_df)
        val_data, val_labels = self._extract_data(val_df)
        history = self.model.fit(
            train_data,
            train_labels,
            validation_data=(val_data, val_labels),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            verbose=1,
            shuffle=True,
        )
        return history.history

    def _evaluate_model(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test data."""
        test_data, test_labels = self._extract_data(test_df)
        loss, accuracy = self.model.evaluate(test_data, test_labels, batch_size=self.config.batch_size, verbose=0)
        return {"loss": loss, "accuracy": accuracy}

    def run_training(self, df: pd.DataFrame) -> Tuple[tf.keras.Model, Dict[str, List[float]], Dict[str, float]]:
        """Run full training pipeline."""
        processed_df = self._prepare_dataset(df)
        train_df, val_df, test_df = self._split_dataset(processed_df)
        self.history = self._train_model(train_df, val_df)
        self.metrics = self._evaluate_model(test_df)
        return self.model
