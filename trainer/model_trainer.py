"""Module for training face classification models using controlled datasets."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DemographicAttribute = Literal["gender", "race", "age"]


@dataclass
class TrainerConfig:
    """Trainer configuration settings."""

    random_seed: Optional[int] = None
    target_col: Optional[DemographicAttribute] = "gender"
    epochs: Optional[int] = 8
    batch_size: Optional[int] = 32
    val_split: Optional[float] = 0.1
    test_split: Optional[float] = 0.2
    input_shape: Optional[tuple[int, int, int]] = (48, 48, 1)
    verbose: Optional[int] = 1


class ModelTrainer:
    """Model trainer for face classification using controlled datasets."""

    def __init__(self, config: TrainerConfig):
        """Initialize the model trainer with configuration."""
        self.config = config
        self.model = None

    def _build_conv_layers(self, model: tf.keras.Sequential) -> tf.keras.Sequential:
        """Build convolutional layers."""
        blocks = [(2, 64), (2, 128), (3, 256)]
        model.add(tf.keras.layers.InputLayer(input_shape=self.config.input_shape, name="input_layer"))
        for idx, (n_convs, filters) in enumerate(blocks):
            for i in range(n_convs):
                model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation="relu", padding="same", name=f"block{idx+1}_conv{i+1}"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{idx+1}_pool"))
        return model

    def _build_classification_head(self, model: tf.keras.Sequential) -> tf.keras.Sequential:
        """Add classification head to model."""
        model.add(tf.keras.layers.Flatten(name="flatten"))
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense_relu"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="output"))
        return model

    def _build_model(self) -> tf.keras.Sequential:
        """Build and compile the Keras model."""
        model = tf.keras.Sequential(name=f"{self.config.target_col}_classifier")
        model = self._build_conv_layers(model)
        model = self._build_classification_head(model)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess images and target column."""
        modified = df.copy()
        target_col = self.config.target_col
        modified["image"] = modified["image"].apply(lambda img: (np.array(img.convert("L").resize(self.config.input_shape[:2])) / 255.0).reshape(self.config.input_shape))
        modified[target_col] = modified[target_col].astype("int32")
        return modified

    def _split_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets using stratified sampling."""
        train_val_df, test_df = train_test_split(df, test_size=self.config.test_split, random_state=self.config.random_seed, stratify=df[self.config.target_col])
        train_df, val_df = train_test_split(train_val_df, test_size=self.config.val_split / (1 - self.config.test_split), random_state=self.config.random_seed, stratify=train_val_df[self.config.target_col])
        return train_df, val_df, test_df

    def _extract_features_and_labels(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract image features and target labels from dataframe."""
        features = np.stack(df["image"].values)
        labels = df[self.config.target_col].values
        return features, labels

    def _train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, list[float]]:
        """Train model and return training history."""
        train_images, train_labels = self._extract_features_and_labels(train_df)
        val_images, val_labels = self._extract_features_and_labels(val_df)
        history = self.model.fit(train_images, train_labels, validation_data=(val_images, val_labels), batch_size=self.config.batch_size, epochs=self.config.epochs, verbose=self.config.verbose, shuffle=True)
        return history.history

    def _evaluate_model(self, test_df: pd.DataFrame) -> dict[str, float]:
        """Evaluate model on test data and return metrics."""
        test_images, test_labels = self._extract_features_and_labels(test_df)
        results = self.model.evaluate(test_images, test_labels, batch_size=self.config.batch_size, verbose=self.config.verbose)
        return {"loss": results[0], "accuracy": results[1]}

    def run_training(self, df: pd.DataFrame) -> tuple[tf.keras.Model, dict[str, list[float]], dict[str, float]]:
        """Run full training pipeline and return model, history, and evaluation metrics."""
        self.model = self._build_model()
        processed_df = self._preprocess_dataset(df)
        train_df, val_df, test_df = self._split_dataset(processed_df)
        history = self._train_model(train_df, val_df)
        metrics = self._evaluate_model(test_df)
        return self.model, history, metrics
