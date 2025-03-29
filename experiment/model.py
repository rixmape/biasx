import numpy as np
import tensorflow as tf

# isort: off
from config import Config
from datatypes import DatasetSplits
from utils import setup_logger

logger = setup_logger(name="experiment.model")


class ModelTrainer:
    """Class that constructs a convolutional neural network model, trains it on the dataset, and produces predictions on test data."""

    def __init__(self, config: Config):
        self.config = config
        logger.info(f"Initializing ModelTrainer with image size: {config.image_size}x{config.image_size}")
        logger.debug(f"Training parameters: batch_size={config.batch_size}, epochs={config.epochs}, grayscale={config.grayscale}")

    def _build_model(self) -> tf.keras.Model:
        """Constructs and compiles a sequential convolutional neural network model based on the configuration settings."""
        logger.info("Building CNN model architecture")
        model = tf.keras.Sequential()

        input_shape = (self.config.image_size, self.config.image_size, 1 if self.config.grayscale else 3)
        model.add(tf.keras.layers.Input(shape=input_shape, name="input"))
        logger.debug(f"Model input shape: {input_shape}")

        for block, (filters, layers) in enumerate([(64, 2), (128, 2), (256, 3)], start=1):
            logger.debug(f"Adding convolutional block {block}: {layers} layers with {filters} filters each")
            for i in range(layers):
                model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{block}_conv{i+1}"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{block}_pool"))

        model.add(tf.keras.layers.Flatten(name="flatten"))
        logger.debug("Adding classification head layers")
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense_1"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="dense_output"))

        logger.debug("Compiling model with Adam optimizer (lr=0.0001) and sparse categorical crossentropy loss")
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        total_params = model.count_params()
        logger.info(f"Model built successfully: {total_params:,} total parameters")

        return model

    def train_and_predict(self, splits: DatasetSplits) -> tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """Trains the model on the training dataset, evaluates it on the validation set, and generates predictions on the test data."""
        logger.info("Starting model training process")
        model = self._build_model()

        logger.info(f"Training model for {self.config.epochs} epochs")
        history = model.fit(splits.train_dataset, validation_data=splits.val_dataset, epochs=self.config.epochs, verbose=0)

        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        logger.info(f"Training completed - Final metrics: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        logger.debug(f"Final loss values: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_acc < 0.6:
            logger.warning(f"Low validation accuracy ({val_acc:.4f}). Model may be underfitting or the task might be challenging.")

        if val_acc < train_acc - 0.1:
            logger.warning(f"Large gap between training and validation accuracy ({train_acc:.4f} vs {val_acc:.4f}). Possible overfitting.")

        logger.info("Generating predictions on test dataset")
        all_predictions = []
        all_labels = []

        batch_count = 0
        total_samples = 0

        for batch in splits.test_dataset:
            batch_count += 1
            images, labels = batch
            batch_size = len(labels)
            total_samples += batch_size

            logger.debug(f"Processing test batch {batch_count}: {batch_size} samples")
            batch_predictions = model.predict(images, verbose=0)
            all_predictions.append(batch_predictions)
            all_labels.append(labels)

        logger.debug(f"Processed {batch_count} test batches with {total_samples} total samples")

        predictions = np.vstack(all_predictions).argmax(axis=1)
        test_labels = np.concatenate(all_labels)

        return model, predictions, test_labels
