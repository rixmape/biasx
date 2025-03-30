import numpy as np
import tensorflow as tf

# isort: off
from config import Config
from utils import setup_logger


class ModelTrainer:
    """Class that constructs a convolutional neural network model, trains it on the dataset, and produces predictions on test data."""

    def __init__(self, config: Config, log_path: str):
        self.config = config
        self.logger = setup_logger(name="model_trainer", log_path=log_path)

    def _build_model(self) -> tf.keras.Model:
        """Constructs and compiles a sequential convolutional neural network model based on the configuration settings."""
        self.logger.info("Building CNN model architecture for facial gender classification")
        model = tf.keras.Sequential()

        input_shape = (self.config.image_size, self.config.image_size, 1 if self.config.grayscale else 3)
        model.add(tf.keras.layers.Input(shape=input_shape, name="input"))
        self.logger.debug(f"Model input shape: {input_shape}")

        for block, (filters, layers) in enumerate([(64, 2), (128, 2), (256, 3)], start=1):
            self.logger.debug(f"Adding convolutional block {block}: {layers} layers with {filters} filters each")
            for i in range(layers):
                model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{block}_conv{i+1}"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{block}_pool"))

        model.add(tf.keras.layers.Flatten(name="flatten"))
        self.logger.debug("Adding classification head layers")
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense_1"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="dense_output"))

        self.logger.debug("Compiling model with Adam optimizer (lr=0.0001) and sparse categorical crossentropy loss")
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        total_params = model.count_params()
        self.logger.info(f"Model built successfully: {total_params:,} total parameters")

        return model

    def train_model(self, train_data: tf.data.Dataset, val_data: tf.data.Dataset) -> tf.keras.Model:
        """Trains the model on the training dataset and evaluates it on the test set."""
        model = self._build_model()

        self.logger.info(f"Training model for {self.config.epochs} epochs with {self.config.batch_size} batch size")
        history = model.fit(train_data, validation_data=val_data, epochs=self.config.epochs, verbose=0)

        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        self.logger.info(f"Training completed: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        self.logger.debug(f"Final loss values: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_acc < 0.6:
            self.logger.warning(f"Low validation accuracy ({val_acc:.4f}). Model may be underfitting.")

        if val_acc < train_acc - 0.1:
            self.logger.warning(f"Large gap between training and validation accuracy ({train_acc:.4f} vs {val_acc:.4f}). Possible overfitting.")

        return model

    def predict(self, model: tf.keras.Model, test_data: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Generates predictions on the dataset using the trained model."""
        self.logger.info("Generating predictions on the dataset")

        all_predictions = []
        all_labels = []
        batch_count = 0
        total_samples = 0

        self.logger.info(f"Processing batches for prediction with {self.config.batch_size} batch size")

        for batch in test_data:
            batch_count += 1
            images, labels = batch
            batch_size = len(labels)
            total_samples += batch_size

            batch_predictions = model.predict(images, verbose=0)
            all_predictions.append(batch_predictions)
            all_labels.append(labels)

            if batch_count % 5 == 0:
                self.logger.debug(f"Processed {batch_count} batches with {total_samples} total samples")

        self.logger.debug(f"Processed {batch_count} batches with {total_samples} total samples")

        predictions = np.vstack(all_predictions).argmax(axis=1)
        labels = np.concatenate(all_labels)

        return predictions, labels
