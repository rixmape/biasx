import logging
from typing import Tuple

import tensorflow as tf

# isort: off
from config import Config
from datatypes import ModelHistory


class ModelTrainer:
    """Builds and trains the CNN model for the experiment."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initializes the model trainer with configuration and logger."""
        self.config = config
        self.logger = logger

    def _build_model(self) -> tf.keras.Model:
        """Constructs the CNN model architecture using TensorFlow/Keras."""
        input_channels = 1 if self.config.dataset.use_grayscale else 3
        input_shape = (self.config.dataset.image_size, self.config.dataset.image_size, input_channels)

        self.logger.info(f"Starting model building: input_shape={input_shape}")

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape, name="input"))

        conv_blocks = [(64, 2), (128, 2), (256, 3)]
        for idx, (filters, layers_count) in enumerate(conv_blocks, start=1):
            for i in range(1, layers_count + 1):
                conv_layer = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{idx}_conv{i}")
                model.add(conv_layer)
            pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{idx}_pool")
            model.add(pooling_layer)

        model.add(tf.keras.layers.Flatten(name="flatten"))
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="output"))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        total_params = model.count_params()
        self.logger.info(f"Completed model building: parameters={total_params:,}")
        return model

    def get_model_and_history(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
    ) -> Tuple[tf.keras.Model, ModelHistory]:
        """Trains the model on the provided data and returns the model and training history."""

        train_data_fit = train_data.map(lambda image, label, _id, _race, _age: (image, label))
        val_data_fit = val_data.map(lambda image, label, _id, _race, _age: (image, label))

        model = self._build_model()
        self.logger.info(f"Starting model training: epochs={self.config.model.epochs}, batch_size={self.config.model.batch_size}.")

        history = model.fit(train_data_fit, validation_data=val_data_fit, epochs=self.config.model.epochs, verbose=0)
        history = ModelHistory(
            train_loss=history.history["loss"],
            train_accuracy=history.history["accuracy"],
            val_loss=history.history["val_loss"],
            val_accuracy=history.history["val_accuracy"],
        )

        self.logger.info(f"Completed model training: train_acc={history.train_accuracy[-1]:.4f}, val_acc={history.val_accuracy[-1]:.4f}")
        return model, history
