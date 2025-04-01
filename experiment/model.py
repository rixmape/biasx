from typing import Tuple

import tensorflow as tf

# isort: off
from datatypes import ModelTrainingHistory
from config import ExperimentsConfig
from utils import setup_logger


class ModelTrainer:

    def __init__(self, config: ExperimentsConfig, log_path: str):
        self.config = config
        self.logger = setup_logger(name="model_trainer", log_path=log_path)

    def _build_model(self) -> tf.keras.Model:
        self.logger.info("Building CNN model architecture.")

        model = tf.keras.Sequential()
        input_channels = 1 if self.config.dataset.use_grayscale else 3
        input_shape = (self.config.dataset.image_size, self.config.dataset.image_size, input_channels)
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
        self.logger.info(f"Model built successfully: {total_params:,} total parameters.")
        return model

    def _remove_image_id_from_dataset(
        self,
        dataset: tf.data.Dataset,
    ) -> tf.data.Dataset:
        return dataset.map(lambda image, label, image_id: (image, label))

    def train_model(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
    ) -> Tuple[tf.keras.Model, ModelTrainingHistory]:

        train_data_fit = self._remove_image_id_from_dataset(train_data)
        val_data_fit = self._remove_image_id_from_dataset(val_data)

        model = self._build_model()
        self.logger.info(f"Training model for {self.config.model.epochs} epochs with batch size {self.config.model.batch_size}.")

        history = model.fit(train_data_fit, validation_data=val_data_fit, epochs=self.config.model.epochs, verbose=0)
        history = ModelTrainingHistory(
            train_loss=history.history["loss"],
            train_accuracy=history.history["accuracy"],
            val_loss=history.history["val_loss"],
            val_accuracy=history.history["val_accuracy"],
        )

        self.logger.info(f"Training completed: final train_acc={history.train_accuracy[-1]:.4f}, final val_acc={history.val_accuracy[-1]:.4f}")
        return model, history
