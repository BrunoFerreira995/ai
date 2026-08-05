import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from phase8_training.training import (
    TrainingConfig,
    build_callbacks,
    build_optimizer,
    configure_mixed_precision,
    create_strategy,
    train_model,
)


class TrainingTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        configure_mixed_precision(False)

    def test_config_optimizer_and_strategy(self):
        config = TrainingConfig(clipnorm=0.5)
        optimizer = build_optimizer(config)
        self.assertEqual(float(optimizer.clipnorm), 0.5)
        self.assertIsInstance(create_strategy(), tf.distribute.Strategy)

    def test_callbacks_and_training_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = TrainingConfig(
                batch_size=4,
                epochs=2,
                patience=1,
                log_dir=f"{temporary}/logs",
                checkpoint_path=f"{temporary}/checkpoints/model.keras",
            )
            callbacks = build_callbacks(config)
            self.assertEqual(len(callbacks), 4)
            model = keras.Sequential(
                [keras.Input(shape=(2,)), keras.layers.Dense(2, activation="softmax")]
            )
            history = train_model(
                model,
                np.ones((8, 2), dtype=np.float32),
                np.array([0, 1] * 4),
                np.ones((2, 2), dtype=np.float32),
                np.array([0, 1]),
                config,
            )
            self.assertIn("loss", history.history)
            self.assertTrue(tf.io.gfile.exists(config.checkpoint_path))


if __name__ == "__main__":
    unittest.main()
