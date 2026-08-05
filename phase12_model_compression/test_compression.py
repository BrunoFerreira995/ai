import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from phase12_model_compression.compression import (
    Distiller,
    cluster_model,
    convert_to_int8_tflite,
    prune_model,
    quantize_model,
    strip_pruning,
)


def tiny_model(classes=2):
    return keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(8, activation="relu"), keras.layers.Dense(classes)])


class CompressionTest(unittest.TestCase):
    def test_quantization_and_pruning(self):
        quantized = quantize_model(tiny_model())
        self.assertTrue(any("quantize" in layer.name for layer in quantized.layers))
        pruned = prune_model(tiny_model(), end_step=10)
        self.assertTrue(any("prune" in layer.name for layer in pruned.layers))
        self.assertIsInstance(strip_pruning(pruned), keras.Model)

    def test_clustering_and_distillation(self):
        clustered = cluster_model(tiny_model(), clusters=4)
        self.assertTrue(any("cluster" in layer.name for layer in clustered.layers))
        teacher, student = tiny_model(), tiny_model()
        distiller = Distiller(student, teacher)
        distiller.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        history = distiller.fit(np.ones((4, 4), dtype=np.float32), np.array([0, 1, 0, 1]), epochs=1, verbose=0)
        self.assertIn("loss", history.history)

    def test_int8_export(self):
        model = tiny_model()
        model(np.zeros((1, 4), dtype=np.float32))

        def representative_dataset():
            for _ in range(2):
                yield [np.ones((1, 4), dtype=np.float32)]

        tflite_model = convert_to_int8_tflite(model, representative_dataset)
        self.assertGreater(len(tflite_model), 0)


if __name__ == "__main__":
    unittest.main()
