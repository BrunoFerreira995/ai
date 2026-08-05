import tempfile
import unittest
from pathlib import Path

import numpy as np
from tensorflow import keras

from phase13_export.exporters import (
    export_onnx,
    export_saved_model,
    export_tensorrt,
    export_tflite,
)


def tiny_model():
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(2, activation="softmax")])
    model(np.zeros((1, 4), dtype=np.float32))
    return model


class ExportTest(unittest.TestCase):
    def test_saved_model_tflite_and_onnx(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = tiny_model()
            saved = export_saved_model(model, root / "saved_model")
            tflite = export_tflite(model, root / "model.tflite")
            onnx = export_onnx(model, root / "model.onnx")
            self.assertTrue((saved / "saved_model.pb").exists())
            self.assertGreater(tflite.stat().st_size, 0)
            self.assertGreater(onnx.stat().st_size, 0)

    def test_tensorrt_requires_supported_runtime(self):
        with self.assertRaises(RuntimeError):
            export_tensorrt("missing_saved_model", "output")


if __name__ == "__main__":
    unittest.main()
