import unittest
from pathlib import Path

import numpy as np

from edge_ai.inference import EdgeInterpreter, detect_platform


class EdgeInferenceTest(unittest.TestCase):
    def test_tflite_inference(self):
        model_path = Path("artifacts/model.tflite")
        if not model_path.exists():
            self.skipTest("artifacts/model.tflite is not available")
        interpreter = EdgeInterpreter(model_path)
        output = interpreter.predict(np.zeros(interpreter.input_shape, dtype=np.float32))
        self.assertEqual(output.shape[0], 1)
        self.assertGreaterEqual(output.shape[-1], 2)

    def test_platform_detection(self):
        self.assertTrue(detect_platform())


if __name__ == "__main__":
    unittest.main()
