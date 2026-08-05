import unittest

import numpy as np
import tensorflow as tf

from phase7_model_design.design import ModelConfig, build_classifier, compile_model


class ModelDesignTest(unittest.TestCase):
    def test_dense_design_and_compile(self):
        config = ModelConfig(input_shape=(4,), num_classes=3, learning_rate=0.01)
        model = compile_model(build_classifier(config), config)
        self.assertEqual(model.input_shape, (None, 4))
        self.assertEqual(model.output_shape, (None, 3))
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.01, places=6)
        history = model.fit(np.ones((4, 4)), np.array([0, 1, 2, 1]), epochs=1, verbose=0)
        self.assertIn("loss", history.history)

    def test_cnn_design(self):
        config = ModelConfig(input_shape=(16, 16, 3), num_classes=2, backbone="cnn")
        model = build_classifier(config)
        self.assertEqual(model(tf.ones((2, 16, 16, 3))).shape, (2, 2))

    def test_invalid_design(self):
        with self.assertRaises(ValueError):
            ModelConfig(input_shape=(0,), num_classes=2)
        with self.assertRaises(ValueError):
            build_classifier(ModelConfig(input_shape=(4,), num_classes=2, backbone="unknown"))


if __name__ == "__main__":
    unittest.main()
