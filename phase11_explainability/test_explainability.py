import unittest

import numpy as np
import tensorflow as tf

from phase6_neural_networks.models import build_cnn
from phase11_explainability.explainability import attention_visualization, grad_cam


class ExplainabilityTest(unittest.TestCase):
    def test_grad_cam(self):
        model = build_cnn((32, 32, 3), 3)
        heatmap, class_index = grad_cam(model, np.ones((32, 32, 3), dtype=np.float32))
        self.assertEqual(heatmap.ndim, 2)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertIn(class_index, range(3))

    def test_attention_weights(self):
        output, weights = attention_visualization(tf.ones((2, 5, 16)), num_heads=4, key_dim=4)
        self.assertEqual(output.shape, (2, 5, 16))
        self.assertEqual(weights.shape, (2, 4, 5, 5))
        self.assertTrue(np.allclose(tf.reduce_sum(weights, axis=-1), 1.0))


if __name__ == "__main__":
    unittest.main()
