import unittest

import numpy as np
import tensorflow as tf

from phase6_neural_networks.models import (
    build_cnn,
    build_dense_network,
    build_pretrained_backbone,
    build_rnn,
    build_seq2seq,
    build_transformer_classifier,
    positional_encoding,
)


class NeuralNetworksTest(unittest.TestCase):
    def test_dense_and_cnn_models(self):
        dense = build_dense_network((8,), 3)
        self.assertEqual(dense(tf.ones((2, 8))).shape, (2, 3))
        cnn = build_cnn((32, 32, 3), 4, architecture="residual")
        self.assertEqual(cnn(tf.ones((2, 32, 32, 3))).shape, (2, 4))

    def test_backbones(self):
        for name in ("efficientnet", "mobilenet", "resnet"):
            model = build_pretrained_backbone(name, input_shape=(32, 32, 3), num_classes=2)
            self.assertEqual(model(tf.ones((1, 32, 32, 3))).shape, (1, 2))

    def test_recurrent_models(self):
        for cell in ("lstm", "gru"):
            model = build_rnn((5, 3), 2, cell=cell)
            self.assertEqual(model(tf.ones((2, 5, 3))).shape, (2, 2))
        seq2seq = build_seq2seq((5, 3), output_length=4, output_features=2)
        self.assertEqual(seq2seq(tf.ones((2, 5, 3))).shape, (2, 4, 2))

    def test_transformer(self):
        encoding = positional_encoding(6, 8)
        self.assertEqual(encoding.shape, (1, 6, 8))
        model = build_transformer_classifier((6, 4), 3)
        output = model(tf.ones((2, 6, 4)))
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(np.allclose(tf.reduce_sum(output, axis=-1), [1, 1]))


if __name__ == "__main__":
    unittest.main()
