import unittest

import numpy as np
import tensorflow as tf

from phase21_classifier_evaluation.metrics import classification_metrics, expected_calibration_error
from phase21_classifier_evaluation.performance import benchmark_inference
from phase21_classifier_evaluation.robustness import evaluate_noise


class Phase21Tests(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.Sequential([tf.keras.layers.Input((2,)), tf.keras.layers.Dense(2, activation="softmax")])
        self.model(np.zeros((1, 2), dtype="float32"))
        self.x = np.ones((8, 2), dtype="float32")
        self.probabilities = np.tile([[0.9, 0.1]], (8, 1))

    def test_metrics_include_classifier_scores(self):
        result = classification_metrics(np.zeros(8, dtype=int), self.probabilities, labels=[0, 1])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertIn("balanced_accuracy", result)
        self.assertIn("ece", result)
        self.assertEqual(len(result["confusion_matrix"]), 2)

    def test_noise_and_performance_reports(self):
        labels = np.argmax(self.model.predict(self.x, verbose=0), axis=1)
        self.assertIn("noisy_accuracy", evaluate_noise(self.model, self.x, labels))
        self.assertIn("latency_ms", benchmark_inference(self.model, self.x, repeats=1, warmup=0))

    def test_ece_range(self):
        self.assertGreaterEqual(expected_calibration_error(np.zeros(8, dtype=int), self.probabilities), 0.0)
        self.assertLessEqual(expected_calibration_error(np.zeros(8, dtype=int), self.probabilities), 1.0)


if __name__ == "__main__":
    unittest.main()
