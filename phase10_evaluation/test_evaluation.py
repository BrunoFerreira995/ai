import unittest

import numpy as np

from phase10_evaluation.evaluation import (
    class_distribution,
    error_analysis,
    evaluate_classifier,
    group_metrics,
    has_class_imbalance,
)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.true = np.array([0, 0, 1, 1, 1, 0])
        self.predicted = np.array([0, 1, 1, 0, 1, 0])

    def test_metrics_and_confusion_matrix(self):
        probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.9, 0.1]])
        results = evaluate_classifier(self.true, self.predicted, probabilities)
        self.assertAlmostEqual(results["accuracy"], 4 / 6)
        self.assertEqual(results["confusion_matrix"].shape, (2, 2))
        self.assertIsNotNone(results["roc_auc"])

    def test_error_and_imbalance_analysis(self):
        errors = error_analysis(self.true, self.predicted)
        self.assertEqual(int(errors["false_positive"].sum()), 1)
        self.assertEqual(int(errors["false_negative"].sum()), 1)
        self.assertEqual(class_distribution([0, 0, 0, 1]).to_dict(), {0: 3, 1: 1})
        self.assertTrue(has_class_imbalance([0, 0, 0, 1], threshold=2))

    def test_group_metrics(self):
        result = group_metrics(self.true, self.predicted, ["a", "a", "a", "b", "b", "b"])
        self.assertEqual(set(result["group"]), {"a", "b"})
        self.assertIn("recall", result.columns)


if __name__ == "__main__":
    unittest.main()
