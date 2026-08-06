import tempfile
import unittest
from pathlib import Path

import numpy as np

from retraining.pipeline import ABTestRouter, ModelRegistry, RetrainingPipeline, should_retrain, validate_dataset


class RetrainingTest(unittest.TestCase):
    def test_validation_and_trigger(self):
        report = validate_dataset(np.ones((4, 2)), [0, 1, 0, 1])
        self.assertEqual(report["samples"], 4)
        self.assertTrue(should_retrain(1000, sample_threshold=1000))
        self.assertTrue(should_retrain(2, sample_threshold=1000, drift_score=0.4))

    def test_registry_and_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "candidate"
            source.mkdir()
            (source / "saved_model.pb").write_bytes(b"model")
            registry = ModelRegistry(Path(temporary) / "registry")
            pipeline = RetrainingPipeline(registry, minimum_metric=0.8)
            result = pipeline.run(
                np.ones((4, 2)),
                [0, 1, 0, 1],
                lambda features, labels: source,
                lambda path: {"accuracy": 0.9},
            )
            self.assertEqual(result["status"], "registered")
            registry.promote(result["version"])
            self.assertTrue((registry.root / "production" / "saved_model.pb").exists())

    def test_ab_assignment_is_stable(self):
        router = ABTestRouter({"champion": 90, "candidate": 10})
        self.assertEqual(router.assign("user-1"), router.assign("user-1"))
        self.assertIn(router.assign("user-2"), {"champion", "candidate"})


if __name__ == "__main__":
    unittest.main()
