import os
import unittest

import numpy as np
from fastapi.testclient import TestClient

os.environ.setdefault("MODEL_PATH", "artifacts/saved_model")

from backend.app import app  # noqa: E402


class BackendTest(unittest.TestCase):
    def test_health_and_prediction(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            response = client.post("/predict", json={"instances": np.ones((1, 8)).tolist()})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(len(body["predictions"]), 1)
            self.assertGreaterEqual(len(body["probabilities"][0]), 2)
            self.assertEqual(len(body["predictions"]), len(body["class_names"]))


if __name__ == "__main__":
    unittest.main()
