import unittest

import numpy as np
import pandas as pd

from phase4_data_engineering.pipeline import (
    clean_dataframe,
    detect_outliers_iqr,
    encode_categorical,
    normalize,
    remove_outliers_iqr,
    select_features_by_variance,
    split_dataset,
    standardize,
    validate_labels,
)


class DataEngineeringTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"age": [10, 20, 30, 100], "score": [1.0, 2.0, 3.0, 4.0], "kind": ["a", "b", "a", "b"]})

    def test_cleaning_and_outliers(self):
        dirty = pd.concat([self.frame.iloc[:3], self.frame.iloc[[0]]], ignore_index=True)
        dirty.loc[1, "kind"] = None
        cleaned = clean_dataframe(dirty, missing="fill")
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned["kind"].isna().sum(), 0)
        self.assertTrue(detect_outliers_iqr(self.frame, ["age"]).iloc[3])
        self.assertEqual(len(remove_outliers_iqr(self.frame, ["age"])), 3)

    def test_features_and_labels(self):
        validate_labels(["cat", "dog"], ["cat", "dog"])
        with self.assertRaises(ValueError):
            validate_labels(["cat", "bird"], ["cat", "dog"])
        normalized = normalize(self.frame, ["age"])
        self.assertTrue(np.allclose(normalized["age"], [0, 1 / 9, 2 / 9, 1]))
        standardized = standardize(self.frame, ["score"])
        self.assertAlmostEqual(float(standardized["score"].mean()), 0.0)
        encoded = encode_categorical(self.frame, ["kind"])
        self.assertIn("kind_a", encoded.columns)
        selected = select_features_by_variance(self.frame, threshold=1.5)
        self.assertIn("age", selected.columns)
        self.assertNotIn("score", selected.columns)

    def test_split(self):
        features = pd.DataFrame({"x": range(20)})
        labels = [index % 2 for index in range(20)]
        result = split_dataset(features, labels, validation_size=0.2, test_size=0.2)
        self.assertEqual(sum(len(part[0]) for part in result.values()), 20)
        self.assertEqual(set(result), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
