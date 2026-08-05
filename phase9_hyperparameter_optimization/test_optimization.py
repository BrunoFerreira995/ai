import tempfile
import unittest

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras

from phase9_hyperparameter_optimization.optimization import (
    HyperparameterSpace,
    build_bayesian_tuner,
    sample_hyperparameters,
)


class HyperparameterOptimizationTest(unittest.TestCase):
    def test_space_sampling(self):
        space = HyperparameterSpace(learning_rates=(1e-3, 1e-4), batch_sizes=(16, 32))
        samples = sample_hyperparameters(space, trials=3, seed=7)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples, sample_hyperparameters(space, trials=3, seed=7))
        self.assertTrue(all("dropout_rate" in sample for sample in samples))

    def test_bayesian_tuner(self):
        def hypermodel(hp):
            units = hp.Int("units", min_value=4, max_value=8, step=4)
            learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
            model = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(units, activation="relu"), keras.layers.Dense(2, activation="softmax")])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            return model

        with tempfile.TemporaryDirectory() as temporary:
            tuner = build_bayesian_tuner(hypermodel, max_trials=2, directory=temporary)
            self.assertIsInstance(tuner, kt.BayesianOptimization)
            self.assertEqual(tuner.oracle.max_trials, 2)


if __name__ == "__main__":
    unittest.main()
