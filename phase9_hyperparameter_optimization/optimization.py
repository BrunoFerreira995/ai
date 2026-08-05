"""Random and Bayesian hyperparameter optimization for Keras models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras


@dataclass(frozen=True)
class HyperparameterSpace:
    """Discrete search space used by the lightweight random search."""

    learning_rates: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
    batch_sizes: tuple[int, ...] = (16, 32, 64)
    optimizers: tuple[str, ...] = ("adam", "sgd", "rmsprop")
    dropout_rates: tuple[float, ...] = (0.0, 0.2, 0.5)
    regularization_strengths: tuple[float, ...] = (0.0, 1e-4, 1e-3)

    def __post_init__(self) -> None:
        if not all(self.learning_rates) or not all(self.batch_sizes):
            raise ValueError("learning rates and batch sizes must be positive")
        if any(rate < 0 or rate >= 1 for rate in self.dropout_rates):
            raise ValueError("dropout rates must be in [0, 1)")
        if any(value < 0 for value in self.regularization_strengths):
            raise ValueError("regularization strengths cannot be negative")


def sample_hyperparameters(
    space: HyperparameterSpace = HyperparameterSpace(), trials: int = 10, seed: int = 42
) -> list[dict[str, Any]]:
    """Sample reproducible hyperparameter configurations without replacement when possible."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    configurations = []
    seen = set()
    max_unique = (
        len(space.learning_rates)
        * len(space.batch_sizes)
        * len(space.optimizers)
        * len(space.dropout_rates)
        * len(space.regularization_strengths)
    )
    target = min(trials, max_unique)
    while len(configurations) < target:
        configuration = {
            "learning_rate": float(rng.choice(space.learning_rates)),
            "batch_size": int(rng.choice(space.batch_sizes)),
            "optimizer": str(rng.choice(space.optimizers)),
            "dropout_rate": float(rng.choice(space.dropout_rates)),
            "regularization_strength": float(rng.choice(space.regularization_strengths)),
        }
        key = tuple(configuration.values())
        if key not in seen:
            seen.add(key)
            configurations.append(configuration)
    return configurations


def _compile_from_config(model: keras.Model, configuration: dict[str, Any]) -> keras.Model:
    optimizer = keras.optimizers.get(configuration["optimizer"])
    optimizer.learning_rate.assign(configuration["learning_rate"])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run_random_search(
    model_builder: Callable[[dict[str, Any]], keras.Model],
    train_data: Any,
    train_labels: Any,
    validation_data: Any,
    validation_labels: Any,
    *,
    space: HyperparameterSpace = HyperparameterSpace(),
    trials: int = 5,
    epochs: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train sampled configurations and return the best result by val accuracy."""
    best: dict[str, Any] | None = None
    for configuration in sample_hyperparameters(space, trials, seed):
        model = _compile_from_config(model_builder(configuration), configuration)
        history = model.fit(
            train_data,
            train_labels,
            validation_data=(validation_data, validation_labels),
            batch_size=configuration["batch_size"],
            epochs=epochs,
            verbose=0,
        )
        score = float(max(history.history.get("val_accuracy", [0.0])))
        result = {"configuration": configuration, "score": score, "model": model, "history": history}
        if best is None or score > best["score"]:
            best = result
    assert best is not None
    return best


def build_bayesian_tuner(
    hypermodel: Callable[[kt.HyperParameters], keras.Model],
    *,
    max_trials: int = 10,
    objective: str = "val_accuracy",
    directory: str = "tuner_results",
    project_name: str = "model_search",
    seed: int = 42,
) -> kt.BayesianOptimization:
    """Create a Keras Tuner Bayesian optimizer.

    The supplied ``hypermodel`` should define choices with ``hp.Float``,
    ``hp.Int``, and ``hp.Choice``, compile the model, and return it.
    """
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    return kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective=kt.Objective(objective, direction="max" if "accuracy" in objective else "min"),
        max_trials=max_trials,
        directory=directory,
        project_name=project_name,
        overwrite=True,
        seed=seed,
    )
