"""Practical Keras training utilities for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible settings for a model training run."""

    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    decay_steps: int = 1000
    decay_rate: float = 0.96
    validation_freq: int = 1
    patience: int = 3
    clipnorm: float | None = 1.0
    mixed_precision: bool = False
    log_dir: str = "logs/training"
    checkpoint_path: str = "checkpoints/best.keras"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0 or self.decay_steps <= 0:
            raise ValueError("batch_size, epochs, and decay_steps must be positive")
        if self.learning_rate <= 0 or not 0 < self.decay_rate <= 1:
            raise ValueError("learning_rate must be positive and decay_rate must be in (0, 1]")
        if self.patience < 0 or self.validation_freq <= 0:
            raise ValueError("patience must be non-negative and validation_freq must be positive")
        if self.clipnorm is not None and self.clipnorm <= 0:
            raise ValueError("clipnorm must be positive when supplied")


def configure_mixed_precision(enabled: bool = True) -> str:
    """Set and return TensorFlow's global mixed-precision policy."""
    policy_name = "mixed_float16" if enabled else "float32"
    tf.keras.mixed_precision.set_global_policy(policy_name)
    return policy_name


def build_optimizer(config: TrainingConfig) -> keras.optimizers.Optimizer:
    """Build Adam with an exponential learning-rate schedule and clipping."""
    schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=False,
    )
    return keras.optimizers.Adam(learning_rate=schedule, clipnorm=config.clipnorm)


def build_callbacks(config: TrainingConfig) -> list[keras.callbacks.Callback]:
    """Create early stopping, LR reduction, checkpoint, and TensorBoard callbacks."""
    checkpoint = Path(config.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config.patience, verbose=0),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint), monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=0
        ),
        keras.callbacks.TensorBoard(log_dir=config.log_dir),
    ]


def create_strategy() -> tf.distribute.Strategy:
    """Create a strategy that uses all visible GPUs, or the local CPU otherwise."""
    if tf.config.list_logical_devices("GPU"):
        return tf.distribute.MirroredStrategy()
    return tf.distribute.OneDeviceStrategy("/cpu:0")


def train_model(
    model: keras.Model,
    train_data: Any,
    train_labels: Any | None,
    validation_data: Any,
    validation_labels: Any | None,
    config: TrainingConfig,
) -> keras.callbacks.History:
    """Compile when needed and train with batching, validation, and callbacks."""
    if config.mixed_precision:
        configure_mixed_precision(True)
    if getattr(model, "optimizer", None) is None:
        model.compile(
            optimizer=build_optimizer(config),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    validation_input = (validation_data, validation_labels) if validation_labels is not None else validation_data
    fit_kwargs = {
        "validation_data": validation_input,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "validation_freq": config.validation_freq,
        "callbacks": build_callbacks(config),
        "verbose": 0,
    }
    if train_labels is None:
        return model.fit(train_data, **fit_kwargs)
    return model.fit(train_data, train_labels, **fit_kwargs)
