"""Define model inputs, outputs, backbone, loss, optimizer, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from phase6_neural_networks.models import build_cnn, build_dense_network, build_pretrained_backbone


@dataclass(frozen=True)
class ModelConfig:
    """Complete configuration for a supervised classifier."""

    input_shape: tuple[int, ...]
    num_classes: int
    backbone: str = "dense"
    loss: str = "sparse_categorical_crossentropy"
    optimizer: str | keras.optimizers.Optimizer = "adam"
    metrics: tuple[str | keras.metrics.Metric, ...] = field(default_factory=lambda: ("accuracy",))
    learning_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.input_shape or any(d <= 0 for d in self.input_shape):
            raise ValueError("input_shape must contain positive dimensions")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def _build_backbone(config: ModelConfig) -> keras.Model:
    key = config.backbone.lower()
    if key == "dense":
        return build_dense_network(config.input_shape, config.num_classes)
    if key in {"cnn", "residual_cnn"}:
        if len(config.input_shape) != 3:
            raise ValueError("CNN backbones require (height, width, channels)")
        architecture = "residual" if key == "residual_cnn" else "standard"
        return build_cnn(config.input_shape, config.num_classes, architecture=architecture)
    if key in {"efficientnet", "mobilenet", "resnet"}:
        if len(config.input_shape) != 3:
            raise ValueError("image backbones require (height, width, channels)")
        return build_pretrained_backbone(key, config.input_shape, config.num_classes)
    raise ValueError(f"unsupported backbone: {config.backbone}")


def build_classifier(config: ModelConfig) -> keras.Model:
    """Build a classifier with explicit input and output tensors."""
    return _build_backbone(config)


def compile_model(model: keras.Model, config: ModelConfig) -> keras.Model:
    """Compile a model from the configured loss, optimizer, and metrics."""
    optimizer = config.optimizer
    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)
    if config.learning_rate is not None:
        optimizer.learning_rate.assign(config.learning_rate)
    model.compile(loss=config.loss, optimizer=optimizer, metrics=list(config.metrics))
    return model
