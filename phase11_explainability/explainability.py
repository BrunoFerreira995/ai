"""Grad-CAM, SHAP, LIME, and attention-visualization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _find_last_conv_layer(model: keras.Model) -> layers.Layer:
    convolutional_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    if not convolutional_layers:
        raise ValueError("model does not contain a Conv2D layer")
    return convolutional_layers[-1]


def grad_cam(
    model: keras.Model,
    image: np.ndarray | tf.Tensor,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> tuple[np.ndarray, int]:
    """Compute a normalized Grad-CAM heatmap for one image.

    Returns ``(heatmap, selected_class)``. The heatmap has the spatial size of
    the selected convolutional layer and values in ``[0, 1]``.
    """
    convolutional_layer = model.get_layer(layer_name) if layer_name else _find_last_conv_layer(model)
    grad_model = keras.Model(model.inputs, [convolutional_layer.output, model.output])
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    if tensor.shape.rank == len(model.input_shape) - 1:
        tensor = tensor[tf.newaxis, ...]
    with tf.GradientTape() as tape:
        activations, predictions = grad_model(tensor, training=False)
        selected_class = int(tf.argmax(predictions[0]).numpy()) if class_index is None else int(class_index)
        score = predictions[:, selected_class]
    gradients = tape.gradient(score, activations)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * activations, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
    return heatmap.numpy(), selected_class


def explain_with_shap(model: keras.Model, background: np.ndarray, samples: np.ndarray) -> Any:
    """Return SHAP values using TensorFlow's gradient-based explainer."""
    try:
        import shap
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("SHAP is required for this explanation") from error
    explainer = shap.GradientExplainer(model, background)
    return explainer.shap_values(samples)


def explain_with_lime(
    model: keras.Model,
    image: np.ndarray,
    *,
    class_names: list[str] | None = None,
    num_samples: int = 1000,
) -> Any:
    """Create a LIME image explanation for a Keras classifier."""
    try:
        from lime import lime_image
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("LIME is required for this explanation") from error
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    explainer = lime_image.LimeImageExplainer()
    return explainer.explain_instance(
        np.asarray(image),
        lambda batch: model.predict(batch, verbose=0),
        top_labels=len(class_names) if class_names else 5,
        hide_color=0,
        num_samples=num_samples,
    )


def attention_visualization(
    query: tf.Tensor,
    key: tf.Tensor | None = None,
    value: tf.Tensor | None = None,
    *,
    num_heads: int = 4,
    key_dim: int = 8,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Run multi-head attention and return both output and attention weights."""
    if key is None:
        key = query
    if value is None:
        value = key
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    output, weights = attention(query=query, key=key, value=value, return_attention_scores=True)
    return output, weights
