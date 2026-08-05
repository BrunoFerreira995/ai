"""Dense, convolutional, recurrent, and Transformer Keras models."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dense_network(
    input_shape: tuple[int, ...],
    num_classes: int,
    hidden_units: tuple[int, ...] = (128, 64),
    dropout_rate: float = 0.2,
) -> keras.Model:
    """Build a fully connected classifier with activations and normalization."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    inputs = keras.Input(shape=input_shape, name="features")
    x = layers.Flatten()(inputs)
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="dense_classifier")


def residual_block(inputs: tf.Tensor, filters: int, stride: int = 1, name: str = "residual_block") -> tf.Tensor:
    """Apply a two-convolution residual block with a projection shortcut when needed."""
    shortcut = inputs
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1")(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    if inputs.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, name=f"{name}_projection")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)
    return layers.Activation("relu", name=f"{name}_output")(layers.Add(name=f"{name}_add")([x, shortcut]))


def build_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    architecture: str = "standard",
) -> keras.Model:
    """Build a CNN using pooling or residual blocks."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    if architecture == "residual":
        x = residual_block(x, 32, name="block1")
        x = residual_block(x, 64, stride=2, name="block2")
    elif architecture == "standard":
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    else:
        raise ValueError("architecture must be 'standard' or 'residual'")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name=f"{architecture}_cnn")


def build_pretrained_backbone(
    name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int | None = None,
    trainable: bool = False,
) -> keras.Model:
    """Build an ImageNet architecture without downloading weights.

    Supported names are ``efficientnet``, ``mobilenet``, and ``resnet``.
    ``num_classes`` adds a classifier head; otherwise the model returns feature maps.
    """
    backbones = {
        "efficientnet": keras.applications.EfficientNetB0,
        "mobilenet": keras.applications.MobileNetV2,
        "resnet": keras.applications.ResNet50,
    }
    key = name.lower()
    if key not in backbones:
        raise ValueError(f"unsupported backbone: {name}")
    backbone = backbones[key](include_top=False, weights=None, input_shape=input_shape)
    backbone.trainable = trainable
    if num_classes is None:
        return backbone
    inputs = keras.Input(shape=input_shape, name="image")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name=f"{key}_classifier")


def build_rnn(
    input_shape: tuple[int, int],
    num_classes: int,
    cell: str = "lstm",
    units: int = 64,
) -> keras.Model:
    """Build an LSTM or GRU sequence classifier."""
    cells = {"lstm": layers.LSTM, "gru": layers.GRU}
    key = cell.lower()
    if key not in cells:
        raise ValueError("cell must be 'lstm' or 'gru'")
    inputs = keras.Input(shape=input_shape, name="sequence")
    x = cells[key](units, return_sequences=False)(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name=f"{key}_classifier")


def build_seq2seq(
    input_shape: tuple[int, int],
    output_length: int,
    output_features: int,
    units: int = 64,
) -> keras.Model:
    """Build a compact encoder-decoder Seq2Seq model."""
    inputs = keras.Input(shape=input_shape, name="encoder_sequence")
    encoded = layers.LSTM(units, return_state=True, name="encoder")
    _, hidden, cell = encoded(inputs)
    repeated = layers.RepeatVector(output_length)(hidden)
    decoded = layers.LSTM(units, return_sequences=True, name="decoder")(repeated, initial_state=[hidden, cell])
    outputs = layers.TimeDistributed(layers.Dense(output_features), name="output_projection")(decoded)
    return keras.Model(inputs, outputs, name="seq2seq")


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """Create sinusoidal positional encodings for Transformer inputs."""
    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    channels = tf.range(depth, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (channels // 2)) / tf.cast(depth, tf.float32))
    angles = positions * angle_rates
    encoding = tf.where(tf.cast(channels, tf.int32) % 2 == 0, tf.sin(angles), tf.cos(angles))
    return encoding[tf.newaxis, ...]


def _transformer_encoder_block(inputs: tf.Tensor, embed_dim: int, heads: int, feed_forward_dim: int) -> tf.Tensor:
    attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim // heads)(inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    feed_forward = layers.Dense(feed_forward_dim, activation="gelu")(x)
    feed_forward = layers.Dense(embed_dim)(feed_forward)
    return layers.LayerNormalization(epsilon=1e-6)(x + feed_forward)


def build_transformer_classifier(
    input_shape: tuple[int, int],
    num_classes: int,
    embed_dim: int = 32,
    heads: int = 4,
    feed_forward_dim: int = 64,
) -> keras.Model:
    """Build a Transformer encoder classifier with positional encoding."""
    if embed_dim % heads != 0:
        raise ValueError("embed_dim must be divisible by heads")
    inputs = keras.Input(shape=input_shape, name="tokens")
    x = layers.Dense(embed_dim)(inputs)
    x = x + positional_encoding(input_shape[0], embed_dim)
    x = _transformer_encoder_block(x, embed_dim, heads, feed_forward_dim)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="transformer_classifier")
