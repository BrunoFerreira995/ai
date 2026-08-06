"""Flash-attention capability detection with a safe TensorFlow fallback."""

from __future__ import annotations

import tensorflow as tf


def flash_attention_available() -> bool:
    """Return whether this TensorFlow build exposes a fused attention op."""
    return hasattr(tf.raw_ops, "ScaledDotProductAttention")


def scaled_dot_product_attention(query, key, value, mask=None):
    """Use fused attention when available, otherwise standard matmul attention."""
    if flash_attention_available():
        return tf.raw_ops.ScaledDotProductAttention(query=query, key=key, value=value, attn_mask=mask, is_causal=mask is None)
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))
    if mask is not None:
        scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
    return tf.matmul(tf.nn.softmax(scores, axis=-1), value)
