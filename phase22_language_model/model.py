"""Decoder-only Transformer with RoPE, multi-query attention, and KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class CausalLMConfig:
    vocab_size: int
    max_sequence_length: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 1
    intermediate_size: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=(input_shape[-1],), initializer="ones")

    def call(self, inputs):
        values = tf.cast(inputs, tf.float32)
        values = values * tf.math.rsqrt(tf.reduce_mean(tf.square(values), axis=-1, keepdims=True) + self.epsilon)
        return tf.cast(values, inputs.dtype) * self.scale


def apply_rope(query: tf.Tensor, key: tf.Tensor, position_ids: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply rotary position embeddings to [batch, heads, sequence, dimension]."""
    dimension = tf.shape(query)[-1]
    half = dimension // 2
    frequencies = 1.0 / tf.pow(10000.0, tf.range(half, dtype=tf.float32) / tf.cast(half, tf.float32))
    angles = tf.cast(position_ids[..., None], tf.float32) * frequencies
    cos, sin = tf.cos(angles)[:, None, :, :], tf.sin(angles)[:, None, :, :]

    def rotate(values):
        first, second = values[..., :half], values[..., half:]
        return tf.concat([first * cos - second * sin, first * sin + second * cos], axis=-1)

    return rotate(query), rotate(key)


class MultiQueryAttention(tf.keras.layers.Layer):
    """Multi-query attention: many Q heads share a smaller number of KV heads."""

    def __init__(self, config: CausalLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        self.query = tf.keras.layers.Dense(config.hidden_size, use_bias=False)
        self.key = tf.keras.layers.Dense(config.num_kv_heads * self.head_dim, use_bias=False)
        self.value = tf.keras.layers.Dense(config.num_kv_heads * self.head_dim, use_bias=False)
        self.out_projection = tf.keras.layers.Dense(config.hidden_size, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, inputs, position_ids, attention_mask=None, past_key_value=None, training=False):
        batch = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        query = tf.reshape(self.query(inputs), [batch, length, self.config.num_heads, self.head_dim])
        key = tf.reshape(self.key(inputs), [batch, length, self.config.num_kv_heads, self.head_dim])
        value = tf.reshape(self.value(inputs), [batch, length, self.config.num_kv_heads, self.head_dim])
        query, key = apply_rope(tf.transpose(query, [0, 2, 1, 3]), tf.transpose(key, [0, 2, 1, 3]), position_ids)
        value = tf.transpose(value, [0, 2, 1, 3])
        if past_key_value is not None:
            key = tf.concat([past_key_value[0], key], axis=2)
            value = tf.concat([past_key_value[1], value], axis=2)
        repeats = self.config.num_heads // self.config.num_kv_heads
        key = tf.repeat(key, repeats, axis=1)
        value = tf.repeat(value, repeats, axis=1)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        key_length = tf.shape(key)[2]
        query_length = tf.shape(query)[2]
        current_causal = tf.linalg.band_part(tf.ones([query_length, query_length]), -1, 0)
        causal = current_causal if past_key_value is None else tf.concat([tf.ones([query_length, key_length - query_length]), current_causal], axis=1)
        scores = tf.where(causal[None, None, :, :] > 0, scores, tf.cast(-1e9, scores.dtype))
        if attention_mask is not None:
            scores += (1.0 - tf.cast(attention_mask[:, None, None, :], scores.dtype)) * -1e9
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, value)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch, query_length, self.config.hidden_size])
        return self.out_projection(output), (key[:, : self.config.num_kv_heads], value[:, : self.config.num_kv_heads])


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, config: CausalLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = RMSNorm(name="attention_norm")
        self.attention = MultiQueryAttention(config, name="mqa")
        self.norm2 = RMSNorm(name="ffn_norm")
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(config.intermediate_size, activation="gelu"), tf.keras.layers.Dense(config.hidden_size)], name="ffn")

    def call(self, inputs, position_ids, attention_mask=None, past_key_value=None, training=False):
        attended, cache = self.attention(self.norm1(inputs), position_ids, attention_mask, past_key_value, training)
        hidden = inputs + attended
        return hidden + self.ffn(self.norm2(hidden), training=training), cache


class DecoderOnlyCausalLM(tf.keras.Model):
    """Trainable decoder-only language model."""

    def __init__(self, config: CausalLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = [DecoderBlock(config, name=f"decoder_block_{i}") for i in range(config.num_layers)]
        self.norm = RMSNorm(name="final_norm")
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False, name="lm_head")

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.__dict__})
        return config

    @classmethod
    def from_config(cls, config):
        model_config = CausalLMConfig(**config.pop("config"))
        return cls(model_config, **config)

    def call(self, input_ids, attention_mask=None, training=False):
        return self.forward(input_ids, attention_mask=attention_mask, training=training)["logits"]

    def forward(self, input_ids, attention_mask=None, past_key_values=None, training=False):
        input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        length = tf.shape(input_ids)[1]
        past_length = 0 if past_key_values is None else tf.shape(past_key_values[0][0])[2]
        position_ids = tf.broadcast_to(tf.range(past_length, past_length + length)[None, :], [tf.shape(input_ids)[0], length])
        hidden = self.embedding(input_ids)
        caches = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, cache = block(hidden, position_ids, attention_mask, past, training)
            caches.append(cache)
        return {"logits": self.lm_head(self.norm(hidden)), "past_key_values": tuple(caches)}
