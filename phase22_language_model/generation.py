"""Greedy, sampling, and beam-search autoregressive generation."""

from __future__ import annotations

import tensorflow as tf


def _next_token(logits, temperature=1.0, top_k=0):
    logits = logits / max(float(temperature), 1e-5)
    if top_k:
        values = tf.math.top_k(logits, k=top_k).values[:, -1, None]
        logits = tf.where(logits < values, tf.cast(-1e9, logits.dtype), logits)
    return tf.random.categorical(logits[:, -1, :], 1)[:, 0]


def generate(model, input_ids, *, max_new_tokens: int = 32, eos_token_id: int | None = None, temperature: float = 0.0, top_k: int = 0):
    """Generate tokens autoregressively, reusing the model's KV cache."""
    tokens = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    cache = None
    for _ in range(max_new_tokens):
        current = tokens if cache is None else tokens[:, -1:]
        result = model.forward(current, past_key_values=cache)
        next_token = tf.argmax(result["logits"][:, -1, :], axis=-1, output_type=tf.int32) if temperature <= 0 else _next_token(result["logits"], temperature, top_k)
        tokens = tf.concat([tokens, next_token[:, None]], axis=1)
        cache = result["past_key_values"]
        if eos_token_id is not None and bool(tf.reduce_all(tf.equal(next_token, eos_token_id))):
            break
    return tokens


def beam_search(model, input_ids, *, num_beams: int = 3, max_new_tokens: int = 32, eos_token_id: int | None = None):
    """Simple length-normalized beam search for small local models."""
    sequences = [(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0.0)]
    for _ in range(max_new_tokens):
        candidates = []
        for sequence, score in sequences:
            result = model.forward(sequence)
            log_probs = tf.nn.log_softmax(result["logits"][:, -1, :], axis=-1)[0]
            values, indices = tf.math.top_k(log_probs, k=num_beams)
            for value, index in zip(values.numpy(), indices.numpy()):
                token = tf.fill([tf.shape(sequence)[0], 1], int(index))
                candidates.append((tf.concat([sequence, token], axis=1), score + float(value)))
        sequences = sorted(candidates, key=lambda item: item[1] / (item[0].shape[1] ** 0.7), reverse=True)[:num_beams]
        if eos_token_id is not None and all(int(sequence[0, -1]) == eos_token_id for sequence, _ in sequences):
            break
    return sequences[0][0]
