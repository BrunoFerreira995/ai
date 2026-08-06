#!/usr/bin/env python3
"""Train the local decoder-only LM on one UTF-8 text example per line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from .model import CausalLMConfig, DecoderOnlyCausalLM
from .tokenizer import CharacterTokenizer
from .vocabulary import make_causal_examples


def masked_loss(pad_id: int):
    def loss(targets, logits):
        values = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
        mask = tf.cast(tf.not_equal(targets, pad_id), values.dtype)
        return tf.reduce_sum(values * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local causal language model")
    parser.add_argument("--text", default="data/lm_train.txt", help="UTF-8 file with one training example per line")
    parser.add_argument("--output-dir", default="artifacts/language_model")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    text_path = Path(args.text)
    if not text_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {text_path}")
    texts = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = CharacterTokenizer().fit(texts)
    inputs, targets = make_causal_examples(tokenizer, texts, args.sequence_length)
    config = CausalLMConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=args.sequence_length)
    model = DecoderOnlyCausalLM(config)
    model(tf.zeros((1, args.sequence_length), dtype=tf.int32))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=masked_loss(tokenizer.pad_id))
    model.fit(inputs, targets, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    model.save_weights(output_dir / "model.weights.h5")
    print(f"Language model saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
