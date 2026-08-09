#!/usr/bin/env python3
"""Train the local decoder-only LM on one UTF-8 text example per line."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import tensorflow as tf

from .model import CausalLMConfig, DecoderOnlyCausalLM
from .tokenizer import CharacterTokenizer, SentencePieceTokenizer
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
    parser.add_argument("--validation-text", default="", help="arquivo independente de validação")
    parser.add_argument("--validation-split", type=float, default=0.02)
    parser.add_argument("--output-dir", default="artifacts/language_model")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer", choices=("character", "sentencepiece"), default="character")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-dir", default="", help="checkpoint anterior para continued pretraining")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    args = parser.parse_args()

    text_path = Path(args.text)
    if not text_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {text_path}")
    tf.keras.utils.set_random_seed(args.seed)
    texts = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.validation_text:
        validation_path = Path(args.validation_text)
        validation = [line.strip() for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        split = max(1, int(len(texts) * args.validation_split))
        validation, texts = texts[:split], texts[split:]
    if not texts or not validation:
        raise ValueError("treino e validação precisam conter exemplos")
    if args.init_dir:
        init_dir = Path(args.init_dir)
        from .tokenizer import load_tokenizer
        tokenizer = load_tokenizer(init_dir / "tokenizer.json")
    else:
        tokenizer_cls = SentencePieceTokenizer if args.tokenizer == "sentencepiece" else CharacterTokenizer
        tokenizer = tokenizer_cls().fit(texts, max_tokens=args.vocab_size) if args.tokenizer == "sentencepiece" else tokenizer_cls().fit(texts)
    inputs, targets = make_causal_examples(tokenizer, texts, args.sequence_length)
    validation_inputs, validation_targets = make_causal_examples(tokenizer, validation, args.sequence_length)
    if args.init_dir:
        config = CausalLMConfig(**json.loads((Path(args.init_dir) / "config.json").read_text(encoding="utf-8")))
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError("tokenizer e checkpoint inicial têm vocabulários incompatíveis")
    else:
        config = CausalLMConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=args.sequence_length)
    model = DecoderOnlyCausalLM(config)
    model(tf.zeros((1, args.sequence_length), dtype=tf.int32))
    if args.init_dir:
        model.load_weights(Path(args.init_dir) / "model.weights.h5")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss=masked_loss(tokenizer.pad_id))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, min_delta=args.min_delta,
        restore_best_weights=True,
    )
    history = model.fit(inputs, targets, validation_data=(validation_inputs, validation_targets), batch_size=args.batch_size, epochs=args.epochs, callbacks=[early_stopping], verbose=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    model.save_weights(output_dir / "model.weights.h5")
    metrics = {key: [float(value) for value in values] for key, values in history.history.items()}
    metrics["perplexity"] = [math.exp(min(value, 50.0)) for value in metrics["loss"]]
    metrics["validation_perplexity"] = [math.exp(min(value, 50.0)) for value in metrics["val_loss"]]
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Language model saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
