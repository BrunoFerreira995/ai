#!/usr/bin/env python3
"""Train the local causal LM with JSONL question/answer examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from .model import CausalLMConfig, DecoderOnlyCausalLM
from .qa import format_qa, load_qa_dataset
from phase20_portuguese_nlp.dictionary import load_portuguese_words
from .tokenizer import CharacterTokenizer
from .train_lm import masked_loss
from .vocabulary import make_causal_examples


def pack_dictionary_words(words: set[str], sequence_length: int) -> list[str]:
    """Pack every dictionary entry into bounded text chunks for lexical pretraining."""
    chunks: list[str] = []
    current = ""
    for word in sorted(words):
        candidate = f"{current} {word}".strip()
        if len(candidate) > sequence_length - 2 and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal LM with QA JSONL")
    parser.add_argument("--data", default="data/qa_train.jsonl")
    parser.add_argument("--output-dir", default="artifacts/language_model")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dictionary-dir", default="data/pt_br_dictionary", help="coleção lexical pt-BR opcional")
    parser.add_argument("--dictionary-epochs", type=int, default=1, help="épocas de pré-treino lexical")
    parser.add_argument("--word-explanations", default="", help="JSONL opcional com explicações estruturais das palavras")
    args = parser.parse_args()

    rows = load_qa_dataset(args.data)
    qa_texts = [format_qa(row) for row in rows]
    dictionary_texts: list[str] = []
    dictionary_words: set[str] = set()
    try:
        dictionary_words = load_portuguese_words(args.dictionary_dir)
        dictionary_texts = pack_dictionary_words(dictionary_words, args.sequence_length)
    except FileNotFoundError:
        print("Aviso: dicionário pt-BR não encontrado; pré-treino lexical ignorado.")
    explanation_texts: list[str] = []
    if args.word_explanations:
        explanation_path = Path(args.word_explanations)
        if not explanation_path.is_file():
            raise FileNotFoundError(f"Arquivo de explicações não encontrado: {explanation_path}")
        explanation_texts = [
            f"Pergunta: {row['question']}\nResposta: {row['answer']}"
            for line in explanation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        ]
        print(f"Explicações de palavras: {len(explanation_texts)} exemplos")
    tokenizer = CharacterTokenizer().fit(qa_texts + dictionary_texts + explanation_texts)
    inputs, targets = make_causal_examples(tokenizer, qa_texts, args.sequence_length)
    config = CausalLMConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=args.sequence_length)
    model = DecoderOnlyCausalLM(config)
    model(tf.zeros((1, args.sequence_length), dtype=tf.int32))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=masked_loss(tokenizer.pad_id))
    if dictionary_texts and args.dictionary_epochs > 0:
        dictionary_inputs, dictionary_targets = make_causal_examples(tokenizer, dictionary_texts, args.sequence_length)
        print(f"Pré-treino lexical: {len(dictionary_words)} palavras em {len(dictionary_texts)} sequências")
        model.fit(dictionary_inputs, dictionary_targets, batch_size=args.batch_size, epochs=args.dictionary_epochs, verbose=1)
    if explanation_texts:
        explanation_inputs, explanation_targets = make_causal_examples(tokenizer, explanation_texts, args.sequence_length)
        model.fit(explanation_inputs, explanation_targets, batch_size=args.batch_size, epochs=1, verbose=1)
    print(f"Treino QA: {len(rows)} exemplos")
    model.fit(inputs, targets, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    model.save_weights(output_dir / "model.weights.h5")
    (output_dir / "training_data.json").write_text(json.dumps({"qa_examples": len(rows), "dictionary_words": len(dictionary_words), "dictionary_sequences": len(dictionary_texts), "word_explanation_examples": len(explanation_texts)}, indent=2), encoding="utf-8")
    try:
        dictionary_size = len(load_portuguese_words(args.dictionary_dir))
        (output_dir / "dictionary_stats.json").write_text(json.dumps({"source": args.dictionary_dir, "words": dictionary_size}, indent=2), encoding="utf-8")
        print(f"Vocabulário pt-BR carregado: {dictionary_size} palavras")
    except FileNotFoundError:
        print("Aviso: dicionário pt-BR não encontrado; treino continua sem ele.")
    print(f"QA Causal LM salvo em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
