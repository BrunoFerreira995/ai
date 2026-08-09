#!/usr/bin/env python3
"""Train the local causal LM with JSONL question/answer examples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import tensorflow as tf

from .model import CausalLMConfig, DecoderOnlyCausalLM
from .qa import format_qa, load_qa_dataset
from phase20_portuguese_nlp.dictionary import load_portuguese_words
from .tokenizer import CharacterTokenizer, SentencePieceTokenizer
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
    parser.add_argument("--data", action="append", default=None, help="dataset QA; pode ser repetido para combinar arquivos")
    parser.add_argument("--output-dir", default="artifacts/language_model")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dictionary-dir", default="data/pt_br_dictionary", help="coleção lexical pt-BR opcional")
    parser.add_argument("--dictionary-epochs", type=int, default=1, help="épocas de pré-treino lexical")
    parser.add_argument("--word-explanations", default="", help="JSONL opcional com explicações estruturais das palavras")
    parser.add_argument("--reference-explanations", default="", help="JSONL opcional com explicações bibliográficas")
    parser.add_argument("--max-examples", type=int, default=None, help="limita QA para controlar memória")
    parser.add_argument("--tokenizer", choices=("character", "sentencepiece"), default="character")
    parser.add_argument("--vocab-size", type=int, default=32000, help="tamanho máximo do vocabulário SentencePiece")
    parser.add_argument("--init-dir", default="", help="checkpoint de pretraining para instruction tuning")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-data", default="", help="dataset QA independente para validação")
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    data_paths = args.data or ["data/qa_train.jsonl"]
    rows: list[dict[str, str]] = []
    for data_path in data_paths:
        rows.extend(load_qa_dataset(data_path, max_examples=args.max_examples))
    if not rows:
        raise ValueError("nenhum exemplo QA foi carregado")
    validation_rows = load_qa_dataset(args.validation_data) if args.validation_data else []
    qa_texts = [format_qa(row) for row in rows]
    validation_texts = [format_qa(row) for row in validation_rows]
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
    reference_explanation_texts: list[str] = []
    if args.reference_explanations:
        reference_path = Path(args.reference_explanations)
        if not reference_path.is_file():
            raise FileNotFoundError(f"Arquivo de explicações de referências não encontrado: {reference_path}")
        reference_explanation_texts = [
            f"Pergunta: {row['question']}\nResposta: {row['answer']}"
            for line in reference_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        ]
        print(f"Explicações de referências: {len(reference_explanation_texts)} exemplos")
    all_texts = qa_texts + dictionary_texts + explanation_texts + reference_explanation_texts
    if args.init_dir:
        from .tokenizer import load_tokenizer
        init_dir = Path(args.init_dir)
        tokenizer = load_tokenizer(init_dir / "tokenizer.json")
        config = CausalLMConfig(**json.loads((init_dir / "config.json").read_text(encoding="utf-8")))
    else:
        if args.tokenizer == "sentencepiece":
            tokenizer = SentencePieceTokenizer().fit(all_texts, max_tokens=args.vocab_size)
        else:
            tokenizer = CharacterTokenizer().fit(all_texts)
        config = None
    inputs, targets = make_causal_examples(tokenizer, qa_texts, args.sequence_length)
    validation_data = None
    if validation_texts:
        validation_data = make_causal_examples(tokenizer, validation_texts, args.sequence_length)
    config = config or CausalLMConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=args.sequence_length)
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("tokenizer e checkpoint inicial têm vocabulários incompatíveis")
    model = DecoderOnlyCausalLM(config)
    model(tf.zeros((1, args.sequence_length), dtype=tf.int32))
    if args.init_dir:
        model.load_weights(Path(args.init_dir) / "model.weights.h5")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss=masked_loss(tokenizer.pad_id))
    if dictionary_texts and args.dictionary_epochs > 0:
        dictionary_inputs, dictionary_targets = make_causal_examples(tokenizer, dictionary_texts, args.sequence_length)
        print(f"Pré-treino lexical: {len(dictionary_words)} palavras em {len(dictionary_texts)} sequências")
        model.fit(dictionary_inputs, dictionary_targets, batch_size=args.batch_size, epochs=args.dictionary_epochs, verbose=1)
    if explanation_texts:
        explanation_inputs, explanation_targets = make_causal_examples(tokenizer, explanation_texts, args.sequence_length)
        model.fit(explanation_inputs, explanation_targets, batch_size=args.batch_size, epochs=1, verbose=1)
    if reference_explanation_texts:
        reference_inputs, reference_targets = make_causal_examples(tokenizer, reference_explanation_texts, args.sequence_length)
        model.fit(reference_inputs, reference_targets, batch_size=args.batch_size, epochs=1, verbose=1)
    print(f"Treino QA: {len(rows)} exemplos")
    callbacks = []
    if validation_data:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True))
    history = model.fit(inputs, targets, validation_data=validation_data, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks, verbose=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    model.save_weights(output_dir / "model.weights.h5")
    metrics = {key: [float(value) for value in values] for key, values in history.history.items()}
    metrics["perplexity"] = [math.exp(min(value, 50.0)) for value in metrics["loss"]]
    if "val_loss" in metrics:
        metrics["validation_perplexity"] = [math.exp(min(value, 50.0)) for value in metrics["val_loss"]]
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_data.json").write_text(json.dumps({"qa_examples": len(rows), "validation_examples": len(validation_rows), "dictionary_words": len(dictionary_words), "dictionary_sequences": len(dictionary_texts), "word_explanation_examples": len(explanation_texts), "reference_explanation_examples": len(reference_explanation_texts)}, indent=2), encoding="utf-8")
    try:
        dictionary_size = len(load_portuguese_words(args.dictionary_dir))
        (output_dir / "dictionary_stats.json").write_text(json.dumps({"source": args.dictionary_dir, "words": dictionary_size}, indent=2), encoding="utf-8")
        print(f"Vocabulário pt-BR carregado: {dictionary_size} palavras")
    except FileNotFoundError:
        print("Aviso: dicionário pt-BR não encontrado; treino continua sem ele.")
    print(f"QA Causal LM salvo em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
