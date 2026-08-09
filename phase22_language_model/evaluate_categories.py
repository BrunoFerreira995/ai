#!/usr/bin/env python3
"""Evaluate causal-LM loss and perplexity grouped by QA category."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import tensorflow as tf

from .model import CausalLMConfig, DecoderOnlyCausalLM
from .qa import format_qa, load_qa_dataset
from .tokenizer import load_tokenizer
from .train_lm import masked_loss
from .vocabulary import make_causal_examples


def evaluate(model, tokenizer, rows, sequence_length):
    grouped = defaultdict(list)
    for row in rows:
        category = row.get("category") or row.get("subject") or row.get("domain") or "uncategorized"
        grouped[category].append(row)
    result = {}
    for category, values in sorted(grouped.items()):
        inputs, targets = make_causal_examples(tokenizer, [format_qa(row) for row in values], sequence_length)
        loss = float(model.evaluate(inputs, targets, verbose=0, return_dict=True)["loss"])
        result[category] = {"examples": len(values), "loss": loss, "perplexity": math.exp(min(loss, 50.0))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Métricas de linguagem por categoria")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    config = CausalLMConfig(**json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
    tokenizer = load_tokenizer(model_dir / "tokenizer.json")
    model = DecoderOnlyCausalLM(config)
    model(tf.zeros((1, config.max_sequence_length), dtype=tf.int32))
    model.compile(optimizer="adam", loss=masked_loss(tokenizer.pad_id))
    model.load_weights(model_dir / "model.weights.h5")
    rows = [row for path in args.data for row in load_qa_dataset(path)]
    report = evaluate(model, tokenizer, rows, config.max_sequence_length)
    output = Path(args.output) if args.output else model_dir / "category_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Category metrics: {output}")


if __name__ == "__main__":
    main()
