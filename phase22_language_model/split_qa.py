#!/usr/bin/env python3
"""Create deterministic train/validation JSONL files from a QA dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .qa import load_qa_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Divide dataset QA em treino e validação")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", default="data/qa_train_split.jsonl")
    parser.add_argument("--validation-output", default="data/qa_validation.jsonl")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        raise SystemExit("--validation-ratio deve estar entre 0 e 1")
    rows = load_qa_dataset(args.input)
    random.Random(args.seed).shuffle(rows)
    count = max(1, int(len(rows) * args.validation_ratio))
    validation, train = rows[:count], rows[count:]
    if not train:
        raise SystemExit("dataset pequeno demais para separar treino e validação")
    for filename, values in ((args.train_output, train), (args.validation_output, validation)):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in values) + "\n", encoding="utf-8")
    print(json.dumps({"train_examples": len(train), "validation_examples": len(validation)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
