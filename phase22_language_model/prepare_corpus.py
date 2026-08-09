#!/usr/bin/env python3
"""Prepare a clean line-oriented text corpus for causal-LM pretraining.

The command intentionally only processes files supplied by the user. It does
not treat bibliographic metadata as article content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def read_text_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line) if suffix == ".jsonl" else None
            if isinstance(value, str):
                rows.append(value)
            elif isinstance(value, dict):
                for key in ("text", "content", "article", "body"):
                    if value.get(key):
                        rows.append(str(value[key]))
                        break
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload if isinstance(payload, list) else [payload]
            rows = [str(item.get("text", item.get("content", ""))) for item in values if isinstance(item, dict)]
        return rows
    raw = path.read_text(encoding="utf-8")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    if not paragraphs:
        return []
    # Keep examples bounded without turning every line into a separate sample.
    blocks: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        current: list[str] = []
        size = 0
        for word in words:
            if current and size + len(word) + 1 > 2000:
                blocks.append(" ".join(current))
                current, size = [], 0
            current.append(word)
            size += len(word) + 1
        if current:
            blocks.append(" ".join(current))
    return blocks


def normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def prepare(paths: list[str], validation_paths: list[str], validation_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    def collect(names: list[str]) -> list[str]:
        values = []
        for name in names:
            path = Path(name)
            if not path.is_file():
                missing.append(str(path))
                continue
            values.extend(normalize(item) for item in read_text_file(path))
        seen = set()
        result = []
        for value in values:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if value and digest not in seen:
                seen.add(digest)
                result.append(value)
        return result

    missing: list[str] = []
    train = collect(paths)
    validation = collect(validation_paths) if validation_paths else []
    if missing:
        raise FileNotFoundError("Corpus não encontrado:\n- " + "\n- ".join(missing))
    if not validation:
        rng = random.Random(seed)
        rng.shuffle(train)
        count = max(1, int(len(train) * validation_ratio))
        validation, train = train[:count], train[count:]
    overlap = set(train) & set(validation)
    if overlap:
        validation = [value for value in validation if value not in overlap]
        print(f"Aviso: {len(overlap)} exemplos duplicados removidos da validação")
    if not train or not validation:
        raise ValueError("treino e validação precisam conter exemplos")
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara corpus textual para pretraining causal")
    parser.add_argument("--input", action="append", required=True, help="arquivo .txt/.json/.jsonl; pode ser repetido")
    parser.add_argument("--validation-input", action="append", default=[], help="corpus independente de validação")
    parser.add_argument("--train-output", default="data/corpus_train.txt")
    parser.add_argument("--validation-output", default="data/corpus_validation.txt")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        raise SystemExit("--validation-ratio deve estar entre 0 e 1")
    train, validation = prepare(args.input, args.validation_input, args.validation_ratio, args.seed)
    for output, values in ((args.train_output, train), (args.validation_output, validation)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    print(json.dumps({"train_examples": len(train), "validation_examples": len(validation)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
