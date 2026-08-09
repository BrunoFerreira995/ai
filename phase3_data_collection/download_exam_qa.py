#!/usr/bin/env python3
"""Stream Portuguese university/exam QA datasets into deduplicated JSONL.GZ."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

EXAM_TERMS = re.compile(
    r"\b(vestibular|concurso|concursos|prova|provas|enem|fuvest|unicamp|unesp|"
    r"universidade|universitário|universitaria|faculdade|oab|edital|cargo|"
    r"servidor público|carreira pública|exame|avaliação)\b", re.IGNORECASE
)

SOURCES = (
    {"id": "recogna-nlp/EduBench", "license": "dataset card; verify terms", "split": "test", "kind": "exam"},
    {"id": "Tropic-AI/BLUEX-v2", "license": "dataset card; verify terms", "split": "train", "kind": "exam"},
    {"id": "recogna-nlp/Bode-reasoning", "license": "dataset card; verify terms", "split": "train", "kind": "exam"},
    {"id": "ruanchaves/faquad-nli", "license": "dataset card; verify terms", "split": "train", "kind": "university"},
    {"id": "emdemor/ptbr-question-and-answer", "license": "CC; retain attribution", "split": "train", "kind": "general"},
    {"id": "Jpzinn654/qa-portuguese-small", "license": "MIT according to dataset card", "split": "train", "kind": "general"},
)


def first(row: dict, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize(row: dict, source: str) -> dict[str, str] | None:
    question = first(row, "question", "subquestion_text", "input")
    answer = first(row, "answer", "expected_answer", "short_answer", "output")
    context = first(row, "context", "supporting_texts", "question_text")
    if not question or not answer:
        return None
    return {"question": question, "answer": answer, "context": context, "source": source}


def fingerprint(row: dict[str, str]) -> str:
    value = f"{row['question'].lower().strip()}\n{row['answer'].lower().strip()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collect(output: str | Path, target: int, include_general: bool) -> int:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Instale datasets: .venv/bin/python -m pip install datasets") from error

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    manifest: list[dict[str, str | int]] = []
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for source in SOURCES:
            if source["kind"] == "general" and not include_general:
                continue
            print(f"Carregando {source['id']} ({source['split']}) em streaming...")
            try:
                dataset = load_dataset(source["id"], split=source["split"], streaming=True)
                source_count = 0
                for raw in dataset:
                    row = normalize(dict(raw), source["id"])
                    if row is None:
                        continue
                    if source["kind"] in {"exam", "university"} and not EXAM_TERMS.search(f"{row['question']} {row['context']}"):
                        continue
                    key = fingerprint(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    source_count += 1
                    if count >= target:
                        break
                manifest.append({"source": source["id"], "license": source["license"], "rows": source_count})
            except Exception as error:
                print(f"Aviso: fonte ignorada ({source['id']}): {error}")
            if count >= target:
                break
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"rows": count, "sources": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pares QA salvos: {count} em {output_path}")
    print(f"Manifesto: {manifest_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa QA português de universidade e concursos")
    parser.add_argument("--output", default="data/exam_qa_1m.jsonl.gz")
    parser.add_argument("--target", type=int, default=1_000_000)
    parser.add_argument("--include-general", action="store_true", help="permite completar com QA geral em português")
    args = parser.parse_args()
    collect(args.output, args.target, args.include_general)


if __name__ == "__main__":
    main()
