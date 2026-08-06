"""Question/answer dataset loading and prompt formatting."""

from __future__ import annotations

import json
from pathlib import Path


def load_qa_dataset(path: str | Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question") or not row.get("answer"):
            raise ValueError(f"linha {line_number}: question e answer são obrigatórios")
        rows.append({"question": str(row["question"]), "answer": str(row["answer"])})
    if not rows:
        raise ValueError("dataset QA vazio")
    return rows


def format_qa(row: dict[str, str]) -> str:
    return f"Pergunta: {row['question']}\nResposta: {row['answer']}"
