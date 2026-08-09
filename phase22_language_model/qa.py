"""Question/answer dataset loading and prompt formatting."""

from __future__ import annotations

import json
import gzip
from pathlib import Path


def load_qa_dataset(path: str | Path, max_examples: int | None = None) -> list[dict[str, str]]:
    rows = []
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
      lines = enumerate(handle, 1)
      for line_number, line in lines:
        if max_examples is not None and len(rows) >= max_examples:
            break
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question") or not row.get("answer"):
            raise ValueError(f"linha {line_number}: question e answer são obrigatórios")
        normalized = {key: str(value) for key, value in row.items() if value is not None}
        normalized["question"] = str(row["question"])
        normalized["answer"] = str(row["answer"])
        rows.append(normalized)
    if not rows:
        raise ValueError("dataset QA vazio")
    return rows


def format_qa(row: dict[str, str]) -> str:
    return f"Pergunta: {row['question']}\nResposta: {row['answer']}"
