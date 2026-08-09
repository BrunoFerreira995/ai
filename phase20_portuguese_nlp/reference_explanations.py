"""Generate transparent explanations for educational references.

The source contains bibliographic metadata, not article abstracts. Explanations
therefore describe the reference's identity and its relation to the subtopic,
without claiming findings that were not downloaded or verified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def explain_reference(key: str, reference: dict[str, str], index: int) -> dict[str, object]:
    subject, topic = key.split(" — ", 1)
    title = reference.get("title", "Título não informado")
    authors = reference.get("authors", "Autores não informados")
    year = reference.get("year", "ano não informado")
    doi = reference.get("doi", "DOI não informado")
    answer = (
        f"A referência {index} apoia o estudo de {topic}, em {subject}. "
        f"Título: {title}. Autoria: {authors}. Ano: {year}. DOI: {doi}. "
        "A relação foi atribuída pelo subtema usado na busca bibliográfica; "
        "é necessário ler o artigo para confirmar método, resultados e limites."
    )
    return {
        "subject": subject,
        "topic": topic,
        "reference_index": index,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "url": reference.get("url", f"https://doi.org/{doi}"),
        "question": f"Explique a referência {index} sobre {topic} em {subject}.",
        "answer": answer,
    }


def iter_reference_explanations(data: dict[str, list[dict[str, str]]]) -> Iterator[dict[str, object]]:
    for key, references in data.items():
        for index, reference in enumerate(references, 1):
            yield explain_reference(key, reference, index)


def write_reference_explanations(
    source: str | Path = "data/educational_references_60.json",
    output: str | Path = "data/reference_explanations.jsonl",
) -> int:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in iter_reference_explanations(data):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

