#!/usr/bin/env python3
"""Build a QA dataset from educational knowledge and scientific references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .education import EDUCATIONAL_KNOWLEDGE
from .lexicon import SYNONYM_DICTIONARY, VERB_DICTIONARY
from .literacy import literacy_response
from .epistemology import epistemology_rows
from .portuguese_language import PORTUGUESE_TOPICS

REFERENCE_LIMIT_PER_SUBJECT = 4


def build_reference_rows(reference_path: str | Path = "docs/educational_references.md") -> list[dict[str, str]]:
    """Turn the educational bibliography into Portuguese QA training examples."""
    path = Path(reference_path)
    if not path.is_file():
        return []
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, str]] = []
        for key, references in records.items():
            if " — " not in key:
                continue
            subject, topic = key.split(" — ", 1)
            for index, item in enumerate(references, 1):
                reference = f"{item['title']} ({item.get('authors', '')}; {item.get('year', '')}; DOI: {item['doi']})"
                rows.append({
                    "question": f"Qual referência científica número {index} aborda {topic} em {subject}?",
                    "answer": f"A referência {index} para {topic} em {subject} é: {reference}.",
                })
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: dict[str, list[str]] = {}
    subsection_blocks: dict[str, list[str]] = {}
    current: str | None = None
    current_subsection: str | None = None
    for line in lines:
        subsection = re.match(r"^### (.+ — subtemas)$", line)
        if subsection:
            current_subsection = subsection.group(1).strip()
            subsection_blocks[current_subsection] = []
            continue
        heading = re.match(r"^## (.+)$", line)
        if heading:
            current = heading.group(1).strip()
            sections[current] = []
            current_subsection = None
            continue
        if current_subsection:
            subsection_blocks[current_subsection].append(line)
            continue
        if current:
            sections[current].append(line)

    references: dict[str, list[str]] = {}
    for subject, content in sections.items():
        if " — subtemas" in subject:
            continue
        entries = []
        for line in content:
            match = re.match(r"^\d+\. \[(.*)\]\((.*)\)$", line.strip())
            if match:
                entries.append(f"{match.group(1)} ({match.group(2)})")
        if entries:
            references[subject] = entries[:REFERENCE_LIMIT_PER_SUBJECT]

    rows: list[dict[str, str]] = []
    for heading, content in subsection_blocks.items():
        subject = heading.split(" — subtemas", 1)[0]
        source_references = references.get(subject, [])
        if len(source_references) < REFERENCE_LIMIT_PER_SUBJECT:
            continue
        for line in content:
            if not line.startswith("|") or line.startswith("| ---") or "Subtema" in line:
                continue
            columns = [column.strip() for column in line.strip("|").split("|")]
            if not columns or not columns[0]:
                continue
            topic = columns[0]
            formatted = "; ".join(f"{index}. {reference}" for index, reference in enumerate(source_references, 1))
            questions = (
                f"Quais artigos científicos apoiam {topic} em {subject}?",
                f"Indique referências para estudar {topic} em {subject}.",
                f"Que pesquisas são relevantes para {topic} em {subject}?",
            )
            for question in questions:
                rows.append({
                    "question": question,
                    "answer": f"Para {topic} em {subject}, referências selecionadas são: {formatted}.",
                })
    return rows


def build_rows(reference_path: str | Path = "docs/educational_references.md") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subject, content in EDUCATIONAL_KNOWLEDGE.items():
        rows.append({"question": f"O que é {subject}?", "answer": f"{subject} é {content['summary']}."})
        for topic in content["topics"]:
            rows.append({"question": f"Explique {topic} em {subject}.", "answer": f"Em {subject}, {topic} é um tema importante de estudo. {content['summary']}."})
    for topic, content in PORTUGUESE_TOPICS.items():
        rows.append({"question": f"O que estudar em {topic}?", "answer": f"Em {topic}, estude {content['summary']}."})
    for word in ("casa", "escola", "matemática", "português", "computador", "biologia"):
        analysis = literacy_response(word)
        rows.append({"question": f"Separe a palavra {word} em sílabas.", "answer": analysis})
    rows.extend([
        {"question": "O que é uma letra?", "answer": "Letra é um símbolo usado para representar os sons da fala na escrita."},
        {"question": "O que é uma sílaba?", "answer": "Sílaba é cada parte sonora em que uma palavra pode ser dividida."},
        {"question": "O que é uma palavra?", "answer": "Palavra é uma unidade da língua formada por letras ou sons e com significado em um contexto."},
    ])
    rows.extend(epistemology_rows())
    rows.extend(build_reference_rows(reference_path))
    for verb, forms in VERB_DICTIONARY.items():
        rows.append({"question": f"Como usar o verbo {verb}?", "answer": f"O verbo {verb} indica uma ação ou estado. Algumas formas são: {', '.join(forms[:6])}."})
    for word, synonyms in SYNONYM_DICTIONARY.items():
        rows.append({"question": f"Qual é um sinônimo de {word}?", "answer": f"Sinônimos de {word}: {', '.join(synonyms)}."})
    return rows


def augment_with_dictionary(rows: list[dict[str, str]], variants_per_row: int = 4) -> list[dict[str, str]]:
    """Stack verb forms and synonyms to increase linguistic coverage."""
    dictionary = {
        canonical: list(forms) for canonical, forms in {**VERB_DICTIONARY, **SYNONYM_DICTIONARY}.items()
    }
    augmented: list[dict[str, str]] = []
    for row in rows:
        augmented.append(row)
        candidates: list[dict[str, str]] = []
        for canonical, variants in dictionary.items():
            if not re.search(rf"\b{re.escape(canonical)}\b", row["question"], flags=re.IGNORECASE):
                continue
            for variant in variants:
                question = re.sub(rf"\b{re.escape(canonical)}\b", variant, row["question"], flags=re.IGNORECASE, count=1)
                if question != row["question"]:
                    candidates.append({"question": question, "answer": row["answer"]})
        augmented.extend(candidates[: max(0, variants_per_row - 1)])
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 20 QA dataset")
    parser.add_argument("--output", default="data/qa_phase20.jsonl")
    parser.add_argument("--dictionary-variants", type=int, default=4, help="quantas variações por pergunta usar")
    parser.add_argument("--references", default="docs/educational_references.md", help="bibliografia educacional em Markdown")
    args = parser.parse_args()
    rows = augment_with_dictionary(build_rows(args.references), args.dictionary_variants)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Dataset Phase 20 criado: {output} ({len(rows)} exemplos)")


if __name__ == "__main__":
    main()
