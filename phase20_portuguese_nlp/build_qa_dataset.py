#!/usr/bin/env python3
"""Build a QA dataset from the Phase 20 educational knowledge base."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .education import EDUCATIONAL_KNOWLEDGE
from .lexicon import SYNONYM_DICTIONARY, VERB_DICTIONARY
from .literacy import literacy_response
from .portuguese_language import PORTUGUESE_TOPICS


def build_rows() -> list[dict[str, str]]:
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
    args = parser.parse_args()
    rows = augment_with_dictionary(build_rows(), args.dictionary_variants)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Dataset Phase 20 criado: {output} ({len(rows)} exemplos)")


if __name__ == "__main__":
    main()
