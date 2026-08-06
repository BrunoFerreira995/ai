"""Create structural Portuguese word explanations from the local lexicon.

The downloaded collection is a word-form lexicon, not a dictionary of
definitions. The generated explanations therefore describe spelling,
syllables and contiguous character subsequences without inventing meanings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .dictionary import load_portuguese_words
from .literacy import analyze_word


def contiguous_subsequences(word: str, min_length: int = 2, max_length: int = 6) -> list[str]:
    """Return unique contiguous character n-grams in reading order.

    Non-contiguous subsequences are intentionally not generated: a word with
    ``n`` letters has exponentially many of them. Contiguous subsequences are
    useful, bounded, and correspond to prefixes, roots and syllable fragments.
    """
    word = word.strip().lower()
    result: list[str] = []
    seen: set[str] = set()
    upper = min(len(word), max_length)
    for size in range(max(1, min_length), upper + 1):
        for start in range(0, len(word) - size + 1):
            part = word[start : start + size]
            if part not in seen:
                seen.add(part)
                result.append(part)
    return result


def explain_word(word: str, max_subsequence_length: int = 6) -> dict[str, object]:
    analysis = analyze_word(word)
    subsequences = contiguous_subsequences(analysis["word"], max_length=max_subsequence_length)
    letters = ", ".join(analysis["letters"])
    syllables = "-".join(analysis["syllables"])
    explanation = (
        f"A palavra '{analysis['word']}' é formada pelas letras {letters}. "
        f"Ela pode ser separada em sílabas como {syllables} e é uma "
        f"{analysis['classification']}. "
        f"Subsequências contíguas encontradas: {', '.join(subsequences) or 'nenhuma'}."
    )
    return {
        "word": analysis["word"],
        "letters": analysis["letters"],
        "syllables": analysis["syllables"],
        "classification": analysis["classification"],
        "subsequences": subsequences,
        "question": f"Explique a palavra {analysis['word']} usando suas letras e subsequências.",
        "answer": explanation,
    }


def iter_word_explanations(words: set[str], max_subsequence_length: int = 6) -> Iterator[dict[str, object]]:
    for word in sorted(words):
        if word:
            yield explain_word(word, max_subsequence_length)


def write_word_explanations(
    output: str | Path,
    dictionary_dir: str | Path,
    max_subsequence_length: int = 6,
) -> int:
    """Write one JSONL explanation for every word in the lexicon."""
    words = load_portuguese_words(dictionary_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in iter_word_explanations(words, max_subsequence_length):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

