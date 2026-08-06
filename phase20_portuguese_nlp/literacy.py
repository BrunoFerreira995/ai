"""Basic Portuguese literacy knowledge: letters, syllables, and words."""

from __future__ import annotations

import re
import unicodedata

ALPHABET = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VOWELS = set("aeiouáéíóúâêôãõàü")


def normalize_word(word: str) -> str:
    return re.sub(r"[^A-Za-zÀ-ÿ]", "", unicodedata.normalize("NFC", word).lower())


def split_syllables(word: str) -> list[str]:
    """Apply a practical heuristic syllabification for common Portuguese words."""
    word = normalize_word(word)
    if not word:
        return []
    groups: list[str] = []
    start = 0
    vowels = [index for index, char in enumerate(word) if char in VOWELS]
    if not vowels:
        return [word]
    onset_clusters = {"br", "cr", "dr", "fr", "gr", "pr", "tr", "bl", "cl", "fl", "gl", "pl"}
    for position, current_vowel in enumerate(vowels):
        next_vowel = vowels[position + 1] if position + 1 < len(vowels) else len(word)
        if next_vowel == len(word):
            boundary = len(word)
        else:
            between = word[current_vowel + 1 : next_vowel]
            if len(between) == 0:
                continue
            cluster = between[-2:]
            boundary = next_vowel - (2 if len(between) >= 2 and cluster in onset_clusters else 1)
        if boundary > start:
            groups.append(word[start:boundary])
            start = boundary
    if start < len(word):
        groups.append(word[start:])
    return groups or [word]


def analyze_word(word: str) -> dict[str, object]:
    normalized = normalize_word(word)
    syllables = split_syllables(normalized)
    count = len(syllables)
    names = {0: "sem sílabas", 1: "monossílaba", 2: "dissílaba", 3: "trissílaba"}
    return {"word": normalized, "letters": list(normalized), "letter_count": len(normalized), "syllables": syllables, "syllable_count": count, "classification": names.get(count, "polissílaba")}


def literacy_response(word: str) -> str:
    analysis = analyze_word(word)
    letters = ", ".join(analysis["letters"])
    return f"A palavra '{analysis['word']}' tem as letras {letters}; separação silábica: {'-'.join(analysis['syllables'])}; classificação: {analysis['classification']}."
