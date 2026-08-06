"""Loader for the downloaded pt-BR lexical collection."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_DICTIONARY_DIR = Path("data/pt_br_dictionary")


def load_portuguese_words(dictionary_dir: str | Path = DEFAULT_DICTIONARY_DIR) -> set[str]:
    """Load lexical entries and conjugations from the MIT-licensed collection."""
    root = Path(dictionary_dir)
    files = [root / "lexico", root / "conjugações", root / "listas" / "verbos"]
    words: set[str] = set()
    for path in files:
        if not path.is_file():
            continue
        words.update(word.strip().lower() for word in path.read_text(encoding="utf-8").splitlines() if re.fullmatch(r"[a-zà-ÿ-]+", word.strip().lower()))
    if not words:
        raise FileNotFoundError(f"Nenhum arquivo de dicionário encontrado em {root}")
    return words


def is_portuguese_word(word: str, dictionary_dir: str | Path = DEFAULT_DICTIONARY_DIR) -> bool:
    return word.strip().lower() in load_portuguese_words(dictionary_dir)
