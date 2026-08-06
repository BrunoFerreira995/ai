"""Vocabulary aliases and causal-language dataset utilities."""

from __future__ import annotations

import numpy as np

from .tokenizer import CharacterTokenizer


def make_causal_examples(tokenizer: CharacterTokenizer, texts: list[str], sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create teacher-forcing input/target pairs for next-token prediction."""
    rows, targets = [], []
    for text in texts:
        ids = tokenizer.encode(text, max_length=sequence_length + 1)
        if len(ids) < 2:
            continue
        ids = ids + [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids[:-1])
        targets.append(ids[1:])
    if not rows:
        raise ValueError("texts must contain at least one non-empty example")
    return np.asarray(rows, dtype=np.int32), np.asarray(targets, dtype=np.int32)
