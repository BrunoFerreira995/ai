"""Deterministic tokenizer and vocabulary for the local causal LM."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class CharacterTokenizer:
    """Unicode character tokenizer suitable for a small Portuguese prototype.

    A production model should replace this with a BPE/SentencePiece tokenizer;
    this implementation is intentionally dependency-free and trainable locally.
    """

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = list(vocabulary or SPECIAL_TOKENS)
        for token in SPECIAL_TOKENS:
            if token not in self.vocabulary:
                self.vocabulary.insert(0, token)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def fit(self, texts: list[str], max_tokens: int | None = None) -> "CharacterTokenizer":
        counts = Counter(character for text in texts for character in text)
        tokens = [token for token, _ in counts.most_common(max_tokens)] if max_tokens else list(counts)
        self.vocabulary = list(SPECIAL_TOKENS) + [token for token in tokens if token not in SPECIAL_TOKENS]
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        return self

    def encode(self, text: str, *, add_special_tokens: bool = True, max_length: int | None = None) -> list[int]:
        values = [self.token_to_id.get(character, self.unk_id) for character in text]
        if add_special_tokens:
            values = [self.bos_id, *values, self.eos_id]
        if max_length is not None:
            values = values[:max_length]
            if add_special_tokens and values and values[-1] != self.eos_id:
                values[-1] = self.eos_id
        return values

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ignored = set(SPECIAL_TOKENS) if skip_special_tokens else set()
        return "".join(self.vocabulary[index] for index in ids if 0 <= index < len(self.vocabulary) and self.vocabulary[index] not in ignored)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocabulary": self.vocabulary}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(payload["vocabulary"]))
