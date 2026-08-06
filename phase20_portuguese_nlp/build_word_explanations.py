#!/usr/bin/env python3
"""Build structural word explanations for Causal LM training."""

from __future__ import annotations

import argparse

from .word_explanations import write_word_explanations


def main() -> None:
    parser = argparse.ArgumentParser(description="Explica todas as palavras do dicionário")
    parser.add_argument("--dictionary-dir", default="data/pt_br_dictionary")
    parser.add_argument("--output", default="data/word_explanations.jsonl")
    parser.add_argument("--max-subsequence-length", type=int, default=6)
    args = parser.parse_args()
    count = write_word_explanations(args.output, args.dictionary_dir, args.max_subsequence_length)
    print(f"Explicações geradas: {count} palavras -> {args.output}")


if __name__ == "__main__":
    main()

