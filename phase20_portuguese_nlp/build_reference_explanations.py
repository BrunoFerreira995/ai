#!/usr/bin/env python3
"""Build one Portuguese explanation for every educational reference."""

from __future__ import annotations

import argparse

from .reference_explanations import write_reference_explanations


def main() -> None:
    parser = argparse.ArgumentParser(description="Explica todas as referências educacionais")
    parser.add_argument("--source", default="data/educational_references_60.json")
    parser.add_argument("--output", default="data/reference_explanations.jsonl")
    args = parser.parse_args()
    count = write_reference_explanations(args.source, args.output)
    print(f"Explicações de referências geradas: {count} -> {args.output}")


if __name__ == "__main__":
    main()

