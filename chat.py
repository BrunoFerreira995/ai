#!/usr/bin/env python3
"""Interactive terminal chat for the trained classifier."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.prefix == sys.base_prefix:
    venv_python = Path(__file__).resolve().parent / ".venv/bin/python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])

from phase20_portuguese_nlp.assistant import PortugueseAssistant, load_intent_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Converse with the Portuguese intent assistant")
    parser.add_argument(
        "--intent-data",
        default="phase20_portuguese_nlp/data/intents.jsonl",
        help="JSONL dataset with Portuguese intent examples",
    )
    args = parser.parse_args()

    dataset_path = Path(args.intent_data)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset de intenções não encontrado: {dataset_path}")
    texts, intents = load_intent_dataset(dataset_path)
    assistant = PortugueseAssistant().fit(texts, intents)

    print("Chat em português iniciado. Digite 'sair' para encerrar.")
    while True:
        try:
            message = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat encerrado.")
            break
        if message.lower() in {"sair", "exit", "quit"}:
            print("Chat encerrado.")
            break
        if not message:
            continue

        result = assistant.respond(message)
        print(f"IA: {result['response']}")
        print(f"   intenção: {result['intent']} (confiança: {result['confidence']:.2%})")
        if result["entities"]:
            entities = ", ".join(f"{item['type']}={item['text']}" for item in result["entities"])
            print(f"   entidades: {entities}")


if __name__ == "__main__":
    main()
