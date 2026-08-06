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


class TextGenerator:
    """Lazy Hugging Face text generator used when a generative model is supplied."""

    def __init__(self, model_name: str, max_new_tokens: int = 128) -> None:
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(
                "Geração de texto exige transformers e torch. "
                "Instale com: .venv/bin/python -m pip install transformers torch"
            ) from error

        self.max_new_tokens = max_new_tokens
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map="auto",
        )

    def generate(self, message: str, context: list[tuple[str, str]]) -> str:
        history = context[-4:]
        prompt = (
            "Você é um tutor educacional cordial e responde em português do Brasil. "
            "Responda de forma clara e curta. Não invente informações.\n\n"
        )
        for user_message, assistant_message in history:
            prompt += f"Usuário: {user_message}\nTutor: {assistant_message}\n"
        prompt += f"Usuário: {message}\nTutor:"
        output = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )[0]["generated_text"]
        return str(output).strip()


def load_local_language_model(model_dir: str):
    """Load the project's own TensorFlow causal LM checkpoint."""
    import json
    import tensorflow as tf

    from phase22_language_model.engine import InferenceEngine
    from phase22_language_model.model import CausalLMConfig, DecoderOnlyCausalLM
    from phase22_language_model.tokenizer import CharacterTokenizer

    path = Path(model_dir)
    tokenizer = CharacterTokenizer.load(path / "tokenizer.json")
    config = CausalLMConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
    model = DecoderOnlyCausalLM(config)
    model(tf.zeros((1, 1), dtype=tf.int32))
    model.load_weights(path / "model.weights.h5")
    return InferenceEngine(model, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converse with the Portuguese intent assistant")
    parser.add_argument(
        "--intent-data",
        default="phase20_portuguese_nlp/data/intents.jsonl",
        help="JSONL dataset with Portuguese intent examples",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CHAT_MODEL"),
        help="modelo generativo Hugging Face; também pode usar CHAT_MODEL",
    )
    parser.add_argument("--mode", choices=("local", "intent", "external"), default="intent")
    parser.add_argument("--lm-model", default="artifacts/language_model", help="checkpoint do Causal LM próprio")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    dataset_path = Path(args.intent_data)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset de intenções não encontrado: {dataset_path}")
    texts, intents = load_intent_dataset(dataset_path)
    assistant = PortugueseAssistant().fit(texts, intents)
    local_engine = None
    if args.mode == "local":
        local_engine = load_local_language_model(args.lm_model)
    generator = TextGenerator(args.model, args.max_new_tokens) if args.mode == "external" and args.model else None

    print("Chat em português iniciado. Digite 'sair' para encerrar.")
    if local_engine:
        print(f"Causal LM local ativo: {args.lm_model}")
    elif generator:
        print(f"Geração de texto ativa: {args.model}")
    else:
        print("Modo local ativo. Use --model para gerar texto com um modelo generativo.")
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
        if local_engine:
            generated = local_engine.generate(f"Pergunta: {message}\nResposta:", max_new_tokens=args.max_new_tokens)
            response = generated.split("Resposta:", 1)[-1].strip() or "Não consegui gerar uma resposta."
        else:
            response = generator.generate(message, assistant.history[:-1]) if generator else result["response"]
        print(f"IA: {response}")
        print(f"   intenção: {result['intent']} (confiança: {result['confidence']:.2%})")
        if result["entities"]:
            entities = ", ".join(f"{item['type']}={item['text']}" for item in result["entities"])
            print(f"   entidades: {entities}")


if __name__ == "__main__":
    main()
