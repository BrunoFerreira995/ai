#!/usr/bin/env python3
"""Run lm-evaluation-harness with the project's own TensorFlow checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_ALIASES = {
    "gpqa_diamond": "gpqa",
    "aime_2024": "aime24",
    "aime_2025": "aime25",
}


def normalize_tasks(value: str) -> list[str]:
    """Convert common benchmark aliases to lm-eval task names."""
    return [
        TASK_ALIASES.get(task.strip(), task.strip())
        for task in value.split(",")
        if task.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local TensorFlow causal LM")
    parser.add_argument("--model-dir", default="artifacts/language_model")
    parser.add_argument("--tasks", default="mmlu,gpqa_diamond,aime24,bbh")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="benchmark_results/language_model")
    args = parser.parse_args()

    tasks = normalize_tasks(args.tasks)
    if not tasks:
        raise SystemExit("Informe pelo menos uma tarefa em --tasks.")

    try:
        from lm_eval import evaluator
    except ImportError as error:
        raise SystemExit("Instale lm-eval: .venv/bin/python -m pip install lm-eval") from error
    from .lm_eval_adapter import TensorFlowCausalLM

    adapter = TensorFlowCausalLM(pretrained=args.model_dir, batch_size=args.batch_size)
    try:
        results = evaluator.simple_evaluate(model=adapter, tasks=tasks, batch_size=args.batch_size)
    except Exception as error:
        message = str(error)
        if "huggingface.co" in message or "nodename nor servname" in message or "datasets" in message.lower():
            raise SystemExit(
                "Os benchmarks precisam baixar os datasets do Hugging Face. "
                "Execute novamente com acesso à internet ou use datasets já armazenados localmente."
            ) from error
        raise
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lm_eval_report.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"LM evaluation report: {path}")


if __name__ == "__main__":
    main()
