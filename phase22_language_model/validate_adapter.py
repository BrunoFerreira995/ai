#!/usr/bin/env python3
"""Run a local smoke test for the lm-eval adapter and A/B/C/D scoring."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from .lm_eval_adapter import TensorFlowCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida o adapter local do lm-eval")
    parser.add_argument("--model-dir", default="artifacts/language_model")
    args = parser.parse_args()
    adapter = TensorFlowCausalLM(pretrained=args.model_dir)
    requests = [SimpleNamespace(args=("Responda somente com a letra correta. 2 + 2 = ", "A")),
                SimpleNamespace(args=("Responda somente com a letra correta. 2 + 2 = ", "B")),
                SimpleNamespace(args=("Responda somente com a letra correta. 2 + 2 = ", "C")),
                SimpleNamespace(args=("Responda somente com a letra correta. 2 + 2 = ", "D"))]
    scores = adapter.loglikelihood(requests)
    if len(scores) != 4 or not all(isinstance(score[0], float) and isinstance(score[1], bool) for score in scores):
        raise SystemExit("Adapter inválido: loglikelihood não retornou (score, greedy) para A/B/C/D")
    best = max(range(4), key=lambda index: scores[index][0])
    print("Adapter: OK")
    print("Scores A/B/C/D:", [round(score[0], 4) for score in scores])
    print("Alternativa mais provável:", "ABCD"[best])


if __name__ == "__main__":
    main()
