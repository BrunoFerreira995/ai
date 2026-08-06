#!/usr/bin/env python3
"""Run and report a suite of language-model benchmarks.

By default this script performs a dry run. Use ``--execute`` only after the
selected model backend and benchmark dependencies have been installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Benchmark:
    slug: str
    category: str
    name: str
    measures: str
    harness_task: str | None = None
    note: str = ""


BENCHMARKS = [
    Benchmark("mmlu", "Conhecimento geral", "MMLU", "Conhecimento em dezenas de disciplinas", "mmlu"),
    Benchmark("mmlu_pro", "Conhecimento avançado", "MMLU-Pro", "Versão mais difícil do MMLU", "mmlu_pro"),
    Benchmark("gpqa_diamond", "Raciocínio científico", "GPQA Diamond", "Física, química e biologia de pós-graduação", "gpqa"),
    Benchmark("math", "Matemática", "MATH", "Problemas matemáticos", "math"),
    Benchmark("aime_2024", "Olimpíadas", "AIME 2024", "Olimpíadas americanas de matemática", "aime24"),
    Benchmark("aime_2025", "Olimpíadas", "AIME 2025", "Olimpíadas americanas de matemática", "aime25"),
    Benchmark("bbh", "Raciocínio", "BBH (BigBench Hard)", "Problemas difíceis de lógica", "bbh"),
    Benchmark("zebralogic", "Lógica", "ZebraLogic", "Problemas tipo Zebra Puzzle", note="adapter externo"),
    Benchmark("humaneval_plus", "Programação", "HumanEval+", "Escrever código correto", note="adapter externo"),
    Benchmark("mbpp_plus", "Programação", "MBPP+", "Problemas básicos de programação", note="adapter externo"),
    Benchmark("livecodebench", "Programação", "LiveCodeBench", "Problemas recentes de programação", note="adapter externo"),
    Benchmark("ifeval", "Seguimento de instruções", "IFEval", "Seguir instruções complexas", "ifeval"),
    Benchmark("ifbench", "Seguimento de instruções", "IFBench", "Avaliação adicional de instruções", note="adapter externo"),
    Benchmark("simpleqa", "Perguntas gerais", "SimpleQA", "Perguntas objetivas", "simpleqa"),
    Benchmark("popqa", "Conhecimento popular", "PopQA", "Conhecimento factual", "popqa"),
    Benchmark("agieval", "AGI", "AGI Eval", "Questões de exames diversos", "agieval"),
    Benchmark("safety", "Segurança", "Safety", "Resistência a solicitações inseguras", note="adapter externo"),
]


def find_benchmarks(selection: str) -> list[Benchmark]:
    if selection == "all":
        return BENCHMARKS
    wanted = {item.strip() for item in selection.split(",")}
    selected = [benchmark for benchmark in BENCHMARKS if benchmark.slug in wanted]
    unknown = wanted - {benchmark.slug for benchmark in BENCHMARKS}
    if unknown:
        raise ValueError(f"benchmarks desconhecidos: {', '.join(sorted(unknown))}")
    return selected


def build_command(benchmark: Benchmark, args: argparse.Namespace, output: Path) -> list[str] | None:
    if not benchmark.harness_task:
        return None
    model_args = args.model_args or f"pretrained={args.model}"
    return [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        args.backend,
        "--model_args",
        model_args,
        "--tasks",
        benchmark.harness_task,
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(output),
    ]


def write_report(results: list[dict[str, object]], output_dir: Path, model: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [f"# Benchmark report\n\nModel: `{model}`\n", "| Benchmark | Status | Score/source |", "| --- | --- | --- |"]
    for result in results:
        lines.append(f"| {result['name']} | {result['status']} | {result.get('result', result.get('note', ''))} |")
    md_path = output_dir / "benchmark_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def is_tensorflow_classifier(model: str) -> bool:
    """Return whether ``model`` points to this project's SavedModel classifier."""
    model_path = Path(model)
    return model_path.is_dir() and (model_path / "saved_model.pb").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run language-model benchmarks and create a report")
    parser.add_argument("--model", default="MODEL_NAME", help="Hugging Face model name or backend model identifier")
    parser.add_argument("--backend", default="hf", help="lm-eval backend, usually hf")
    parser.add_argument("--model-args", help="raw lm-eval model_args, e.g. pretrained=...,dtype=float16")
    parser.add_argument("--benchmark", default="all", help="all or comma-separated benchmark slugs")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--execute", action="store_true", help="actually execute lm-eval commands")
    parser.add_argument("--list", action="store_true", help="list available benchmarks")
    args = parser.parse_args()

    if args.list:
        for benchmark in BENCHMARKS:
            runner = benchmark.harness_task or benchmark.note
            print(f"{benchmark.slug:20} {benchmark.name:20} [{runner}]")
        return 0

    selected = find_benchmarks(args.benchmark)
    output_dir = Path(args.output_dir)
    results = []
    classifier_model = is_tensorflow_classifier(args.model)
    for benchmark in selected:
        result_dir = output_dir / benchmark.slug
        command = build_command(benchmark, args, result_dir)
        result = asdict(benchmark)
        if classifier_model:
            result.update(
                status="not_applicable",
                note="modelo TensorFlow classificador; benchmark exige geração de texto",
            )
            print(f"N/A {benchmark.name}: classificador numérico não gera respostas de texto")
        elif command is None:
            result.update(status="skipped", note="requires an adapter or dedicated evaluator")
            print(f"SKIP {benchmark.name}: adapter externo necessário")
        elif not args.execute:
            result.update(status="planned", command=command)
            print("PLAN", " ".join(command))
        elif shutil.which(command[0]) is None:
            result.update(status="failed", note="Python executable not found")
        else:
            result_dir.mkdir(parents=True, exist_ok=True)
            print("RUN", benchmark.name)
            completed = subprocess.run(command, check=False)
            result.update(status="passed" if completed.returncode == 0 else "failed", returncode=completed.returncode, result=str(result_dir))
        results.append(result)

    json_path, md_path = write_report(results, output_dir, args.model)
    print(f"Reports: {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"Erro: {error}", file=sys.stderr)
        raise SystemExit(2)
