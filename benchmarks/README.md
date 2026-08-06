# Benchmark runner

O script [run_benchmarks.py](run_benchmarks.py) executa ou planeja a suíte de
benchmarks solicitada e gera:

- `benchmark_results/benchmark_report.json`
- `benchmark_results/benchmark_report.md`

## Listar benchmarks

```bash
.venv/bin/python benchmarks/run_benchmarks.py --list
```

## Gerar plano sem executar

```bash
.venv/bin/python benchmarks/run_benchmarks.py \
  --model meta-llama/Llama-3.1-8B-Instruct
```

## Executar com lm-evaluation-harness

Instale o harness separadamente e use um modelo de linguagem compatível:

```bash
.venv/bin/python -m pip install lm-eval
.venv/bin/python benchmarks/run_benchmarks.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --benchmark mmlu,mmlu_pro,gpqa_diamond,math,ifeval \
  --execute
```

Benchmarks marcados como `adapter externo` precisam de implementações próprias
para o formato e avaliador de cada dataset. O script não executa esses itens
silenciosamente: registra-os como `skipped` no relatório.

O modelo atual do projeto é um classificador TensorFlow, não um modelo de
linguagem generativo. Portanto, ele não deve ser passado diretamente ao
`lm-eval`; use um checkpoint causal/instruction-tuned ou um adaptador adequado.
