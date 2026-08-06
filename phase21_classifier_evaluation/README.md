# Phase 21 — Avaliação do Classificador

Esta fase avalia o `artifacts/saved_model` como classificador numérico. Ela
calcula métricas de classificação, calibração, robustez, OOD, latência,
throughput, memória Python, batch sizes e profiling opcional do TensorFlow.

Execute:

```bash
.venv/bin/python -m phase21_classifier_evaluation.evaluate_model
```

O relatório é salvo em `benchmark_results/classifier/classifier_report.json`.
Para gerar um trace do TensorBoard:

```bash
.venv/bin/python -m phase21_classifier_evaluation.evaluate_model --profile
```
