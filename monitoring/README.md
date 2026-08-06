# Phase 16 — Monitoring

Implemented monitoring capabilities:

- Structured application logging
- Population Stability Index drift detection
- Inference latency, p95, request count, and throughput
- CPU, memory, and optional NVIDIA GPU utilization
- Threshold-based automatic alerts with pluggable handlers

Run the tests:

```bash
.venv/bin/python -m unittest discover -s monitoring -p 'test_*.py'
```
