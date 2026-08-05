# Phase 12 — Model Compression

Implemented compression techniques:

- Quantization-aware training and full-integer TFLite conversion
- Polynomial-magnitude pruning and wrapper stripping
- Knowledge distillation from a teacher to a student model
- Weight clustering with configurable centroid count

Run the tests:

```bash
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m unittest discover -s phase12_model_compression -p 'test_*.py'
```
