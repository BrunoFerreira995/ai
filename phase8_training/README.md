# Phase 8 — Training

Training utilities included in this module:

- Batch training and validation through `model.fit`
- Exponential learning-rate schedule
- Mixed precision policy configuration
- Gradient clipping through optimizer `clipnorm`
- Early stopping, learning-rate reduction, model checkpointing, and TensorBoard
- GPU-aware distributed strategy creation

Run the tests:

```bash
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m unittest discover -s phase8_training -p 'test_*.py'
```
