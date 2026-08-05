# Phase 13 — Export

Export utilities for:

- TensorFlow SavedModel
- TensorFlow Lite, including optional INT8 quantization
- ONNX through tf2onnx
- TensorRT on Linux with NVIDIA TensorRT installed
- Core ML `.mlpackage` through coremltools

Run the tests:

```bash
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m unittest discover -s phase13_export -p 'test_*.py'
```
