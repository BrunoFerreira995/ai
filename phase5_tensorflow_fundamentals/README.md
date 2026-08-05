# Phase 5 — TensorFlow Fundamentals

This module demonstrates:

- `tf.Tensor`, tensor operations, shapes, reductions, and broadcasting
- Mutable trainable `tf.Variable` values
- Eager execution and graph execution through `tf.function`
- `tf.data.Dataset` batching, shuffling, caching, prefetching, and parallel mapping
- Parallel file loading with `tf.io.read_file`

Run the tests from the project root:

```bash
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m unittest discover -s phase5_tensorflow_fundamentals -p 'test_*.py'
```
